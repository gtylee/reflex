from datetime import datetime, timedelta, timezone
import gzip

import pytest
from pydantic import ValidationError

from reflex.core.models import Event, LensCandidate, Observation, CandidateStatus, stable_id
from reflex.core.storage import EventStore, StateStore
from reflex.core.lenses import DEFAULT_SQL, SqlLens
from reflex.core.policy import BoundedTools
from reflex.core.ingestion import read_bpi2017, read_loghub


def make_event(event_id: str, at: datetime, amount: float = 100.0) -> Event:
    return Event(event_id=event_id, event_type="review", event_time=at, entity_ids=("acct-1",),
                 attributes={"amount": amount, "nested": {"tags": ["a"]}}, source="test")


def test_event_is_deeply_immutable_and_requires_timezone():
    event = make_event("e1", datetime(2025, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(TypeError):
        event.attributes["amount"] = 2
    with pytest.raises(TypeError):
        event.attributes["nested"]["key"] = "value"
    with pytest.raises(ValidationError):
        Event(event_type="x", event_time=datetime(2025, 1, 1))


def test_duckdb_event_replay_is_ordered_idempotent_and_immutable(tmp_path):
    store = EventStore(tmp_path / "events.duckdb")
    first = make_event("e1", datetime(2025, 1, 2, tzinfo=timezone.utc), 5000)
    second = make_event("e2", datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert store.add_events([first, second]) == 2
    assert store.add_events([first]) == 0
    assert [row.event_id for row in store.events()] == ["e2", "e1"]
    with pytest.raises(ValueError, match="different content"):
        store.add_events([make_event("e1", first.event_time, 10)])
    store.close()


def test_sql_lens_replay_supports_state_and_rejects_external_sql(tmp_path):
    store = EventStore(tmp_path / "events.duckdb")
    store.add_events([
        make_event("low", datetime(2025, 1, 1, tzinfo=timezone.utc), 100),
        make_event("high", datetime(2025, 1, 2, tzinfo=timezone.utc), 6000),
    ])
    lens = SqlLens("sql.high_amount", "1.0", "threshold", DEFAULT_SQL)
    rows = lens.replay(store, start=datetime(2025, 1, 2, tzinfo=timezone.utc), state_snapshot={"baseline": 1})
    assert [row.supporting_event_ids for row in rows] == [("high",)]
    assert rows[0].state_snapshot["baseline"] == 1
    with pytest.raises(ValueError, match="external"):
        store.query_sql_lens("SELECT * FROM read_csv_auto('/tmp/private.csv')")
    with pytest.raises(ValueError, match="one SELECT"):
        store.query_sql_lens("DELETE FROM events")
    store.close()


def test_state_versions_audit_and_budget_limits(tmp_path):
    state = StateStore(tmp_path / "state.sqlite")
    assert state.set("operational", "count", {"value": 1}, "initial") == 1
    assert state.set("operational", "count", {"value": 2}, "advance") == 2
    assert state.get("operational", "count") == {"value": 2}
    assert [row["to_version"] for row in state.audit("operational")] == [1, 2]
    state.initialize_budget("run", max_investigations=1, max_queries=2)
    assert state.consume_budget("run", "investigation")
    assert not state.consume_budget("run", "investigation")
    assert state.consume_budget("run", "query", 2)
    assert not state.consume_budget("run", "query")
    state.close()


def test_historical_state_snapshot_skips_future_rolling_state(tmp_path):
    state = StateStore(tmp_path / "state.sqlite")
    state.set("operational", "baselines", {"mean": 2}, "training baseline")
    state.set("operational", "rolling_metrics", {"as_of": "2025-01-05T00:00:00+00:00", "count": 5}, "batch 1")
    state.set("operational", "rolling_metrics", {"as_of": "2025-01-10T00:00:00+00:00", "count": 9}, "batch 2")
    state.set("operational", "rolling_metrics", {"as_of": "2025-01-03T00:00:00+00:00", "count": 2}, "historical replay")
    snapshot = state.snapshot_at("operational", datetime(2025, 1, 7, tzinfo=timezone.utc))
    assert snapshot["baselines"] == {"mean": 2}
    assert snapshot["rolling_metrics"]["count"] == 5
    state.close()


def test_candidate_lifecycle_requires_validation_and_explicit_approval(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    validation = datetime(2025, 1, 2, tzinfo=timezone.utc)
    state = StateStore(tmp_path / "state.sqlite")
    event_store = EventStore(tmp_path / "events.duckdb")
    event_store.add_events([make_event("later", validation, 5000)])
    candidate = LensCandidate(lens_id="candidate.large", version="1", sql=DEFAULT_SQL,
        description="Large observed amount", applicability="all events", training_boundary=start,
        validation_boundary=validation, expected_outputs={"columns": ["event_id"]})
    state.put_candidate(candidate)
    state.initialize_budget("candidate-validation", max_investigations=0, max_queries=5)
    tools = BoundedTools(event_store, state, run_id="candidate-validation")
    result = tools.validate_lens(candidate.candidate_id, event_store, validation_start=validation,
                                 validation_end=validation + timedelta(days=1))
    assert result["valid_sql"] and result["output_count"] == 1
    assert state.transition_candidate(candidate.candidate_id, CandidateStatus.APPROVED.value).status == CandidateStatus.APPROVED
    assert state.transition_candidate(candidate.candidate_id, CandidateStatus.ACTIVE.value).status == CandidateStatus.ACTIVE
    with pytest.raises(ValueError, match="invalid candidate transition"):
        state.transition_candidate(candidate.candidate_id, CandidateStatus.PROPOSED.value)
    state.close()
    event_store.close()


def test_candidate_validation_detects_active_lens_redundancy(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    validation = datetime(2025, 1, 2, tzinfo=timezone.utc)
    state = StateStore(tmp_path / "state.sqlite")
    state.set("lens", "sql.high_amount", {"status": "active", "sql": DEFAULT_SQL}, "register SQL lens")
    event_store = EventStore(tmp_path / "events.duckdb")
    event_store.add_events([make_event("later", validation, 5000)])
    candidate = LensCandidate(lens_id="candidate.redundant", version="1", sql=DEFAULT_SQL,
        description="Duplicate", applicability="all", training_boundary=start, validation_boundary=validation)
    state.put_candidate(candidate)
    state.initialize_budget("redundant-validation", max_investigations=0, max_queries=5)
    tools = BoundedTools(event_store, state, run_id="redundant-validation")
    result = tools.validate_lens(candidate.candidate_id, event_store, validation_start=validation,
                                 validation_end=validation + timedelta(days=2))
    assert result["redundancy_jaccard"] == 1.0
    assert result["false_positive_rate"] is None
    assert state.get("lens", f"candidate:{candidate.candidate_id}")["status"] == "rejected"
    state.close()
    event_store.close()


def test_observation_replay_id_is_stable():
    first = stable_id("lens", "1", "acct-1", "window")
    assert first == stable_id("lens", "1", "acct-1", "window")
    observation = Observation(observation_id=first, observation_type="score", lens_id="lens", lens_version="1",
        subject="acct-1", window_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2025, 1, 2, tzinfo=timezone.utc), value={"score": 1})
    assert Observation.model_validate_json(observation.model_dump_json()) == observation


def test_bpi_xes_gzip_and_loghub_hdfs_adapters(tmp_path):
    xes = '''<?xml version="1.0"?><log xmlns="http://www.xes-standard.org/">
      <trace><string key="concept:name" value="case-1"/><event>
        <string key="concept:name" value="A_SUBMITTED"/><date key="time:timestamp" value="2016-01-01T10:00:00+01:00"/>
        <string key="org:resource" value="user-4"/><int key="amount" value="500"/>
      </event></trace></log>'''
    archive = tmp_path / "bpi.xes.gz"
    with gzip.open(archive, "wb") as file:
        file.write(xes.encode())
    bpi_events = list(read_bpi2017(archive))
    assert bpi_events[0].entity_ids == ("case-1",)
    assert bpi_events[0].event_type == "A_SUBMITTED"
    assert bpi_events[0].attributes["amount"] == 500
    assert bpi_events[0].event_time.hour == 9

    log_path = tmp_path / "HDFS_2k.log"
    log_path.write_text("081109 203518 143 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1 terminating\n")
    log_events = list(read_loghub(log_path))
    assert log_events[0].entity_ids == ("143",)
    assert log_events[0].event_type == "INFO"
