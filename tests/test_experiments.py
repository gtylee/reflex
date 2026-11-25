import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from reflex.core.models import Event, LensCandidate
from reflex.core.storage import EventStore, StateStore
from reflex.experiments.adapters import (AutoencoderAdapter, CompositeLensAdapter, CorrelationShiftAdapter,
                                         GaussianMixtureAdapter, IsolationForestAdapter,
                                         QuantileLensAdapter, SequencePatternAdapter,
                                         default_phase_one_adapters)
from reflex.experiments.datasets import (GDELT_EVENT_COLUMNS, adapt_fnspid_rows, adapt_gdelt_rows,
                                         adapt_market_price_rows, parse_reuters21578)
from reflex.experiments import datasets as dataset_module
from reflex.experiments.generators import SCENARIOS, default_configs, generate_scenario
from reflex.experiments.heldout import run_heldout_benchmark, write_heldout_report
from reflex.experiments.harnesses import ElizaExperimentAdapter, MockElizaHarness
from reflex.experiments.metrics import decision_variance, mean_interval, score_predictions
from reflex.experiments.models import AdapterOutput, GroundTruthLabel, RunBudget
from reflex.experiments.performance import benchmark_event_store
from reflex.experiments import public_replay
from reflex.experiments.runner import repeat_adapter, run_comparison
from reflex.experiments.visualizations import write_metric_svg, write_news_timeline_svg
from reflex.experiments.workflows import (PortfolioExposure, RepairSandbox, SimulatedOperationsSystem,
                                           diagnose_incident,
                                           evaluate_topic_association, process_news_events,
                                           process_news_timeline,
                                           state_dependent_observation,
                                           synthetic_feedback, synthetic_incidents,
                                           synthetic_news_outcomes, synthetic_news_story,
                                           validate_candidate)


def config_for(name, *, seed=11, days=30, entities=32):
    return next(item for item in default_configs(seed=seed, days=days, entities=entities)
                if item.scenario == name)


def test_generators_are_seeded_and_ground_truth_is_separate(tmp_path):
    for name in SCENARIOS:
        first = generate_scenario(config_for(name))
        again = generate_scenario(config_for(name))
        changed = generate_scenario(config_for(name, seed=19))
        assert first.input_hash == again.input_hash
        assert [event.event_id for event in first.events] == [event.event_id for event in again.events]
        assert first.input_hash != changed.input_hash
        assert first.ground_truth
        assert "mechanism" not in first.agent_input().model_dump(mode="json")
        assert all("mechanism" not in event.attributes for event in first.events)
    case = generate_scenario(config_for("exposure_drift"))
    paths = case.write(tmp_path / "agent", tmp_path / "evaluator")
    public_text = (tmp_path / "agent" / "events.jsonl").read_text() + (tmp_path / "agent" / "experiment.json").read_text()
    assert "gradual_exposure_drift" not in public_text
    assert "mechanism" not in public_text
    assert json.loads((tmp_path / "evaluator" / "ground_truth.json").read_text())
    assert paths["input_hash"] == case.input_hash


def test_phase_one_scenario_invariants():
    exposure = generate_scenario(config_for("exposure_drift"))
    assert max(event.attributes["exposure"] for event in exposure.events) < 0.95
    collateral = generate_scenario(config_for("collateral_deterioration"))
    target = [e for e in collateral.events if e.entity_ids == ("acct-0000",)]
    assert target[-1].attributes["coverage"] < target[0].attributes["coverage"]
    rare = generate_scenario(config_for("rare_sequence"))
    target_events = [e for e in rare.events if e.entity_ids == ("acct-0000",)]
    event_types = [e.event_type for e in target_events]
    assert "settlement.file_received" in event_types and "settlement.failed" in event_types
    assert {e.attributes["sequence_score"] for e in target_events} == {0.0}
    assert all(stage in event_types for stage in (
        "settlement.file_received", "reconciliation.flagged",
        "counterparty.confirmation_missing", "settlement.failed"))
    sequence = default_phase_one_adapters(rare.config)["C"]
    sequence_output = sequence.run(rare.agent_input(), RunBudget(), trial_seed=rare.config.seed)
    assert set(sequence_output.predicted_event_ids) == {
        label.event_id for label in rare.ground_truth if label.event_time >= rare.calibration_end
    }
    correlation = generate_scenario(config_for("correlation_shift", days=60))
    pre = [e for e in correlation.events if e.event_time < correlation.train_end]
    post = [e for e in correlation.events if e.event_time >= correlation.calibration_end]
    assert np.corrcoef([e.attributes["factor_x"] for e in pre], [e.attributes["factor_y"] for e in pre])[0, 1] > 0.5
    assert np.corrcoef([e.attributes["factor_x"] for e in post], [e.attributes["factor_y"] for e in post])[0, 1] < -0.5
    correlation_truth = [label for label in correlation.ground_truth
                         if label.event_time >= correlation.calibration_end]
    assert correlation_truth and all(label.event_time >= correlation.calibration_end
                                     for label in correlation.ground_truth)
    novel = generate_scenario(config_for("unseen_mechanism"))
    labeled = {item.event_id for item in novel.ground_truth}
    assert all(next(e for e in novel.events if e.event_id == item).event_time >= novel.start + timedelta(days=15)
               for item in labeled)


def test_metrics_bootstrap_and_variance_are_deterministic():
    labels = [GroundTruthLabel(event_id="hit", entity_id="x", mechanism="m",
                               event_time=datetime.now(timezone.utc))]
    first = score_predictions(["hit", "false"], labels, {"hit": ("hit",)},
                              universe_event_ids=["hit", "false", "miss"])
    again = score_predictions(["hit", "false"], labels, {"hit": ("hit",)},
                              universe_event_ids=["hit", "false", "miss"])
    assert first == again and first["true_positives"] == 1 and first["false_positives"] == 1
    assert first["evidence_completeness"] == 0.5
    assert first["f1_interval_95"]["lower"] <= first["f1"] <= first["f1_interval_95"]["upper"]
    assert mean_interval([1.0, 2.0, 3.0]) == mean_interval([1.0, 2.0, 3.0])
    assert decision_variance(["approve", "reject", "approve"])["disagreement_rate"] > 0


def test_event_level_evidence_scoring_rejects_future_and_cross_entity_refs():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events = {
        "target": Event(event_id="target", event_type="signal", event_time=base + timedelta(hours=1),
                        entity_ids=("account-a",)),
        "prior": Event(event_id="prior", event_type="context", event_time=base,
                       entity_ids=("account-a",)),
        "future": Event(event_id="future", event_type="context", event_time=base + timedelta(hours=2),
                        entity_ids=("account-a",)),
        "other": Event(event_id="other", event_type="context", event_time=base,
                       entity_ids=("account-b",)),
    }
    truth = [GroundTruthLabel(event_id="target", entity_id="account-a", mechanism="drift",
                              event_time=base + timedelta(hours=1))]
    metrics = score_predictions(
        ["target"], truth,
        {"target": ("target", "prior", "future", "other", "missing")},
        universe_event_ids=events,
        events_by_id=events,
    )
    assert metrics["precision"] == metrics["recall"] == 1.0
    assert metrics["evidence_reference_integrity"] == 0.8
    assert metrics["evidence_temporal_validity"] == 0.6
    assert metrics["evidence_entity_alignment"] == 0.6
    assert metrics["evidence_integrity"] == 0.4
    assert metrics["evidence_target_inclusion"] == 1.0
    assert metrics["evidence_supported_prediction_rate"] == 0.0


def test_heldout_benchmark_writes_seed_level_report(tmp_path):
    report = run_heldout_benchmark(first_seed=101, seed_count=2, entities=8, days=30,
                                   scenarios=("rare_sequence",))
    scenario = report["scenarios"]["rare_sequence"]
    assert report["protocol"]["seeds"] == [101, 102]
    assert report["protocol"]["expert_labeled"] is False
    assert scenario["completed_seed_count"] == 2
    assert scenario["per_seed_metric_summary"]["f1"]["mean"] == 1.0
    assert scenario["per_seed_metric_summary"]["evidence_integrity"]["mean"] == 1.0
    json_path, markdown_path = write_heldout_report(report, tmp_path / "heldout")
    assert json.loads(json_path.read_text())["scenarios"]["rare_sequence"]["completed_seed_count"] == 2
    assert "Labels are held-out synthetic" in markdown_path.read_text()


class SameInputMock:
    execution_kind = "mock"

    def __init__(self, variant, seen):
        self.variant_id = variant
        self.seen = seen

    def run(self, experiment, budgets, *, trial_seed):
        self.seen.append((self.variant_id, experiment.input_hash, budgets, trial_seed,
                          hasattr(experiment, "scenario"), hasattr(experiment, "ground_truth")))
        return AdapterOutput()


def test_comparison_uses_shared_label_blind_input_and_marks_mock(tmp_path):
    config = config_for("exposure_drift")
    dataset = generate_scenario(config)
    seen = []
    report = run_comparison(config, dataset,
                            {"A": SameInputMock("A", seen), "B": SameInputMock("B", seen)},
                            variants=("A", "B"))
    assert len({item[1] for item in seen}) == 1
    assert len({id(item[2]) for item in seen}) == 1
    assert all(not item[4] and not item[5] for item in seen)
    assert all(run.execution_kind == "mock" for run in report.runs)
    assert report.comparison_ready is False
    absent = run_comparison(config, dataset, variants=("A", "B"))
    assert all(run.status == "adapter_unavailable" for run in absent.runs)

    class OverBudget:
        variant_id = "A"
        execution_kind = "external"

        def run(self, experiment, budgets, *, trial_seed):
            return AdapterOutput(tool_calls=budgets.tool_calls + 1)

    over = run_comparison(config, dataset, {"A": OverBudget()}, variants=("A",))
    assert over.runs[0].status == "over_budget"
    try:
        run_comparison(config, dataset, variants=("A", "A"))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate variants must be rejected")

    class RecordedExternal:
        execution_kind = "external"

        def __init__(self, variant, metadata):
            self.variant_id, self.metadata = variant, metadata

        def run(self, experiment, budgets, *, trial_seed):
            return AdapterOutput(metadata=self.metadata)

    no_provenance = run_comparison(config, dataset,
                                   {"A": RecordedExternal("A", {}), "B": RecordedExternal("B", {})},
                                   variants=("A", "B"))
    assert no_provenance.comparison_ready is False
    assert any("model_version" in reason for reason in no_provenance.readiness_reasons)
    shared = {"model_version": "model-x", "sampling_parameters": {"temperature": 0}}
    ready = run_comparison(config, dataset,
                           {"A": RecordedExternal("A", shared),
                            "B": RecordedExternal("B", {**shared, "tool_versions": {"detector": "1"}})},
                           variants=("A", "B"))
    assert ready.comparison_ready is True
    metric_chart = write_metric_svg(ready, tmp_path / "comparison.svg")
    assert "comparison ready: true" in metric_chart.read_text()


def test_phase_one_detectors_return_only_known_ids():
    config = config_for("exposure_drift", entities=24, days=30)
    dataset = generate_scenario(config)
    input_ids = {e.event_id for e in dataset.events}
    for adapter in (QuantileLensAdapter("exposure"), IsolationForestAdapter(("exposure",)),
                    GaussianMixtureAdapter(("exposure",), components=1), AutoencoderAdapter(("exposure",), epochs=3)):
        output = adapter.run(dataset.agent_input(), RunBudget(), trial_seed=5)
        assert set(output.predicted_event_ids) <= input_ids
    combined = CompositeLensAdapter((QuantileLensAdapter("exposure"),
                                     IsolationForestAdapter(("exposure",))), strategy="any")
    combined_output = combined.run(dataset.agent_input(), RunBudget(), trial_seed=5)
    assert set(combined_output.predicted_event_ids) <= input_ids
    assert combined_output.metadata["detector"] == "composite"
    correlation_case = generate_scenario(config_for("correlation_shift", days=60, entities=16))
    corr = CorrelationShiftAdapter(window=12).run(correlation_case.agent_input(), RunBudget(), trial_seed=5)
    assert set(corr.predicted_event_ids) <= {e.event_id for e in correlation_case.events}
    seq_case = generate_scenario(config_for("rare_sequence"))
    seq = SequencePatternAdapter(("settlement.file_received", "reconciliation.flagged",
                                  "counterparty.confirmation_missing", "settlement.failed"))
    seq_output = seq.run(seq_case.agent_input(), RunBudget(), trial_seed=11)
    assert set(seq_output.predicted_event_ids) <= {e.event_id for e in seq_case.events}


def test_state_restart_event_replay_and_tenant_isolation(tmp_path):
    state_path = tmp_path / "state.sqlite"
    events_path = tmp_path / "events.duckdb"
    state = StateStore(state_path)
    state.set("operational", "portfolio-a:incident-1", {"status": "investigating", "as_of": "2025-01-01T00:00:00+00:00"}, "begin")
    state.set("operational", "portfolio-b:incident-1", {"status": "new"}, "private")
    state.close()
    reopened = StateStore(state_path)
    assert reopened.get("operational", "portfolio-a:incident-1")["status"] == "investigating"
    assert reopened.get("operational", "portfolio-b:incident-1")["status"] == "new"
    dated = datetime(2025, 1, 1, tzinfo=timezone.utc)
    reopened.set("knowledge", "relationship", {"value": "candidate", "as_of": dated.isoformat()}, "initial")
    reopened.set("knowledge", "relationship", {"value": "superseded", "as_of": (dated + timedelta(days=1)).isoformat()}, "revision")
    assert reopened.snapshot_at("knowledge", dated)["relationship"]["value"] == "candidate"
    same_event = Event(event_id="same", event_type="exposure", event_time=dated,
                       entity_ids=("acct-1",), attributes={"exposure": 0.4})
    history = [Event(event_id=f"hist-{i}", event_type="exposure", event_time=dated + timedelta(days=i),
                     entity_ids=("acct-1",), attributes={"exposure": value})
               for i, value in enumerate((0.2, 0.25, 0.3))]
    assert state_dependent_observation(same_event, history)["action"] == "investigate"
    assert state_dependent_observation(same_event, [])["action"] == "record"
    store = EventStore(events_path)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    late = Event(event_id="late", event_type="mark", event_time=base + timedelta(hours=2))
    early = Event(event_id="early", event_type="mark", event_time=base)
    assert store.add_events([late, early]) == 2
    assert store.add_events([late, early]) == 0
    assert [item.event_id for item in store.events()] == ["early", "late"]
    store.close()
    reopened.close()


def test_news_story_dedup_confirmation_retraction_and_outcome_chronology(tmp_path):
    story = synthetic_news_story(3)
    reactions, state = process_news_events(story, PortfolioExposure("p1", {"rare_earths": 0.32}))
    by_event = {row["event_id"]: row for row in reactions}
    assert by_event[story[1].event_id]["action"] == "duplicate_suppressed"
    assert by_event[story[4].event_id]["action"] == "notify_portfolio_manager"
    assert by_event[story[-1].event_id]["action"] == "reopen_and_reassess"
    repeated, next_state = process_news_events((story[4],), PortfolioExposure("p1", {"rare_earths": 0.32}), state)
    assert repeated[0]["action"] == "duplicate_suppressed"
    assert next_state["notification_count"] == state["notification_count"]
    feedback = synthetic_feedback()
    assert {e.event_type for e in feedback} >= {"pm.acknowledgement", "pm.disagreement", "portfolio.change"}
    timeline_reactions, timeline_state = process_news_timeline(
        story, synthetic_feedback(3, count=6),
        PortfolioExposure("p1", {"rare_earths": 0.32, "materials_equity": 0.18,
                                  "industrial_metals": 0.12}),
    )
    timeline_by_event = {row["event_id"]: row for row in timeline_reactions}
    assert timeline_by_event[story[0].event_id]["action"] == "notify_portfolio_manager"
    assert timeline_by_event[story[4].event_id]["action"] == "investigate"
    assert timeline_by_event[story[-1].event_id]["action"] == "reopen_and_reassess"
    assert timeline_state["feedback_version"] == 6
    outcomes = synthetic_news_outcomes(events=240)
    assert all("regime" not in event.attributes for event in outcomes)
    evaluation = evaluate_topic_association(outcomes)
    assert evaluation["train_count"] + evaluation["test_count"] == 240
    assert "not a causal" in evaluation["evaluation"]
    report_path = tmp_path / "news-smoke.json"
    report_path.write_text(json.dumps({"story_reactions": reactions}))
    chart_text = write_news_timeline_svg(report_path, report_path.with_suffix(".svg")).read_text()
    assert "reopen" in chart_text and "reassess" in chart_text


def test_feedback_changes_only_news_events_that_follow_it():
    story = synthetic_news_story(7)
    portfolio = PortfolioExposure("p1", {"rare_earths": 0.32, "materials_equity": 0.18})
    reactions, state = process_news_timeline(story, synthetic_feedback(7, count=6), portfolio)
    by_event = {row["event_id"]: row for row in reactions}
    assert by_event[story[0].event_id]["action"] == "notify_portfolio_manager"
    assert by_event[story[4].event_id]["action"] == "investigate"
    assert by_event[story[-1].event_id]["action"] == "reopen_and_reassess"
    assert state["feedback_version"] == 6
    assert state["feedback_counts"]["notification.false_positive"] == 1
    assert "rare_earths_exposure" in state


def test_operational_hidden_causes_and_sandbox_controls():
    data = synthetic_incidents(4, count=28)
    assert all("cause" not in event.attributes for event in data.events)
    system = SimulatedOperationsSystem.create(data.causes_by_event)
    sandbox = RepairSandbox.create(system)
    incident = data.events[0]
    cause = data.causes_by_event[incident.event_id]
    assert diagnose_incident(incident) == cause
    awaiting = sandbox.repair(incident, diagnosis=cause, proposed_action=f"sandbox:{cause}", approved=False)
    assert awaiting["status"] == "awaiting_approval"
    approved = sandbox.repair(incident, diagnosis=cause, proposed_action=f"sandbox:{cause}", approved=True)
    assert approved["status"] == "verified"
    assert approved["residual_diagnosis"] is None
    assert approved["symptoms_before"] != approved["symptoms_after"]
    wrong_incident = data.events[1]
    wrong = sandbox.repair(wrong_incident, diagnosis="wrong", proposed_action="sandbox:wrong", approved=True)
    assert wrong["status"] == "validation_failed"
    assert wrong["residual_diagnosis"] == data.causes_by_event[wrong_incident.event_id]
    other = data.events[2]
    partial = sandbox.repair(other, diagnosis=data.causes_by_event[other.event_id], proposed_action="fix",
                             approved=True, external_result="partial")
    assert partial["status"] == "rolled_back" and partial["rolled_back"]
    timeout_event = data.events[3]
    timeout = sandbox.repair(timeout_event, diagnosis=data.causes_by_event[timeout_event.event_id],
                             proposed_action="fix", approved=True,
                             external_result="timeout", retries=1)
    assert timeout["status"] == "escalated_timeout" and timeout["attempts"] == 2
    assert sandbox.repair(incident, diagnosis="wrong", proposed_action="retry", approved=True)["status"] == "verified"
    assert len(sandbox.ledger) == 4  # replay did not repeat the completed external side effect


def test_simulated_repair_postconditions_cover_every_cause():
    data = synthetic_incidents(21, count=40)
    sandbox = RepairSandbox.create(SimulatedOperationsSystem.create(data.causes_by_event))
    for incident in data.events:
        diagnosis = diagnose_incident(incident)
        assert diagnosis == data.causes_by_event[incident.event_id]
        result = sandbox.repair(incident, diagnosis=diagnosis,
                                proposed_action=f"sandbox:{diagnosis}", approved=True)
        assert result["status"] == "verified"
        assert result["residual_diagnosis"] is None


def test_curation_validation_and_mock_eliza_contract():
    accepted = validate_candidate("drift", "value", [0.0, 0.2], [0.1, 0.9, 0.95], 0.8,
                                 [False, True, True])
    rejected = validate_candidate("spurious", "confounder", [0, 1], [0.1, 0.9, 0.95], 0.8,
                                  [False, False, False])
    assert accepted.status == "active" and accepted.validation["f1"] == 1.0
    assert rejected.status == "rejected"
    from reflex.core.models import CandidateStatus
    from datetime import datetime
    candidate = LensCandidate(lens_id="drift", version="1", sql="SELECT 1", description="test",
                              applicability="held-out drift", training_boundary=datetime(2025, 1, 1, tzinfo=timezone.utc),
                              validation_boundary=datetime(2025, 1, 2, tzinfo=timezone.utc),
                              status=CandidateStatus.PROPOSED)
    store = StateStore(":memory:")
    store.put_candidate(candidate)
    for next_status in ("testing", "validated", "approved", "active", "retired"):
        candidate = store.transition_candidate(candidate.candidate_id, next_status,
                                               validation=accepted.validation if next_status == "validated" else None)
    assert candidate.status == CandidateStatus.RETIRED
    try:
        store.transition_candidate(candidate.candidate_id, "active")
    except ValueError:
        pass
    else:
        raise AssertionError("retired candidates must not reactivate")
    store.close()
    config = config_for("exposure_drift")
    eliza_report = run_comparison(config, generate_scenario(config),
                                 {"G": ElizaExperimentAdapter(MockElizaHarness())}, variants=("G",))
    assert eliza_report.runs[0].execution_kind == "mock"
    assert eliza_report.runs[0].output.metadata["integration_claim"] == "not_eliza_integration"
    assert eliza_report.comparison_ready is False


def test_public_dataset_adapters_and_local_performance_artifacts(tmp_path):
    gdelt = adapt_gdelt_rows([{"DATEADDED": "20260102123456", "GLOBALEVENTID": "123",
                               "EventCode": "010", "GoldsteinScale": "-2.5"}])
    assert gdelt[0].event_time.year == 2026 and gdelt[0].attributes["goldstein_scale"] == -2.5
    fnspid = adapt_fnspid_rows([{"Date": "2025-01-01T08:00:00Z", "Ticker": "AAA",
                                 "Article_title": "Test"}])
    assert fnspid[0].entity_ids == ("equity:AAA",)
    assert adapt_fnspid_rows([{"Date": "2020-06-05 06:30:54 UTC", "Stock_symbol": "A"}])[0].event_time.hour == 6
    market = adapt_market_price_rows([{"symbol": "AAA", "date": "20250101", "close": "12.5"}])
    assert market[0].attributes["close"] == 12.5
    docs = parse_reuters21578('<REUTERS NEWID="1"><DATE>26-FEB-1987</DATE><TITLE>Example</TITLE><BODY>Text</BODY></REUTERS>')
    assert docs[0]["title"] == "Example" and docs[0]["text"] == "Text"
    result = benchmark_event_store((20, 50), batch_size=10, output_dir=tmp_path / "perf")
    assert [row["event_count_after_replay"] for row in result["measurements"]] == [20, 50]
    assert all(row["idempotent_replay_inserted"] == 0 for row in result["measurements"])
    assert (tmp_path / "perf" / "performance.svg").exists()


def test_gdelt_download_maps_the_official_58_column_row(tmp_path, monkeypatch):
    assert len(GDELT_EVENT_COLUMNS) == 58
    values = [""] * len(GDELT_EVENT_COLUMNS)
    values[GDELT_EVENT_COLUMNS.index("GLOBALEVENTID")] = "123456"
    values[GDELT_EVENT_COLUMNS.index("Actor1Name")] = "Example actor"
    values[GDELT_EVENT_COLUMNS.index("DATEADDED")] = "20261123123456"
    values[GDELT_EVENT_COLUMNS.index("SOURCEURL")] = "https://example.test/article"
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w") as archive:
        archive.writestr("20261123.export.CSV", "\t".join(values) + "\n")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return archive_buffer.getvalue()

    monkeypatch.setattr(dataset_module, "urlopen", lambda *_args, **_kwargs: Response())
    events, metadata = dataset_module.download_gdelt_day("20261123", limit=10)
    assert metadata["parsed_row_count"] == metadata["selected_event_count"] == 1
    assert events[0].event_time == datetime(2026, 11, 23, 12, 34, 56, tzinfo=timezone.utc)
    assert events[0].attributes["source_url"] == "https://example.test/article"


def test_public_gdelt_replay_pins_event_count_and_deduplicates(tmp_path, monkeypatch):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fake_events = tuple(Event(event_id=f"gdelt-{index}", event_type="news.gdelt_event",
                              event_time=base + timedelta(minutes=index), source="GDELT-Events-2.0")
                        for index in reversed(range(25)))
    monkeypatch.setattr(public_replay, "download_gdelt_day", lambda date, limit: (
        fake_events, {"dataset": "GDELT Events 2.0", "date": date, "parsed_row_count": 25,
                      "selected_event_count": len(fake_events), "archive_sha256": "abc"}))
    report = public_replay.replay_gdelt_day("20260101", tmp_path / "gdelt", limit=25, batch_size=10)
    assert report["replay_event_count"] == 25
    assert report["chronological_after_replay"] is True
    assert report["duplicate_replay_inserted"] == 0
    repeated = public_replay.replay_gdelt_day("20260101", tmp_path / "gdelt", limit=25, batch_size=10)
    assert repeated["inserted_event_count"] == 25
    assert repeated["database_file"] != report["database_file"]


def test_repeat_adapter_counts_outputs_not_reasoning_paths():
    class FixedAdapter:
        variant_id = "A"
        execution_kind = "mock"

        def run(self, experiment, budgets, *, trial_seed):
            positive = next((event for event in experiment.events
                             if event.event_time >= experiment.calibration_end), None)
            return AdapterOutput(predicted_event_ids=(positive.event_id,) if positive else (),
                                 evidence_refs={positive.event_id: (positive.event_id,)} if positive else {},
                                 compute_seconds=0.01)

    result = repeat_adapter(FixedAdapter(), generate_scenario(config_for("exposure_drift")),
                           RunBudget(), repeats=3)
    assert result["execution_kind"] == "mock"
    assert result["runs"] == 3
    assert "Reasoning-path variation is not measured" in result["note"]
