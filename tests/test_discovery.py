from datetime import datetime, timedelta, timezone
import hashlib
import json

from reflex.core.discovery import fit_lenses, load_lenses, score_rows
from reflex.core.evaluation import (append_run_record, bind_protocol_input_hashes,
                                    protocol_manifest, protocol_report, score_synthetic_run)
from reflex.core.features import build_entity_windows, event_level_features
from reflex.core.models import Event
from reflex.core.synthetic import generate_synthetic, write_dataset


def test_synthetic_generator_separates_truth_and_has_chronological_splits(tmp_path):
    dataset = generate_synthetic(seed=5, entities=16, days=30, events_per_day=2)
    assert len(dataset.events) > 0 and dataset.ground_truth
    assert all("mechanism" not in event.attributes for event in dataset.events)
    assert dataset.start < dataset.train_end < dataset.calibration_end < dataset.end
    paths = write_dataset(dataset, tmp_path / "data")
    evaluator_path = tmp_path / "data-evaluator" / "private_ground_truth.json"
    assert json.loads(evaluator_path.read_text()) == dataset.ground_truth
    assert not (tmp_path / "data" / "private_ground_truth.json").exists()
    assert "ground_truth_path" not in json.loads((tmp_path / "data" / "dataset.json").read_text())


def test_feature_windows_do_not_cross_boundaries_and_event_features_have_interarrival():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events = [
        Event(event_id="a", event_type="open", event_time=start + timedelta(days=1), entity_ids=("x",), attributes={"amount": 3}),
        Event(event_id="b", event_type="close", event_time=start + timedelta(days=2), entity_ids=("x",), attributes={"amount": 7}),
        Event(event_id="c", event_type="open", event_time=start + timedelta(days=6), entity_ids=("x",), attributes={"amount": 20}),
    ]
    windows = build_entity_windows(events, start=start, end=start + timedelta(days=10), window_days=5)
    assert len(windows) == 2
    assert windows[0].feature_values["event_count"] == 2
    assert windows[1].feature_values["event_count"] == 1
    assert "amount_latency_correlation" in windows[0].feature_values
    assert "partner_mean_amount_ratio" in windows[0].feature_values
    features = event_level_features(events)
    assert features[0]["interarrival_hours"] == 0
    assert features[1]["interarrival_hours"] == 24


def test_statistical_lenses_calibrate_and_reload_without_retraining(tmp_path):
    dataset = generate_synthetic(seed=9, entities=16, days=30, events_per_day=2)
    train = build_entity_windows(dataset.events, start=dataset.start, end=dataset.train_end)
    calibration = build_entity_windows(dataset.events, start=dataset.train_end, end=dataset.calibration_end)
    test = build_entity_windows(dataset.events, start=dataset.calibration_end, end=dataset.end)
    fitted = fit_lenses(train, calibration, tmp_path / "artifacts", seed=9, epochs=4,
                        training_cutoff=dataset.train_end)
    first = score_rows(test, fitted)
    loaded = load_lenses(tmp_path / "artifacts")
    second = score_rows(test, loaded)
    assert len(first) == len(test) * 2
    assert [item.observation_id for item in first] == [item.observation_id for item in second]
    assert [item.value["score"] for item in first] == [item.value["score"] for item in second]
    assert all(item.quality["calibrated"] for item in first)
    manifest = json.loads((tmp_path / "artifacts" / "manifest.json").read_text())
    assert manifest["training_cutoff"] == dataset.train_end.isoformat()
    assert "feature_names" in manifest


def test_hidden_truth_evaluator_counts_lens_observations():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    event = Event(event_id="hidden", event_type="x", event_time=start + timedelta(hours=1), entity_ids=("e",))
    from reflex.core.models import Observation
    observation = Observation(observation_type="score", lens_id="one", lens_version="v1", subject="e",
        window_start=start, window_end=start + timedelta(days=1), value={"is_anomaly": True})
    metrics = score_synthetic_run([observation], [{"event_id": "hidden", "entity_id": "e", "mechanism": "m"}], {event.event_id: event.event_time})
    assert metrics["hidden_event_recall"] == 1
    assert metrics["mechanisms_recovered"] == ["m"]
    assert metrics["false_discovery_rate"] == 0


def test_manual_protocol_requires_recorded_inputs_and_costs(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol_manifest(seeds=[11], objective="discover", inference_budget=100,
        compute_budget_seconds=60), indent=2))
    batches = ["one-shot", "batch-1", "batch-2", "batch-3", "batch-4", "batch-5", "batch-6"]
    input_hashes = {"11": {batch: hashlib.sha256(f"{batch}-input".encode()).hexdigest() for batch in batches}}
    bind_protocol_input_hashes(path, input_hashes)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifact = artifact_dir / "run-output.json"
    output = {"validated_discoveries": 1, "hypotheses": [{"claim": "a candidate"}]}
    artifact.write_text(json.dumps(output))
    record = {"variant": "A", "seed": 11, "batch": "one-shot",
              "input_hash": input_hashes["11"]["one-shot"], "objective": "discover",
              "output": output, "inference_tokens": 50, "inference_cost": 0.01,
              "compute_seconds": 20, "investigation_count": 2, "evidence_refs": ["e1"],
              "workspace_ref": "run/a", "execution": {"system": "test-agent", "version": "1",
                  "artifact_path": "artifacts/run-output.json",
                  "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    append_run_record(path, record)
    report = protocol_report(path)
    assert report["recorded_runs"] == 1
    assert report["input_hashes_bound"] is True
    assert report["complete"] is False
    assert report["variants"][0]["validated_discoveries"] == 1
    assert len(report["variants"][0]["missing_runs"]) == 6
    try:
        append_run_record(path, record)
    except ValueError as error:
        assert "already recorded" in str(error)
    else:
        raise AssertionError("duplicate run should be rejected")


def test_manual_protocol_rejects_mismatched_input_objective_and_artifact(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol_manifest(seeds=[7], objective="fixed objective", inference_budget=100,
        compute_budget_seconds=60), indent=2))
    batches = ["one-shot", "batch-1", "batch-2", "batch-3", "batch-4", "batch-5", "batch-6"]
    hashes = {"7": {batch: hashlib.sha256(batch.encode()).hexdigest() for batch in batches}}
    bind_protocol_input_hashes(path, hashes)
    artifact = tmp_path / "output.json"
    output = {"validated_discoveries": 0, "hypotheses": []}
    artifact.write_text(json.dumps(output))
    base = {"variant": "B", "seed": 7, "batch": "one-shot", "input_hash": hashes["7"]["one-shot"],
            "objective": "fixed objective", "output": output, "inference_tokens": 0, "inference_cost": 0.0,
            "compute_seconds": 0, "investigation_count": 0, "evidence_refs": [], "workspace_ref": "commit-x",
            "execution": {"system": "agent", "version": "model-x", "artifact_path": "output.json",
                          "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    for changes, message in (({"input_hash": "f" * 64}, "input_hash"),
                             ({"objective": "changed"}, "objective"),
                             ({"execution": {**base["execution"], "artifact_sha256": "0" * 64}}, "artifact hash")):
        try:
            append_run_record(path, {**base, **changes})
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"expected {message} mismatch to be rejected")
