"""Synthetic transaction discovery workflow used by REFLEX experiments."""

from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from reflex.core.discovery import fit_lenses, load_lenses, score_rows, cluster_observations, rows_signature
from reflex.core.lenses import SqlLens
from reflex.core.models.lenses import Observation
from reflex.core.policy import BoundedTools, LocalInvestigator, PolicyEngine
from reflex.core.storage import EventStore, StateStore
from .evaluation import mean_interval, score_synthetic_run
from .synthetic_data import generate_synthetic, write_dataset
from .transaction_features import TRANSACTION_FEATURE_NAMES, build_transaction_windows
from .transaction_lenses import TRANSACTION_HIGH_AMOUNT_SQL


def transaction_investigation_feature(observation: Observation) -> str:
    """Choose the demo profile's transaction field for a cohort summary."""
    return ("manual_channel_share" if observation.observation_type == "autoencoder_reconstruction_score"
            else "mean_amount")


def _dt(value: str) -> datetime:
    """Parse an ISO timestamp accepted by the dataset manifest."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_discovery(data_dir: Path, *, epochs: int = 35, seed: int | None = None,
                  evaluator_dir: Path | None = None) -> dict[str, Any]:
    """Run the local discovery path using chronological feature windows.

    The manifest supplies start/train/calibration/end boundaries. The model
    uses only native events to build each split, fits or loads versioned model
    artifacts, scores calibration and later rows, saves observations, records
    policy/investigation state, and returns a compact summary. The ordinary
    data directory remains label-blind. If ``evaluator_dir`` is explicitly
    supplied, hidden labels are opened only after the observations are made.

    This is a demonstration workflow over a generated dataset. It is not a
    streaming service, and its anomaly flags are not human-validated labels.
    """
    manifest = json.loads((data_dir / "dataset.json").read_text())
    event_store = EventStore(data_dir / "events.duckdb")
    state = StateStore(data_dir / "state.sqlite")
    try:
        events = event_store.native_events()
        start, train_end, calibration_end, end = (_dt(manifest[key]) for key in ("start", "train_end", "calibration_end", "end"))
        train_rows = build_transaction_windows(events, start=start, end=train_end)
        calibration_rows = build_transaction_windows(events, start=train_end, end=calibration_end)
        test_rows = build_transaction_windows(events, start=calibration_end, end=end)
        baseline_event_rate = sum(event.event_time < train_end for event in events) / max((train_end - start).total_seconds() / 86400, 1.0)
        chosen_seed = int(manifest["seed"] if seed is None else seed)
        artifact_config = {"seed": chosen_seed, "epochs": epochs, "training_cutoff": train_end.isoformat(),
                           "feature_names": list(TRANSACTION_FEATURE_NAMES), "model_recipe": "reflex-transaction-profile-v1",
                           "training_signature": rows_signature(train_rows),
                           "calibration_signature": rows_signature(calibration_rows)}
        artifact_key = hashlib.sha256(json.dumps(artifact_config, sort_keys=True).encode()).hexdigest()[:16]
        artifacts = data_dir / "artifacts" / f"statistical-{artifact_key}"
        model_manifest = artifacts / "manifest.json"
        if not model_manifest.exists():
            lenses = fit_lenses(train_rows, calibration_rows, artifacts, seed=chosen_seed,
                                epochs=epochs, training_cutoff=train_end,
                                feature_names=TRANSACTION_FEATURE_NAMES)
        else:
            lenses = load_lenses(artifacts)

        training_events = [event for event in events if event.event_time < train_end]
        baseline: dict[str, Any] = {}
        for feature_name in TRANSACTION_FEATURE_NAMES:
            values = [float(row.feature_values[feature_name]) for row in train_rows]
            mean = sum(values) / len(values) if values else 0.0
            variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
            baseline[feature_name] = {"mean": mean, "std": variance ** 0.5, "training_windows": len(values)}
        entity_relationships: dict[str, list[str]] = {}
        for event in training_events:
            for entity in event.entity_ids:
                links = entity_relationships.setdefault(entity, [])
                links.extend(str(value) for value in event.relationships if str(value) not in links)
        state.set("operational", "dataset", {"dataset": manifest["dataset"], "seed": manifest["seed"],
                    "training_event_count": len(training_events), "baseline_events_per_day": baseline_event_rate,
                    "train_end": train_end.isoformat()}, "training data loaded")
        state.set("operational", "baselines", baseline, "training-window baselines fitted")
        state.set("operational", "entity_relationships", entity_relationships, "relationships learned from training events")
        previous = state.get("lens", "latest_statistical")
        if previous and previous.get("model_version") != lenses.model_version:
            old_key = f"statistical:{previous['model_version']}"
            old_record = state.get("lens", old_key)
            if old_record and old_record.get("status") == "active":
                old_record["status"] = "retained"
                state.set("lens", old_key, old_record, "new statistical lens version selected")
        state.set("lens", f"statistical:{lenses.model_version}", {"model_version": lenses.model_version, "config": lenses.config,
                    "status": "active", "artifacts": str(artifacts)}, "statistical lens version registered")
        state.set("lens", "latest_statistical", {"model_version": lenses.model_version, "artifact_dir": str(artifacts)}, "statistical lenses fitted or loaded")
        state.set("lens", "sql.high_amount", {"lens_id": "sql.high_amount", "version": "1.0",
                    "status": "active", "sql": TRANSACTION_HIGH_AMOUNT_SQL}, "deterministic SQL lens registered")
        scored_rows = calibration_rows + test_rows
        ordered_event_times = [event.event_time for event in events]

        def rolling_snapshot(as_of: datetime) -> dict[str, Any]:
            left = bisect_left(ordered_event_times, as_of - timedelta(days=5))
            right = bisect_left(ordered_event_times, as_of)
            recent = events[left:right]
            expected_count = baseline_event_rate * 5
            ratio = len(recent) / max(expected_count, 1.0)
            regime = "elevated" if ratio >= 1.2 else "reduced" if ratio <= 0.8 else "baseline"
            return {"as_of": as_of.isoformat(), "window_days": 5, "event_count": len(recent),
                    "events_per_day": len(recent) / 5, "event_type_counts": dict(sorted(
                        (kind, sum(event.event_type == kind for event in recent))
                        for kind in {event.event_type for event in recent})),
                    "baseline_expected_count": expected_count, "baseline_volume_ratio": ratio, "regime": regime}

        for as_of in sorted({row.window_end for row in scored_rows}):
            state.set("operational", "rolling_metrics", rolling_snapshot(as_of), f"replay through {as_of.isoformat()}")

        def state_at_row(row):
            snapshot = state.snapshot_at("operational", row.window_end)
            snapshot["rolling_metrics"] = rolling_snapshot(row.window_end)
            snapshot["entity_relationship"] = snapshot.get("entity_relationships", {}).get(row.subject, [])
            snapshot.pop("entity_relationships", None)
            return snapshot

        observations = score_rows(scored_rows, lenses, state_snapshot=state_at_row)
        sql_lens = SqlLens("sql.high_amount", "1.0", "A deterministic high-amount threshold", TRANSACTION_HIGH_AMOUNT_SQL)

        def sql_state_at_row(row):
            at = row["event_time"]
            if at < train_end:
                return {"as_of": at.isoformat(), "phase": "before_operational_baseline", "baseline": {}}
            snapshot = state.snapshot_at("operational", at)
            snapshot["as_of"] = at.isoformat()
            snapshot["rolling_metrics"] = rolling_snapshot(at)
            entity_ids = row.get("entity_ids", [])
            if isinstance(entity_ids, str):
                entity_ids = json.loads(entity_ids)
            snapshot["entity_relationship"] = snapshot.get("entity_relationships", {}).get(entity_ids[0], []) if entity_ids else []
            snapshot.pop("entity_relationships", None)
            return snapshot

        sql_observations = sql_lens.replay(event_store, state_snapshot=sql_state_at_row, limit=1000)
        event_store.save_observations([*observations, *sql_observations])

        test_keys = {(row.subject, row.window_start, row.window_end) for row in test_rows}
        test_observations = [obs for obs in observations if (obs.subject, obs.window_start, obs.window_end) in test_keys]
        clusters = cluster_observations(test_observations)
        state.set("discovery", "latest_clusters", clusters, "statistical observation clustering")
        state.set("discovery", "latest_observation_count", len(observations) + len(sql_observations), "observation batch persisted")

        policy = PolicyEngine(state, run_id=f"demo-{manifest['seed']}", max_investigations=3, max_queries=20)
        tools = BoundedTools(event_store, state, run_id=f"demo-{manifest['seed']}", query_limit=500)
        investigator = LocalInvestigator(tools, state, feature_selector=transaction_investigation_feature)
        flagged = [obs for obs in test_observations if isinstance(obs.value, dict) and obs.value.get("is_anomaly")]
        flagged.sort(key=lambda obs: float(obs.value.get("score", 0)), reverse=True)
        by_window: dict[tuple[str, datetime, datetime], list] = {}
        for item in observations:
            key = (item.subject, item.window_start, item.window_end)
            by_window.setdefault(key, []).append(item)
        traces = []
        for item in flagged:
            decision = policy.decide(item)
            state.set("discovery", f"policy:{decision.decision_id}", decision, "deterministic policy decision")
            if decision.action.value == "investigate":
                key = (item.subject, item.window_start, item.window_end)
                traces.append(investigator.investigate(item, by_window.get(key, [item])))
        eval_by_lens = None
        if evaluator_dir is not None:
            truth_path = evaluator_dir / "private_ground_truth.json"
            truth = json.loads(truth_path.read_text())
            truth_event_ids = {item["event_id"] for item in truth}
            truth_times = {event.event_id: event.event_time for event in events if event.event_id in truth_event_ids}
            eval_by_lens = {
                lens_id: score_synthetic_run([obs for obs in test_observations if obs.lens_id == lens_id], truth, truth_times)
                for lens_id in ("stat.autoencoder", "stat.isolation_forest")
            }
        summary = {
            "dataset": manifest["dataset"], "seed": manifest["seed"], "events": event_store.count_native_events(),
            "windows": {"train": len(train_rows), "calibration": len(calibration_rows), "test": len(test_rows)},
            "model_version": lenses.model_version, "artifacts": str(artifacts),
            "observations": {"statistical": len(observations), "sql": len(sql_observations),
                             "anomalous_test": len(flagged), "clusters": len(clusters)},
            "investigations": len(traces), "policy_budget": state.budget(f"demo-{manifest['seed']}"),
            "held_out_metrics": eval_by_lens,
            "limits": (["Synthetic evaluator reads labels only after observations are produced.",
                        "This evaluates REFLEX statistical detection; it is not a comparison against variants A–E."]
                       if evaluator_dir is not None else ["No hidden labels were read by this discovery run.",
                                                         "This is not a comparison against variants A–E."]),
        }
        (data_dir / "latest-run.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        event_store.close()
        state.close()
