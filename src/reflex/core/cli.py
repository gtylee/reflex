"""REFLEX CLI for event ingestion, discovery, bounded actions, and benchmark records."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .discovery import fit_lenses, load_lenses, score_rows, cluster_observations, rows_signature
from .evaluation import append_run_record, mean_interval, protocol_manifest, protocol_report, score_synthetic_run
from .features import FEATURE_NAMES, build_entity_windows
from .ingestion import read_bpi2017, read_loghub
from .lenses import DEFAULT_SQL, SqlLens
from .models.lenses import CandidateStatus, LensCandidate
from .policy import BoundedTools, LocalInvestigator, PolicyEngine
from .storage import EventStore, StateStore
from .synthetic import generate_synthetic, write_dataset


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_discovery(data_dir: Path, *, epochs: int = 35, seed: int | None = None,
                  evaluator_dir: Path | None = None) -> dict[str, Any]:
    """Fit on early events, calibrate on the next interval, then score later events.

    The event directory stays label-blind. Evaluator labels are opened only when
    an explicit evaluator directory is passed after the observations are produced.
    """
    manifest = json.loads((data_dir / "dataset.json").read_text())
    event_store = EventStore(data_dir / "events.duckdb")
    state = StateStore(data_dir / "state.sqlite")
    try:
        events = event_store.native_events()
        start, train_end, calibration_end, end = (_dt(manifest[key]) for key in ("start", "train_end", "calibration_end", "end"))
        train_rows = build_entity_windows(events, start=start, end=train_end)
        calibration_rows = build_entity_windows(events, start=train_end, end=calibration_end)
        test_rows = build_entity_windows(events, start=calibration_end, end=end)
        baseline_event_rate = sum(event.event_time < train_end for event in events) / max((train_end - start).total_seconds() / 86400, 1.0)
        chosen_seed = int(manifest["seed"] if seed is None else seed)
        artifact_config = {"seed": chosen_seed, "epochs": epochs, "training_cutoff": train_end.isoformat(),
                           "feature_names": list(FEATURE_NAMES), "model_recipe": "reflex-statistical-v1",
                           "training_signature": rows_signature(train_rows),
                           "calibration_signature": rows_signature(calibration_rows)}
        artifact_key = hashlib.sha256(json.dumps(artifact_config, sort_keys=True).encode()).hexdigest()[:16]
        artifacts = data_dir / "artifacts" / f"statistical-{artifact_key}"
        model_manifest = artifacts / "manifest.json"
        if not model_manifest.exists():
            lenses = fit_lenses(train_rows, calibration_rows, artifacts, seed=chosen_seed, epochs=epochs, training_cutoff=train_end)
        else:
            lenses = load_lenses(artifacts)

        training_events = [event for event in events if event.event_time < train_end]
        baseline: dict[str, Any] = {}
        for feature_name in FEATURE_NAMES:
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
                    "status": "active", "sql": DEFAULT_SQL}, "deterministic SQL lens registered")
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
        sql_lens = SqlLens("sql.high_amount", "1.0", "A deterministic high-amount threshold", DEFAULT_SQL)

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
        investigator = LocalInvestigator(tools, state)
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


def _json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reflex", description="REFLEX event-driven discovery tools")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="create a seeded synthetic event dataset")
    generate.add_argument("--out", default=".reflex-demo")
    generate.add_argument("--seed", type=int, default=11)
    generate.add_argument("--entities", type=int, default=80)
    generate.add_argument("--days", type=int, default=30)
    generate.add_argument("--events-per-day", type=int, default=3)
    generate.add_argument("--evaluator-dir")

    demo = sub.add_parser("demo", help="run a complete synthetic REFLEX discovery path")
    demo.add_argument("--out", default=".reflex-demo")
    demo.add_argument("--seed", type=int, default=11)
    demo.add_argument("--entities", type=int, default=80)
    demo.add_argument("--days", type=int, default=30)
    demo.add_argument("--events-per-day", type=int, default=3)
    demo.add_argument("--epochs", type=int, default=35)
    demo.add_argument("--evaluator-dir")

    discover = sub.add_parser("discover", help="fit or replay the statistical lenses for a generated dataset")
    discover.add_argument("--data-dir", default=".reflex-demo")
    discover.add_argument("--epochs", type=int, default=35)
    discover.add_argument("--seed", type=int)
    discover.add_argument("--evaluator-dir", help="optional evaluator-only directory; omit to keep this run label-blind")

    ingest_bpi = sub.add_parser("ingest-bpi", help="ingest a locally acquired BPI 2017 CSV")
    ingest_bpi.add_argument("path")
    ingest_bpi.add_argument("--db", default="events.duckdb")

    ingest_loghub = sub.add_parser("ingest-loghub", help="ingest a local Loghub CSV, JSONL, or HDFS log")
    ingest_loghub.add_argument("path")
    ingest_loghub.add_argument("--db", default="events.duckdb")

    replay = sub.add_parser("replay", help="replay the deterministic SQL lens in event-time batches")
    replay.add_argument("--db", default="events.duckdb")
    replay.add_argument("--state", default="state.sqlite")
    replay.add_argument("--batch-days", type=int, default=5)
    replay.add_argument("--start")
    replay.add_argument("--end")
    replay.add_argument("--limit", type=int, default=1000)

    export = sub.add_parser("export-parquet", help="export a time-ordered Parquet event snapshot")
    export.add_argument("--db", default="events.duckdb")
    export.add_argument("--out", required=True)

    state_cmd = sub.add_parser("state", help="inspect durable state and transition audit")
    state_cmd.add_argument("--db", default="state.sqlite")
    state_cmd.add_argument("--namespace", choices=sorted(StateStore.NAMESPACES))
    state_cmd.add_argument("--audit", action="store_true")

    lens = sub.add_parser("lens", help="manage proposed SQL lens candidates")
    lens.add_argument("--state", default="state.sqlite")
    lens_sub = lens.add_subparsers(dest="lens_command", required=True)
    propose = lens_sub.add_parser("propose")
    propose.add_argument("spec")
    approve = lens_sub.add_parser("approve")
    approve.add_argument("candidate_id")
    activate = lens_sub.add_parser("activate")
    activate.add_argument("candidate_id")
    retire = lens_sub.add_parser("retire")
    retire.add_argument("candidate_id")
    validate = lens_sub.add_parser("validate")
    validate.add_argument("candidate_id")
    validate.add_argument("--db", default="events.duckdb")
    validate.add_argument("--start", required=True)
    validate.add_argument("--end", required=True)
    validate.add_argument("--run-id", default="candidate-validation")

    benchmark = sub.add_parser("discovery-benchmark", help="initialize/report manually recorded A–E comparison runs")
    bench_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    init = bench_sub.add_parser("init")
    init.add_argument("--out", default="benchmark-protocol.json")
    init.add_argument("--seeds", default="11,23,37")
    init.add_argument("--objective", default="Find recurring or novel behaviour worth investigating, cite event evidence, and distinguish hypotheses from validated discoveries.")
    init.add_argument("--inference-budget", type=int, default=20000)
    init.add_argument("--compute-budget-seconds", type=int, default=900)
    record = bench_sub.add_parser("record")
    record.add_argument("--protocol", required=True)
    record.add_argument("--record", required=True)
    report = bench_sub.add_parser("report")
    report.add_argument("--protocol", required=True)
    pilot = bench_sub.add_parser("pilot", help="run repeated seeded REFLEX synthetic evaluation")
    pilot.add_argument("--out", default="reports/reflex-pilot")
    pilot.add_argument("--seeds", default="11,23,37")
    pilot.add_argument("--entities", type=int, default=40)
    pilot.add_argument("--days", type=int, default=30)
    pilot.add_argument("--epochs", type=int, default=20)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        dataset = generate_synthetic(args.seed, args.entities, args.days, args.events_per_day)
        files = write_dataset(dataset, args.out, args.evaluator_dir)
        store = EventStore(Path(args.out) / "events.duckdb")
        inserted = store.add_events(dataset.events)
        result = {"files": files, "events_inserted": inserted, "events_total": store.count_native_events(),
                  "warning": "ground truth is in the separate evaluator directory; keep it outside agent workspaces during comparative runs."}
        store.close()
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "demo":
        dataset = generate_synthetic(args.seed, args.entities, args.days, args.events_per_day)
        eval_dir = Path(args.evaluator_dir) if args.evaluator_dir else Path(args.out).parent / f"{Path(args.out).name}-evaluator"
        files = write_dataset(dataset, args.out, eval_dir)
        store = EventStore(Path(args.out) / "events.duckdb")
        store.add_events(dataset.events)
        store.close()
        print(json.dumps({"dataset_files": files, "run": run_discovery(Path(args.out), epochs=args.epochs, evaluator_dir=eval_dir)}, indent=2))
        return 0
    if args.command == "discover":
        print(json.dumps(run_discovery(Path(args.data_dir), epochs=args.epochs, seed=args.seed,
                                       evaluator_dir=Path(args.evaluator_dir) if args.evaluator_dir else None), indent=2))
        return 0
    if args.command in {"ingest-bpi", "ingest-loghub"}:
        events = read_bpi2017(args.path) if args.command == "ingest-bpi" else read_loghub(args.path)
        store = EventStore(args.db)
        inserted = store.add_events(events)
        print(json.dumps({"inserted": inserted, "total": store.count_native_events(), "db": str(Path(args.db).resolve())}, indent=2))
        store.close()
        return 0
    if args.command == "replay":
        store = EventStore(args.db)
        state = StateStore(args.state)
        lens = SqlLens("sql.high_amount", "1.0", "A deterministic high-amount threshold", DEFAULT_SQL)
        start, end = _dt(args.start) if args.start else None, _dt(args.end) if args.end else None
        if (start is None) != (end is None):
            raise ValueError("replay requires both --start and --end, or neither")
        if args.batch_days <= 0:
            raise ValueError("batch-days must be positive")
        all_events = store.native_events()
        event_times = [event.event_time for event in all_events]
        operational = state.snapshot("operational")
        dataset_state = operational.get("dataset", {})
        baseline_event_rate = float(dataset_state.get("baseline_events_per_day", 0.0))
        training_cutoff = _dt(dataset_state["train_end"]) if dataset_state.get("train_end") else None

        def rolling_at(as_of: datetime) -> dict[str, Any]:
            left = bisect_left(event_times, as_of - timedelta(days=5))
            right = bisect_left(event_times, as_of)
            recent = all_events[left:right]
            expected = baseline_event_rate * 5
            ratio = len(recent) / max(expected, 1.0) if baseline_event_rate else 0.0
            return {"as_of": as_of.isoformat(), "window_days": 5, "event_count": len(recent),
                    "events_per_day": len(recent) / 5,
                    "event_type_counts": {kind: sum(item.event_type == kind for item in recent)
                                          for kind in sorted({item.event_type for item in recent})},
                    "baseline_expected_count": expected, "baseline_volume_ratio": ratio,
                    "regime": ("elevated" if ratio >= 1.2 else "reduced" if ratio <= 0.8 else "baseline") if baseline_event_rate else "unavailable"}

        def snapshot_for_sql(row):
            as_of = row["event_time"]
            if training_cutoff and as_of < training_cutoff:
                return {"as_of": as_of.isoformat(), "phase": "before_operational_baseline"}
            snapshot = state.snapshot_at("operational", as_of)
            snapshot["as_of"] = as_of.isoformat()
            snapshot["rolling_metrics"] = rolling_at(as_of)
            entity_ids = row.get("entity_ids", [])
            if isinstance(entity_ids, str):
                entity_ids = json.loads(entity_ids)
            snapshot["entity_relationship"] = snapshot.get("entity_relationships", {}).get(entity_ids[0], []) if entity_ids else []
            snapshot.pop("entity_relationships", None)
            return snapshot

        results = []
        if start and end:
            cursor = start
            while cursor < end:
                batch_end = min(cursor + timedelta(days=args.batch_days), end)
                batch_observations = lens.replay(store, start=cursor, end=batch_end, state_snapshot=snapshot_for_sql, limit=args.limit)
                store.save_observations(batch_observations)
                results.append({"start": cursor.isoformat(), "end": batch_end.isoformat(), "observations": len(batch_observations)})
                state.set("operational", "rolling_metrics", rolling_at(batch_end), f"SQL lens replay through {batch_end.isoformat()}")
                cursor = batch_end
        else:
            for batch_start, batch_end, _count in store.batches(args.batch_days):
                batch_observations = lens.replay(store, start=batch_start, end=batch_end, state_snapshot=snapshot_for_sql, limit=args.limit)
                store.save_observations(batch_observations)
                results.append({"start": batch_start.isoformat(), "end": batch_end.isoformat(), "observations": len(batch_observations)})
                state.set("operational", "rolling_metrics", rolling_at(batch_end), f"SQL lens replay through {batch_end.isoformat()}")
        print(json.dumps({"lens_id": lens.definition.lens_id, "version": lens.definition.version,
                          "batches": results, "observations_total": sum(row["observations"] for row in results)}, indent=2))
        state.close()
        store.close()
        return 0
    if args.command == "export-parquet":
        store = EventStore(args.db)
        count = store.export_parquet(args.out)
        store.close()
        print(json.dumps({"rows": count, "path": str(Path(args.out).resolve())}, indent=2))
        return 0
    if args.command == "state":
        state = StateStore(args.db)
        data = state.audit(args.namespace) if args.audit else state.snapshot(args.namespace)
        print(json.dumps(data, indent=2, default=str))
        state.close()
        return 0
    if args.command == "lens":
        state = StateStore(args.state)
        if args.lens_command == "propose":
            candidate = LensCandidate.model_validate(_json(args.spec))
            state.put_candidate(candidate)
            result: Any = candidate.model_dump(mode="json")
        elif args.lens_command == "validate":
            event_store = EventStore(args.db)
            policy = PolicyEngine(state, run_id=args.run_id, max_investigations=0, max_queries=5)
            tools = BoundedTools(event_store, state, run_id=args.run_id)
            result = tools.validate_lens(args.candidate_id, event_store, validation_start=_dt(args.start), validation_end=_dt(args.end))
            event_store.close()
        else:
            transitions = {"approve": CandidateStatus.APPROVED.value, "activate": CandidateStatus.ACTIVE.value, "retire": CandidateStatus.RETIRED.value}
            candidate_id = args.candidate_id
            result = state.transition_candidate(candidate_id, transitions[args.lens_command]).model_dump(mode="json")
        print(json.dumps(result, indent=2, default=str))
        state.close()
        return 0
    if args.command == "discovery-benchmark":
        if args.benchmark_command == "init":
            seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError("provide one or more distinct random seeds")
            manifest = protocol_manifest(seeds=seeds, objective=args.objective,
                                         inference_budget=args.inference_budget,
                                         compute_budget_seconds=args.compute_budget_seconds)
            Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"protocol": str(Path(args.out).resolve()), "variants": list("ABCDE"), "seeds": seeds}, indent=2))
        elif args.benchmark_command == "record":
            append_run_record(args.protocol, _json(args.record))
            print(json.dumps({"recorded": True, "protocol": str(Path(args.protocol).resolve())}))
        elif args.benchmark_command == "report":
            result = protocol_report(args.protocol)
            print(json.dumps(result, indent=2))
        else:
            seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError("provide one or more distinct random seeds")
            output = Path(args.out)
            output.mkdir(parents=True, exist_ok=True)
            runs = []
            for seed_value in seeds:
                run_dir = output / "runs" / f"seed-{seed_value}"
                evaluator_dir = output / "evaluator" / f"seed-{seed_value}"
                dataset = generate_synthetic(seed_value, args.entities, args.days, 3)
                write_dataset(dataset, run_dir, evaluator_dir)
                event_store = EventStore(run_dir / "events.duckdb")
                event_store.add_events(dataset.events)
                event_store.close()
                started = time.perf_counter()
                run = run_discovery(run_dir, epochs=args.epochs, evaluator_dir=evaluator_dir)
                run["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                runs.append(run)
            metrics = {}
            for lens_id in ("stat.autoencoder", "stat.isolation_forest"):
                metrics[lens_id] = {
                    key: mean_interval([float(run["held_out_metrics"][lens_id][key]) for run in runs])
                    for key in ("false_discovery_rate", "hidden_event_recall", "mechanism_recovery_count")
                }
            report = {"title": "REFLEX synthetic pilot", "runs": len(runs), "seeds": seeds,
                      "entities": args.entities, "days": args.days, "epochs": args.epochs,
                      "metrics": metrics, "wall_seconds": mean_interval([run["elapsed_seconds"] for run in runs]),
                      "per_seed": [{"seed": run["seed"], "events": run["events"], "metrics": run["held_out_metrics"]} for run in runs],
                      "timing_scope": "Includes feature construction, model fitting, scoring, bounded investigations, and evaluation; excludes synthetic generation and event ingestion.",
                      "scoring_unit": "Each prediction covers one entity in a five-day window. It is counted as a true positive when the window contains a hidden event. Recall counts hidden events inside flagged windows.",
                      "interpretation": "This pilot measures the REFLEX statistical lenses against hidden synthetic labels. It does not compare variants A–E or establish that persistent state improves discovery."}
            (output / "pilot-results.json").write_text(json.dumps(report, indent=2) + "\n")
            lines = ["# REFLEX synthetic pilot", "", f"Repeated seeds: {len(runs)} ({', '.join(map(str, seeds))}).",
                     f"Configuration: {args.entities} entities, {args.days} days, {args.epochs} training epochs per seed.", "",
                     "| Lens | False discovery rate, mean (95% interval) | Hidden event recall, mean (95% interval) | Mechanisms recovered, mean (95% interval) |",
                     "|---|---:|---:|---:|"]
            for lens_id, values in metrics.items():
                fdr, recall, recovered = (values[key] for key in ("false_discovery_rate", "hidden_event_recall", "mechanism_recovery_count"))
                lines.append(f"| {lens_id} | {fdr['mean']:.3f} ({fdr['lower']:.3f}–{fdr['upper']:.3f}) | {recall['mean']:.3f} ({recall['lower']:.3f}–{recall['upper']:.3f}) | {recovered['mean']:.2f} ({recovered['lower']:.2f}–{recovered['upper']:.2f}) |")
            lines.extend(["", f"Mean wall time: {report['wall_seconds']['mean']:.2f}s (95% bootstrap interval {report['wall_seconds']['lower']:.2f}–{report['wall_seconds']['upper']:.2f}s).", "",
                          report["scoring_unit"], report["timing_scope"], "",
                          report["interpretation"], "", "All records are generated from actual runs. Ground truth is read only by the evaluator after scoring."])
            (output / "pilot-results.md").write_text("\n".join(lines) + "\n")
            # Lightweight SVG keeps the report chart dependency-free.
            bars = []
            colors = ["#2d7e75", "#bb744b"]
            for index, lens_id in enumerate(metrics):
                value = metrics[lens_id]["hidden_event_recall"]["mean"]
                y = 56 + index * 64
                bars.append(f'<text x="16" y="{y + 19}" font-size="13" fill="#263a43">{lens_id}</text><rect x="210" y="{y}" width="{max(2, value * 440):.1f}" height="28" rx="5" fill="{colors[index]}"/><text x="{218 + value * 440:.1f}" y="{y + 19}" font-size="13" fill="#263a43">{value:.2f}</text>')
            svg = '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="210" viewBox="0 0 720 210"><rect width="100%" height="100%" fill="#f7f5ef"/><text x="16" y="25" font-family="sans-serif" font-size="15" fill="#263a43">Held-out hidden-event recall</text>' + ''.join(bars) + '<text x="210" y="194" font-size="11" fill="#60727a">0.00</text><text x="632" y="194" font-size="11" fill="#60727a">1.00</text></svg>'
            (output / "pilot-recall.svg").write_text(svg + "\n")
            print(json.dumps({"report": str((output / "pilot-results.md").resolve()),
                              "data": str((output / "pilot-results.json").resolve()),
                              "chart": str((output / "pilot-recall.svg").resolve()), "runs": len(runs)}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
