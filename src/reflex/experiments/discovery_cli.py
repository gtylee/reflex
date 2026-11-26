"""CLI for the transaction-shaped synthetic discovery demonstrations.

The reusable event store and detector functions live in ``reflex.core``.
This module supplies one experiment profile: generated transaction-like data,
its feature schema, synthetic labels, and report formatting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from reflex.core.storage import EventStore
from .correlation_benchmark import run_correlation_benchmark, write_correlation_benchmark
from .evaluation import (append_run_record, bind_protocol_input_hashes, mean_interval,
                         protocol_manifest, protocol_report, score_synthetic_run)
from .local_discovery import run_discovery
from .synthetic_data import generate_synthetic, write_dataset


def _json(path: str) -> dict:
    """Read a JSON file used to define one manual benchmark record."""
    return json.loads(Path(path).read_text())


def _run_pilot(seeds: list[int], *, entities: int, days: int, epochs: int, output: Path) -> dict:
    """Run the REFLEX synthetic detector on each requested generator seed."""
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        run_dir = output / "runs" / f"seed-{seed}"
        evaluator_dir = output / "evaluator" / f"seed-{seed}"
        dataset = generate_synthetic(seed, entities, days, 3)
        write_dataset(dataset, run_dir, evaluator_dir)
        store = EventStore(run_dir / "events.duckdb")
        try:
            store.add_events(dataset.events)
        finally:
            store.close()
        started = time.perf_counter()
        run = run_discovery(run_dir, epochs=epochs, evaluator_dir=evaluator_dir)
        run["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        runs.append(run)

    metrics = {}
    for lens_id in ("stat.autoencoder", "stat.isolation_forest"):
        metrics[lens_id] = {
            key: mean_interval([float(run["held_out_metrics"][lens_id][key]) for run in runs])
            for key in ("false_discovery_rate", "hidden_event_recall", "mechanism_recovery_count")
        }
    report = {
        "title": "REFLEX synthetic pilot", "runs": len(runs), "seeds": seeds,
        "entities": entities, "days": days, "epochs": epochs, "metrics": metrics,
        "wall_seconds": mean_interval([run["elapsed_seconds"] for run in runs]),
        "per_seed": [{"seed": run["seed"], "events": run["events"],
                      "metrics": run["held_out_metrics"]} for run in runs],
        "timing_scope": "Includes feature construction, model fitting, scoring, bounded investigations, and evaluation; excludes synthetic generation and event ingestion.",
        "scoring_unit": "Each prediction covers one entity in a five-day window. A window is a true positive when it contains a hidden event. Recall counts hidden events inside flagged windows.",
        "interpretation": "This pilot measures REFLEX statistical lenses against synthetic generator labels. It does not compare actual external agent runs or establish real-world anomaly quality.",
    }
    (output / "pilot-results.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# REFLEX synthetic pilot", "", f"Runs: {len(runs)}", "", "| Lens | False discovery rate | Hidden-event recall | Mechanisms recovered |", "|---|---:|---:|---:|"]
    for lens_id, values in metrics.items():
        fdr, recall, recovered = (values[key] for key in (
            "false_discovery_rate", "hidden_event_recall", "mechanism_recovery_count"))
        lines.append(f"| {lens_id} | {fdr['mean']:.3f} ({fdr['lower']:.3f}–{fdr['upper']:.3f}) | {recall['mean']:.3f} ({recall['lower']:.3f}–{recall['upper']:.3f}) | {recovered['mean']:.2f} ({recovered['lower']:.2f}–{recovered['upper']:.2f}) |")
    lines.extend(["", report["scoring_unit"], report["timing_scope"], "", report["interpretation"], ""])
    (output / "pilot-results.md").write_text("\n".join(lines))

    colors = ("#2d7e75", "#bb744b")
    bars = []
    for index, lens_id in enumerate(metrics):
        value = metrics[lens_id]["hidden_event_recall"]["mean"]
        y = 56 + index * 64
        bars.append(f'<text x="16" y="{y + 19}" font-size="13" fill="#263a43">{lens_id}</text><rect x="210" y="{y}" width="{max(2, value * 440):.1f}" height="28" rx="5" fill="{colors[index]}"/><text x="{218 + value * 440:.1f}" y="{y + 19}" font-size="13" fill="#263a43">{value:.2f}</text>')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="210" viewBox="0 0 720 210"><rect width="100%" height="100%" fill="#f7f5ef"/><text x="16" y="25" font-family="sans-serif" font-size="15" fill="#263a43">Held-out hidden-event recall</text>' + "".join(bars) + '<text x="210" y="194" font-size="11" fill="#60727a">0.00</text><text x="632" y="194" font-size="11" fill="#60727a">1.00</text></svg>'
    (output / "pilot-recall.svg").write_text(svg + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build parser for synthetic discovery, not the reusable core commands."""
    parser = argparse.ArgumentParser(prog="reflex experiments discovery", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="create a synthetic dataset with labels stored separately")
    generate.add_argument("--out", default=".reflex-demo")
    generate.add_argument("--seed", type=int, default=11)
    generate.add_argument("--entities", type=int, default=80)
    generate.add_argument("--days", type=int, default=30)
    generate.add_argument("--events-per-day", type=int, default=3)
    generate.add_argument("--evaluator-dir")

    demo = sub.add_parser("demo", help="generate a dataset and run its transaction profile")
    demo.add_argument("--out", default=".reflex-demo")
    demo.add_argument("--seed", type=int, default=11)
    demo.add_argument("--entities", type=int, default=80)
    demo.add_argument("--days", type=int, default=30)
    demo.add_argument("--events-per-day", type=int, default=3)
    demo.add_argument("--epochs", type=int, default=35)
    demo.add_argument("--evaluator-dir")

    discover = sub.add_parser("run", help="run discovery on an existing generated dataset")
    discover.add_argument("--data-dir", default=".reflex-demo")
    discover.add_argument("--epochs", type=int, default=35)
    discover.add_argument("--seed", type=int)
    discover.add_argument("--evaluator-dir", help="optional evaluator-only labels; omit for label-blind scoring")

    benchmark = sub.add_parser("benchmark", help="manage the synthetic A–E research protocol")
    bench = benchmark.add_subparsers(dest="benchmark_command", required=True)
    init = bench.add_parser("init")
    init.add_argument("--out", default="benchmark-protocol.json")
    init.add_argument("--seeds", default="11,23,37")
    init.add_argument("--objective", default="Find recurring or novel behaviour worth investigating, cite event evidence, and distinguish hypotheses from validated discoveries.")
    init.add_argument("--inference-budget", type=int, default=20000)
    init.add_argument("--compute-budget-seconds", type=int, default=900)
    record = bench.add_parser("record")
    record.add_argument("--protocol", required=True)
    record.add_argument("--record", required=True)
    report = bench.add_parser("report")
    report.add_argument("--protocol", required=True)
    bind = bench.add_parser("bind-inputs", help="pin the exact bytes for each seed and batch before runs")
    bind.add_argument("--protocol", required=True)
    bind.add_argument("--files", required=True, help="JSON mapping seed -> batch -> input file path")
    correlation = bench.add_parser("correlation-shift", help="compare correlation-aware lenses over paired seeds")
    correlation.add_argument("--out", default="reports/reflex-experiments/correlation-shift-v2")
    correlation.add_argument("--first-seed", type=int, default=101)
    correlation.add_argument("--seed-count", type=int, default=30)
    correlation.add_argument("--entities", type=int, default=40)
    correlation.add_argument("--days", type=int, default=60)
    pilot = bench.add_parser("pilot", help="run repeated synthetic detector evaluation")
    pilot.add_argument("--out", default="reports/reflex-pilot")
    pilot.add_argument("--seeds", default="11,23,37")
    pilot.add_argument("--entities", type=int, default=40)
    pilot.add_argument("--days", type=int, default=30)
    pilot.add_argument("--epochs", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one experiment-only command and return its exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        dataset = generate_synthetic(args.seed, args.entities, args.days, args.events_per_day)
        files = write_dataset(dataset, args.out, args.evaluator_dir)
        store = EventStore(Path(args.out) / "events.duckdb")
        try:
            inserted = store.add_events(dataset.events)
            total = store.count_native_events()
        finally:
            store.close()
        print(json.dumps({"files": files, "events_inserted": inserted, "events_total": total,
                          "warning": "keep evaluator labels outside agent workspaces during comparisons."}, indent=2))
        return 0
    if args.command == "demo":
        dataset = generate_synthetic(args.seed, args.entities, args.days, args.events_per_day)
        evaluator_dir = Path(args.evaluator_dir) if args.evaluator_dir else Path(args.out).parent / f"{Path(args.out).name}-evaluator"
        files = write_dataset(dataset, args.out, evaluator_dir)
        store = EventStore(Path(args.out) / "events.duckdb")
        try:
            store.add_events(dataset.events)
        finally:
            store.close()
        result = run_discovery(Path(args.out), epochs=args.epochs, evaluator_dir=evaluator_dir)
        print(json.dumps({"dataset_files": files, "run": result}, indent=2))
        return 0
    if args.command == "run":
        result = run_discovery(Path(args.data_dir), epochs=args.epochs, seed=args.seed,
                               evaluator_dir=Path(args.evaluator_dir) if args.evaluator_dir else None)
        print(json.dumps(result, indent=2))
        return 0
    if args.benchmark_command == "init":
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("provide one or more distinct random seeds")
        manifest = protocol_manifest(seeds=seeds, objective=args.objective,
                                     inference_budget=args.inference_budget,
                                     compute_budget_seconds=args.compute_budget_seconds)
        Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"protocol": str(Path(args.out).resolve()), "variants": list("ABCDE"), "seeds": seeds}, indent=2))
        return 0
    if args.benchmark_command == "record":
        append_run_record(args.protocol, _json(args.record))
        print(json.dumps({"recorded": True, "protocol": str(Path(args.protocol).resolve())}))
        return 0
    if args.benchmark_command == "report":
        print(json.dumps(protocol_report(args.protocol), indent=2))
        return 0
    if args.benchmark_command == "bind-inputs":
        files_path = Path(args.files)
        file_map = _json(args.files)
        hashes = {}
        for seed, batches in file_map.items():
            hashes[str(seed)] = {}
            for batch, input_file in batches.items():
                resolved = (files_path.parent / input_file).resolve()
                try:
                    content = resolved.read_bytes()
                except OSError as error:
                    raise ValueError(f"cannot read input bytes for seed {seed}, batch {batch}: {input_file}") from error
                hashes[str(seed)][batch] = hashlib.sha256(content).hexdigest()
        bind_protocol_input_hashes(args.protocol, hashes)
        print(json.dumps({"input_hashes_bound": True, "seed_count": len(hashes),
                          "batch_count_per_seed": len(next(iter(hashes.values()), {})),
                          "protocol": str(Path(args.protocol).resolve())}, indent=2))
        return 0
    if args.benchmark_command == "correlation-shift":
        report = run_correlation_benchmark(first_seed=args.first_seed, seed_count=args.seed_count,
                                           entities=args.entities, days=args.days)
        json_path, markdown_path = write_correlation_benchmark(report, args.out)
        print(json.dumps({"report": str(json_path.resolve()), "summary": str(markdown_path.resolve()),
                          "seeds": report["protocol"]["seeds"],
                          "f1": {name: value["f1"] for name, value in report["methods"].items()}}, indent=2))
        return 0
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide one or more distinct random seeds")
    report = _run_pilot(seeds, entities=args.entities, days=args.days, epochs=args.epochs,
                        output=Path(args.out))
    print(json.dumps({"report": str((Path(args.out) / "pilot-results.json").resolve()),
                      "runs": report["runs"], "seeds": report["seeds"], "metrics": report["metrics"]}, indent=2))
    return 0


__all__ = ["build_parser", "main"]
