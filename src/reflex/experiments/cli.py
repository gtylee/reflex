"""Experiment, benchmark, and synthetic demonstration command handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .portfolio_benchmark import append_run, benchmark_manifest, report, run_reflex_variant
from .news_demo import run_news_demo
from .adapters import GaussianMixtureAdapter, IsolationForestAdapter, default_phase_one_adapters
from .generators import SCENARIOS, default_configs, generate_scenario
from .runner import run_comparison, write_report
from .performance import benchmark_event_store
from .public_replay import replay_gdelt_day
from .heldout import run_heldout_benchmark, write_heldout_report
from .visualizations import write_metric_svg, write_news_timeline_svg
from .workflows import (PortfolioExposure, RepairSandbox, diagnose_incident,
                        SimulatedOperationsSystem, evaluate_topic_association,
                        process_news_timeline,
                        synthetic_feedback, synthetic_incidents, synthetic_news_outcomes,
                        synthetic_news_story)


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _add_portfolio_benchmark_commands(parent: Any, name: str) -> None:
    """Add the portfolio-news protocol under legacy and namespaced routes."""
    benchmark = parent.add_parser(name, help="prepare or record fair A/B/C portfolio-news runs")
    bench_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    init = bench_sub.add_parser("init", help="write an empty shared-input protocol")
    init.add_argument("--out", default="reflex-benchmark.json")
    init.add_argument("--seeds", default="11,23,37")
    init.add_argument("--inference-token-budget", type=int, default=20000)
    init.add_argument("--compute-budget-seconds", type=int, default=900)
    init.add_argument("--objective", default="Identify portfolio-relevant developments, cite event evidence, suppress duplicates, and state uncertainty.")
    record = bench_sub.add_parser("record", help="record one actual A, B, or C result")
    record.add_argument("--protocol", required=True)
    record.add_argument("--run", required=True, help="JSON file containing one completed run record")
    run_c = bench_sub.add_parser("run-c", help="run and record the REFLEX C variant")
    run_c.add_argument("--protocol", required=True)
    run_c.add_argument("--out", default=".reflex-benchmark-c")
    run_c.add_argument("--seed", type=int, default=11)
    summary = bench_sub.add_parser("report", help="summarize recorded runs without filling missing baselines")
    summary.add_argument("--protocol", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflex",
        description="REFLEX experiment and benchmark workflows",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo-news", help="run the synthetic portfolio news Reaction")
    demo.add_argument("--out", default=".reflex-news-demo")
    _add_portfolio_benchmark_commands(sub, "benchmark")

    experiments = sub.add_parser("experiments", help="run seeded research experiments")
    experiment_sub = experiments.add_subparsers(dest="experiment_command", required=True)
    experiment_demo = experiment_sub.add_parser("demo-news", help="run the synthetic portfolio news Reaction")
    experiment_demo.add_argument("--out", default=".reflex-news-demo")
    _add_portfolio_benchmark_commands(experiment_sub, "portfolio-benchmark")
    discovery = experiment_sub.add_parser("discovery", help="run the transaction-shaped synthetic discovery profile")
    discovery.add_argument("discovery_args", nargs=argparse.REMAINDER,
                           help="generate, run, or benchmark a synthetic discovery experiment")
    transaction_replay = experiment_sub.add_parser("transaction-replay", help="replay the example transaction high-amount Lens")
    transaction_replay.add_argument("--db", default="events.duckdb")
    transaction_replay.add_argument("--state", default="state.sqlite")
    transaction_replay.add_argument("--batch-days", type=int, default=5)
    transaction_replay.add_argument("--start")
    transaction_replay.add_argument("--end")
    transaction_replay.add_argument("--limit", type=int, default=1000)
    for command, label in (("ingest-bpi", "BPI Challenge 2017"), ("ingest-loghub", "Loghub")):
        adapter = experiment_sub.add_parser(command, help=f"ingest a local {label} event file")
        adapter.add_argument("path")
        adapter.add_argument("--db", default="events.duckdb")
    experiment_sub.add_parser("list", help="list registered phase-one synthetic cases")
    generate = experiment_sub.add_parser("generate", help="generate a label-blind synthetic event stream")
    generate.add_argument("scenario", choices=sorted(SCENARIOS))
    generate.add_argument("--seed", type=int, default=11)
    generate.add_argument("--entities", type=int, default=32)
    generate.add_argument("--days", type=int, default=30)
    generate.add_argument("--out", required=True)
    generate.add_argument("--evaluator-dir")
    run = experiment_sub.add_parser("run", help="run registered adapters and score in evaluator output")
    run.add_argument("scenario", choices=sorted(SCENARIOS))
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--entities", type=int, default=32)
    run.add_argument("--days", type=int, default=30)
    run.add_argument("--variants", default="A,B,C,D,E,F,G")
    run.add_argument("--detector", choices=("quantile", "isolation-forest", "gaussian-mixture"), default="quantile")
    run.add_argument("--out", required=True)
    run.add_argument("--evaluator-dir")
    heldout = experiment_sub.add_parser("heldout", help="evaluate variant C over unseen seeds for synthetic cases")
    heldout.add_argument("--first-seed", type=int, default=101)
    heldout.add_argument("--seed-count", type=int, default=30)
    heldout.add_argument("--entities", type=int, default=40)
    heldout.add_argument("--days", type=int, default=60)
    heldout.add_argument("--scenarios", default=",".join(SCENARIOS))
    heldout.add_argument("--out", required=True)
    news = experiment_sub.add_parser("news", help="simulate an evolving news story and chronological market outcomes")
    news.add_argument("--seed", type=int, default=11)
    news.add_argument("--out", required=True)
    operations = experiment_sub.add_parser("operations", help="run simulated incident diagnosis and sandbox repair")
    operations.add_argument("--seed", type=int, default=11)
    operations.add_argument("--count", type=int, default=40)
    operations.add_argument("--out", required=True)
    performance = experiment_sub.add_parser("performance", help="measure local event-store scaling")
    performance.add_argument("--sizes", default="1000,10000,100000")
    performance.add_argument("--batch-size", type=int, default=2000)
    performance.add_argument("--out", required=True)
    gdelt = experiment_sub.add_parser("gdelt", help="download, normalize, and replay one official GDELT day")
    gdelt.add_argument("--date", required=True, help="archive date in YYYYMMDD format")
    gdelt.add_argument("--limit", type=int, default=10_000)
    gdelt.add_argument("--batch-size", type=int, default=2_000)
    gdelt.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an experiment command; core commands are routed by ``reflex.cli``."""
    command_args = sys.argv[1:] if argv is None else argv
    if command_args and command_args[0] in {"demo", "generate", "discover", "discovery-benchmark"}:
        from .discovery_cli import main as discovery_main

        return discovery_main(command_args)
    if command_args and command_args[0] in {"ingest-bpi", "ingest-loghub"}:
        from .log_cli import main as ingestion_main

        format_name = "bpi" if command_args[0] == "ingest-bpi" else "loghub"
        return ingestion_main([format_name, *command_args[1:]])
    args = build_parser().parse_args(command_args)
    if args.command == "experiments" and args.experiment_command == "discovery":
        from .discovery_cli import main as discovery_main

        return discovery_main(args.discovery_args)
    if args.command == "experiments" and args.experiment_command == "transaction-replay":
        from reflex.core.cli import main as core_main
        from .transaction_lenses import TRANSACTION_HIGH_AMOUNT_SQL

        replay_args = ["replay", "--db", args.db, "--state", args.state,
                       "--batch-days", str(args.batch_days), "--limit", str(args.limit),
                       "--sql", TRANSACTION_HIGH_AMOUNT_SQL]
        if args.start:
            replay_args.extend(("--start", args.start))
        if args.end:
            replay_args.extend(("--end", args.end))
        return core_main(replay_args)
    if args.command == "experiments" and args.experiment_command in {"ingest-bpi", "ingest-loghub"}:
        from .log_cli import main as ingestion_main

        format_name = "bpi" if args.experiment_command == "ingest-bpi" else "loghub"
        return ingestion_main([format_name, args.path, "--db", args.db])
    if args.command == "demo-news" or (args.command == "experiments" and args.experiment_command == "demo-news"):
        print(json.dumps(run_news_demo(args.out), indent=2))
        return 0
    if args.command == "experiments" and args.experiment_command != "portfolio-benchmark":
        if args.experiment_command == "list":
            configs = default_configs()
            print(json.dumps([{"family": item.family, "scenario": item.scenario,
                               "config_id": item.config_id, "version": item.version,
                               "objective": item.objective} for item in configs], indent=2))
            return 0
        if args.experiment_command == "news":
            output = Path(args.out)
            output.mkdir(parents=True, exist_ok=True)
            story = synthetic_news_story(args.seed)
            portfolio = PortfolioExposure("portfolio-demo", {"rare_earths": 0.32,
                                                                  "materials_equity": 0.18,
                                                                  "industrial_metals": 0.12})
            feedback = synthetic_feedback(args.seed, count=6)
            reactions, state = process_news_timeline(story, feedback, portfolio)
            outcomes = synthetic_news_outcomes(args.seed, events=1200)
            evaluation = evaluate_topic_association(outcomes)
            (output / "story-events.jsonl").write_text("".join(json.dumps(event.model_dump(mode="json")) + "\n" for event in story))
            (output / "feedback-events.jsonl").write_text("".join(json.dumps(event.model_dump(mode="json")) + "\n" for event in feedback))
            (output / "market-outcome-events.jsonl").write_text("".join(json.dumps(event.model_dump(mode="json")) + "\n" for event in outcomes))
            summary = {"synthetic_only": True, "story_event_count": len(story), "story_reactions": reactions,
                       "portfolio_id": portfolio.portfolio_id, "feedback_event_count": len(feedback),
                       "chronological_market_evaluation": evaluation,
                       "market_outcome_event_count": len(outcomes), "state": state,
                       "feedback_events_processed_in_event_time_order": len(feedback),
                       "note": "Associations are predictive diagnostics, not causal estimates."}
            (output / "news-report.json").write_text(json.dumps(summary, indent=2) + "\n")
            chart_path = write_news_timeline_svg(output / "news-report.json", output / "news-story-timeline.svg")
            summary["visualization"] = str(chart_path)
            (output / "news-report.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"report": str((output / "news-report.json").resolve()), **summary}, indent=2))
            return 0
        if args.experiment_command == "operations":
            output = Path(args.out)
            output.mkdir(parents=True, exist_ok=True)
            dataset = synthetic_incidents(args.seed, args.count)
            sandbox = RepairSandbox.create(SimulatedOperationsSystem.create(dataset.causes_by_event))
            traces = []
            correct = 0
            verified = 0
            for event in dataset.events:
                diagnosis = diagnose_incident(event)
                is_correct = diagnosis == dataset.causes_by_event[event.event_id]
                correct += int(is_correct)
                record = sandbox.repair(event, diagnosis=diagnosis or "undetermined",
                                        proposed_action=f"sandbox:{diagnosis or 'escalate'}",
                                        approved=False)
                verified += int(record["status"] == "verified")
                traces.append(record)
            (output / "incidents.jsonl").write_text("".join(json.dumps(event.model_dump(mode="json")) + "\n" for event in dataset.events))
            summary = {"synthetic_only": True, "sandbox_only": True, "incident_count": len(dataset.events),
                       "diagnosis_accuracy": correct / len(dataset.events) if dataset.events else 0.0,
                       "verified_repairs": verified,
                       "awaiting_human_approval": sum(item["status"] == "awaiting_approval" for item in traces),
                       "traces": traces,
                       "note": "The transparent rule baseline is deterministic. Repairs are verified against symptom changes in the simulated operations system. Hidden causes are used only to score diagnosis accuracy."}
            (output / "operations-report.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"report": str((output / "operations-report.json").resolve()), **summary}, indent=2))
            return 0
        if args.experiment_command == "performance":
            sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
            result = benchmark_event_store(sizes, batch_size=args.batch_size, output_dir=args.out)
            print(json.dumps({"report": str((Path(args.out) / "performance.json").resolve()), **result}, indent=2))
            return 0
        if args.experiment_command == "gdelt":
            print(json.dumps(replay_gdelt_day(args.date, args.out, limit=args.limit,
                                              batch_size=args.batch_size), indent=2))
            return 0
        if args.experiment_command == "heldout":
            selected_scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
            report = run_heldout_benchmark(
                first_seed=args.first_seed, seed_count=args.seed_count,
                entities=args.entities, days=args.days, scenarios=selected_scenarios,
            )
            json_path, markdown_path = write_heldout_report(report, args.out)
            print(json.dumps({
                "json_report": str(json_path.resolve()),
                "markdown_report": str(markdown_path.resolve()),
                "seeds": report["protocol"]["seeds"],
                "scenarios": {name: result["per_seed_metric_summary"]["f1"]
                              for name, result in report["scenarios"].items()},
                "expert_labeled": False,
                "comparative_claim_allowed": False,
            }, indent=2))
            return 0
        config = next(item for item in default_configs(seed=args.seed, entities=args.entities, days=args.days)
                      if item.scenario == args.scenario)
        dataset = generate_scenario(config)
        if args.experiment_command == "generate":
            print(json.dumps(dataset.write(args.out, args.evaluator_dir), indent=2))
            return 0
        adapters = default_phase_one_adapters(config)
        if args.detector == "isolation-forest":
            adapters["C"] = IsolationForestAdapter(tuple(config.parameters["model_features"]))
        elif args.detector == "gaussian-mixture":
            adapters["C"] = GaussianMixtureAdapter(tuple(config.parameters["model_features"]))
        selected = [value.strip() for value in args.variants.split(",") if value.strip()]
        output_root = Path(args.out)
        evaluator_root = Path(args.evaluator_dir) if args.evaluator_dir else output_root.parent / f"{output_root.name}-evaluator"
        paths = dataset.write(output_root / "input", evaluator_root)
        result = run_comparison(config, dataset, adapters, variants=selected)
        report_path = write_report(result, evaluator_root / "results.json")
        chart_path = write_metric_svg(result, evaluator_root / "results.svg")
        summary = {"scenario": config.scenario, "input_hash": result.input_hash,
                   "comparison_ready": result.comparison_ready,
                   "readiness_reasons": result.readiness_reasons,
                   "runs": [{"variant": item.variant_id, "status": item.status,
                             "execution_kind": item.execution_kind, "metrics": item.metrics,
                             "error": item.error} for item in result.runs],
                   "input_events": paths["events"], "evaluator_results": str(report_path.resolve()),
                   "evaluator_visualization": str(chart_path.resolve()),
                   "evaluator_truth": paths["evaluator_truth"]}
        print(json.dumps(summary, indent=2))
        return 0
    is_portfolio_benchmark = args.command == "benchmark" or (
        args.command == "experiments" and args.experiment_command == "portfolio-benchmark"
    )
    if not is_portfolio_benchmark:
        raise ValueError(f"unsupported command: {args.command}")
    if args.benchmark_command == "init":
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        manifest = benchmark_manifest(seeds=seeds, objective=args.objective,
                                      inference_token_budget=args.inference_token_budget,
                                      compute_budget_seconds=args.compute_budget_seconds)
        Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"protocol": manifest["protocol"], "input_hash": manifest["input_hash"],
                          "variants": manifest["variants"], "path": str(Path(args.out).resolve())}, indent=2))
        return 0
    if args.benchmark_command == "record":
        manifest = append_run(args.protocol, _read_json(args.run))
        print(json.dumps({"recorded_runs": len(manifest["runs"]), "baseline_status": manifest["baseline_status"]}, indent=2))
        return 0
    if args.benchmark_command == "run-c":
        print(json.dumps(run_reflex_variant(args.protocol, args.out, seed=args.seed), indent=2))
        return 0
    if args.benchmark_command == "report":
        print(json.dumps(report(args.protocol), indent=2))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
