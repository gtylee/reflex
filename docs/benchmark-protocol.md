# REFLEX A–E discovery comparison protocol

The A–E discovery protocol measures pattern discovery and validation. The A/B/C portfolio-news protocol measures news routing in a different task. Keep their labels, datasets, and conclusions separate.

## REFLEX portfolio-news A/B/C protocol

The variants are A (direct LLM processing), B (tool-enabled agent processing), and C (`portfolio_news_intelligence@1.0`). `reflex experiments portfolio-benchmark init` writes the fixed synthetic input hash, portfolio state, objective, information scope, inference and compute ceilings, and required metrics. Each run must use that same hash and exact objective. The run recorder rejects mismatches and over-budget runs. The older top-level `reflex benchmark` route remains available as a compatibility alias.

```bash
uv run reflex experiments portfolio-benchmark init --out reflex-benchmark.json --seeds 11,23,37
uv run reflex experiments portfolio-benchmark run-c --protocol reflex-benchmark.json --out .reflex-benchmark-c --seed 11
uv run reflex experiments portfolio-benchmark report --protocol reflex-benchmark.json
```

Record actual A or B runs with `uv run reflex experiments portfolio-benchmark record --protocol reflex-benchmark.json --run run.json`. Required metrics are alert precision and recall, duplicate suppression, decision consistency, latency, inference-token count, compute time, evidence completeness, replay reproducibility, and action reliability. The deterministic C run has zero inference-token use; its local compute time is recorded. The recorder requires reported token and compute metrics to match the run's resource usage. The benchmark action sink is local and synthetic for all variants. External delivery is disabled for a fair comparison.

The harness can execute C and validate records, but it does not provide an LLM or tool-agent baseline. An A/B/C report is not comparison-ready until every variant has an actual run for every declared seed. Partial seed coverage is labeled as partial. No missing result is filled, inferred, or presented as a model result.

The benchmark asks whether persistent, governed state improves discovery quality or efficiency over a general-purpose coding agent with equivalent event access and analytical capability.

## Variants

| Variant | System |
|---|---|
| A | General-purpose coding agent with raw events |
| B | General-purpose agent with raw events and structured persistent state |
| C | General-purpose agent with raw events and shared autoencoder/Isolation Forest observations |
| D | REFLEX with shared statistical lenses, persistent state, policy, and curation |
| E | Unrestricted general-purpose agent with persistent workspace and self-built analytical infrastructure |

## Controls

1. Use identical event bytes, event-time order, batch boundaries, objective wording, and random seeds for every variant.
2. Hold synthetic ground truth in the evaluator directory. Do not place it in any agent's input or workspace.
3. Give every system the same inference-token and non-inference compute budgets. Record model/provider, token counts, inference cost, elapsed time, and compute time where available.
4. If C and D receive shared statistical scores, provide identical feature definitions, train/calibration windows, scalers, weights, and thresholds.
5. Give B–E persistent storage across every successive batch. Record what survives between batches.
6. Allow A, B, C, and E to build detectors and other analysis tools within budget. Do not restrict their SQL, scripting, or statistical methods to match a preferred REFLEX workflow.
7. Run both one-shot discovery and longitudinal discovery. For the longitudinal arm, provide one five-day batch at a time and retain each agent's permitted workspace.
8. Record raw inputs, outputs, evidence references, costs, investigations, and workspace commit/hash. Do not retrospectively fill missing values with estimates.

## Run manifest and recording

```bash
uv run reflex experiments discovery benchmark init --out benchmark-protocol.json --seeds 11,23,37 --inference-budget 20000 --compute-budget-seconds 900
```

For each variant, seed, and batch, write the actual run to a JSON file with these fields:

```json
{
  "variant": "D",
  "seed": 11,
  "batch": "batch-1",
  "input_hash": "sha256-of-exact-batch-input",
  "objective": "exact shared objective text",
  "output": {"validated_discoveries": 0, "hypotheses": []},
  "inference_tokens": 0,
  "inference_cost": 0.0,
  "compute_seconds": 0,
  "investigation_count": 0,
  "evidence_refs": [],
  "workspace_ref": "commit-or-artifact-reference"
}
```

Record and report actual observations:

```bash
uv run reflex experiments discovery benchmark record --protocol benchmark-protocol.json --record run.json
uv run reflex experiments discovery benchmark report --protocol benchmark-protocol.json
```

The report marks the protocol complete only after it has all six five-day batches plus the one-shot run for each requested seed and each of the five variants.

## Independent scoring

Deduplicate outputs that describe the same entity/cohort, event-time window, and underlying relationship. Score hidden mechanism recovery, event-window precision/recall, false discovery rate, out-of-sample generalization, and mechanism diversity using labels unavailable to agents. Have an independent reviewer adjudicate non-synthetic claims and ambiguous semantic duplicates while blind to the variant. Preserve the adjudication evidence.

Calculate per-seed quality and cost before pooling. Report means and uncertainty intervals across seeds, each variant's cost per independently validated discovery, rediscovery count, investigation latency, lens reuse, regime-shift adaptation, and novel-mechanism sensitivity. Report governance measures from the audit traces: evidence completeness, reproducibility, version consistency, budget compliance, and unsupported knowledge promotion.

## Interpretation rule

The synthetic pilot is only a pipeline check. A successful detector run says nothing by itself about the A–E hypothesis. Draw no superiority conclusion from a single seed, raw anomaly counts, unreviewed explanations, or a run with unequal budgets. Keep missing agent runs missing; the report code never fabricates them.
