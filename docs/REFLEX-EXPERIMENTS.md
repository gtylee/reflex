# REFLEX experimental suite

## Purpose and current status

This framework tests the REFLEX hypothesis without assuming that it wins. It keeps generated labels out of adapter inputs, shares an input hash and resource budget across variants, records unavailable baselines as unavailable, and stores evaluator outputs separately from agent-visible event streams.

This is a research harness, not a complete evaluation of the seven configurations. The current repository has working synthetic cases, deterministic analytical adapters, a state and curation contract, news and operations simulators, row mappers for public datasets, a date-pinned GDELT replay, repeated-run measurement, and an opt-in event-store benchmark. It does **not** yet run a conventional LLM, a tool-enabled LLM, an actual Eliza agent, or a public-data relevance/market-outcome experiment. Those results remain unmeasured.

## Run the experiments

```bash
# See the six seeded synthetic discovery cases
uv run reflex experiments list

# Compare available adapters on the same event bundle; A and B remain absent
uv run reflex experiments run exposure_drift --entities 32 --days 60 \
  --seed 11 --variants A,B,C --out reports/reflex-experiments/exposure-drift

# Evaluate deterministic C on 30 new seeds for each synthetic case
uv run reflex experiments heldout --first-seed 101 --seed-count 30 \
  --entities 40 --days 60 --out reports/reflex-experiments/heldout-seeds-101-130

# Try other deterministic analytical lenses from Python
uv run python -c 'from reflex.experiments import *'

# Evolving story, portfolio routing, feedback events, and chronological topic/return split
uv run reflex experiments news --seed 11 --out reports/reflex-experiments/news

# Download one official GDELT archive, select 10k chronological events, and replay them
uv run reflex experiments gdelt --date 20261123 --limit 10000 \
  --batch-size 2000 --out reports/reflex-experiments/gdelt-20261123

# Simulated operational diagnosis and sandbox repair
uv run reflex experiments operations --seed 11 --count 40 \
  --out reports/reflex-experiments/operations

# Local append, query, replay, and memory measurements. One million events is opt-in.
uv run reflex experiments performance --sizes 1000,10000,100000 \
  --batch-size 2000 --out reports/reflex-experiments/performance

# The suite itself
uv run pytest -q
```

The Python adapters are importable from `reflex.experiments`. For repeated runs, `repeat_adapter(adapter, dataset, budgets, repeats=20)` records structured outcome disagreement, F1, compute time, and token counts for the same label-blind case. A real agent adapter must supply its execution kind as `external`, plus a model version and sampling parameters. B must also record tool versions; A/B model and sampling settings must match before the report says the comparison is ready. Fixtures say `mock` and are excluded.

The held-out command uses new generator seeds 101–130, reports exact event-ID scores and resamples seeds for uncertainty intervals. Its evidence checks measure whether references exist, share an entity with the prediction, and occur no later than it. These checks do not determine causal sufficiency. Labels remain synthetic oracle labels; expert-labeled public or operational cases are not available in this workspace, and the command does not report a comparative win.

### Held-out result, 30 seeds per scenario

Run with 40 entities and 60 days per case. F1 is exact event-level and the intervals resample seeds.

| Scenario | Test truth events | F1, mean (95% interval) | Evidence integrity |
|---|---:|---:|---:|
| Exposure drift | 3,000 | 0.999 (0.998–1.000) | 1.000 |
| Collateral deterioration | 3,000 | 1.000 (1.000–1.000) | 1.000 |
| Rare sequence | 300 | 1.000 (1.000–1.000) | 1.000 |
| Correlation shift | 4,800 | 0.274 (0.265–0.282) | 1.000 |
| Spurious confounder | 3,935 | 1.000 (1.000–1.000) | 1.000 |
| Unseen mechanism | 3,000 | 1.000 (1.000–1.000) | 1.000 |

The low correlation-shift score is a useful failure signal. Variant C currently applies a univariate quantile rule to the product feature in this scenario. Evidence integrity is high because the adapter cites the flagged event itself; it is not a causal-evidence score. Full per-seed results and precision/recall are in [`heldout-report.json`](../reports/reflex-experiments/heldout-seeds-101-130/heldout-report.json).

## Experiment register

| Family | Hypothesis and design | Truth and baseline | Current result or limitation |
|---|---|---|---|
| 1A exposure drift | A time-aware lens can detect increasing exposure before the static limit is crossed. Seeded entity streams are split into train, calibration, and later test windows. | Target entities and post-shift event IDs live in evaluator labels. The quantile lens is runnable; A/B are adapters still needed. | Seed 11, 32 entities, 60 days: C produced 80/80 true positives, no false positives, F1 1.0 on the held-out truth. The confidence interval is degenerate because this single synthetic run made no errors. This is a smoke run, not a comparative result. |
| 1B collateral | Several modest, correlated changes can combine into deterioration. | Stress labels are generated separately from event attributes. Quantile, Isolation Forest, GMM, and autoencoder lenses are available. | Implemented and invariant-tested; no multi-seed result table yet. |
| 1C rare sequence | A stateful temporal rule can detect a failure only after an ordered event sequence. | Sequence completion events are evaluator labels; sequence lens sees events only. | Generator and temporal lens are implemented; no comparative result yet. |
| 1D correlation shift | Correlation changes can be detected while marginal distributions remain similar. | Regime event IDs are evaluator labels. Rolling correlation and multivariate lenses are available. | Generator invariants and adapter smoke-tested; no aggregate result. |
| 1E spurious confounder | A relationship that breaks out of sample should not be curated as stable knowledge. | Post-shift true-driver cases are labeled outside the input. | Scenario exists; automated curation contamination/recovery experiment still needs a full repeated-batch runner. |
| 1F unseen mechanism | New relationships after the midpoint should be discoverable without relying only on known thresholds. | Only novel relationship events are labeled. | Scenario exists; no validated novelty-versus-known-lens comparison yet. |
| 2 news and outcomes | Deterministic relevance routing can suppress duplicate work while reopening on meaningful updates; topic associations must be tested chronologically. | Seven-event rare-earth story, direct and indirect portfolio exposure, feedback events, and 1,200 synthetic news/outcome pairs. | Smoke run suppressed the duplicate, deferred a contradiction, honored a manager's later investigation request after false-positive feedback raised the notification threshold, and reopened on retraction. On seed 11, train association was about +0.015, later association about -0.0037, and held-out MAE about 0.0178. That is a synthetic regime-break example, not a market claim. |
| 2E public dataset replay | The event store can ingest and replay public chronological events reproducibly at moderate volume. | Official GDELT Events 2.0 daily archive, SHA-256 pinned in the report; sampled 10,000 chronological events from 101,301 rows. | 10,000 events replayed in event-time order in about 6.0 seconds (about 1,661 events/s); duplicate replay inserted zero rows. This tests ingestion and replay only; it does not score news relevance, story quality, or market prediction. |
| 3 operations | Diagnosis, validation, approval, retry, idempotency, rollback, and verification can be traced without a live system. | Incident causes are held in evaluator-only data; events expose symptoms. | Rule diagnostic matched 40/40 generated incidents. 36 sandbox repairs verified; four were held for approval. The rules were authored from the simulator's symptom catalogue, so 100% is not evidence of generalization. |
| 4 persistent state | Identical events can be interpreted differently using history; state should survive restart and isolate tenants. | State snapshots, event ids, and state transitions are auditable. | Covered by restart, historical snapshot, duplicate replay, event-time ordering, state-dependent interpretation, and tenant-key tests. Cross-process concurrent writer behavior is not evaluated. |
| 5 curation | A proposed lens should move through validation and approval before activation, then be versioned and retireable. | Held-out labels gate a candidate. The existing LensCandidate lifecycle is used. | Candidate lifecycle and rejection tests pass. Repeated-batch learning curves, contamination propagation, and replacement under drift remain future experiments. |
| 6 variance | REFLEX may reduce business-outcome variation while maintaining correctness. | Same input/budget is repeated per adapter; correctness is scored separately from outcome disagreement. | Repetition API is implemented, but no real stochastic model adapter is installed. No 100-event × 20-run claim is made. |
| 7 Eliza | A portable event interface should support an Eliza host. | `ElizaHarness` is the boundary; `MockElizaHarness` is explicitly mock. | No Eliza workspace or runtime is present in this repository. There is no production integration result. |
| Performance | Measure where append, query, replay, and Python allocation costs arise before redesigning storage. | Synthetic immutable events, no model calls; machine-specific measurements. | On macOS 26.5 arm64, Python 3.13.3, DuckDB 1.5.6, the 100k run wrote about 1,154 events/s and a 2,000-row batch p50/p95 near 1.63/1.76s; the DuckDB file plus journal was about 30.9 MB. This includes Python event construction, JSON serialization, and `tracemalloc`, and is only a smoke benchmark. Exact measurements and the chart are in `reports/reflex-experiments/performance/`. No SQL-only, vectorized, stateful, or alternative batching comparison has been run. |

Performance chart: [event-store throughput by run size](../reports/reflex-experiments/performance/performance.svg). Full measurements are in [`performance.json`](../reports/reflex-experiments/performance/performance.json).

## Interpreting results

- An event prediction is scored against evaluator labels after an explicit calibration boundary. Missing A/B/D/E/F/G adapters are not silently replaced with mocks.
- The synthetic event generators are deterministic for a fixed config, but their mechanisms are intentionally simple and often visible as patterns in the features. High synthetic F1 does not predict real-world detection quality.
- The news-to-market simulator has a known regime break and random returns. Topic/return statistics are predictive diagnostics only. They cannot support causal language.
- The operations baseline is a transparent rule table built from the simulator's symptom fields. It measures that the action state machine handles cases, not that an agent can autonomously investigate real workflows.
- Bootstrap intervals in a single synthetic case describe event sampling under that case. They do not capture variation across scenarios, datasets, model versions, or market regimes. Multi-seed paired runs are still needed.
- Performance numbers must be regenerated on the target host and include build/runtime versions before being compared with another implementation.

## Next experiments needed for comparative conclusions

1. Add actual A/B and D/E/F adapters behind the existing common-input contract. Pin model/tool versions, sampling parameters, prompt and policy versions, and budgets in each run record.
2. Run the six discovery generators over a pre-registered seed set and paired baseline inputs; report per-mechanism and aggregate distributions.
3. Finish a batch curation protocol for false relationships, regime breaks, and reusable reactions. Include both held-out validation and explicit retirement/recovery.
4. Extend the GDELT ingestion smoke run to a relevance-annotated event benchmark; then load and pin a legally usable matched news/price sample, aligning news availability time to next-observable returns and preserving source timestamps, adjustments, and data vintages.
5. Implement persistent state in each orchestration arm and run the same 100+ cases 20+ times for stochastic arms. Report business decision disagreement, action disagreement, correctness, costs, and replay fidelity separately.
6. Add SQL/vectorized and stateless/stateful throughput baselines before tuning the architecture; test 1M events only on suitable hardware.
7. Connect an actual Eliza execution harness before marking configuration G complete.

## Architecture implications

The existing event/state/Reaction abstractions are sufficient for seeded discovery, immutable event replay, bounded local actions, and candidate status transitions. The smallest gaps exposed by this suite are experimental rather than core-model redesigns: external provenance is checked in the comparison readiness gate but is not yet a typed run-result contract; a trace exporter should capture per-event state snapshots and action decisions; and scenario/dataset manifests need a stable public-data source/version record. Real A/B/F and Eliza adapters can be added at the experiment boundary without making Eliza a REFLEX dependency.
