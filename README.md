# REFLEX

REFLEX turns an event stream into evidence, a decision, and a bounded action. It keeps incoming events immutable, records derived findings separately, and requires explicit policy and approval before an action runs.

## Reusable core and experiments

`reflex.core` contains the reusable event records, stores, SQL Lens runner, schema-driven statistical fitting/scoring, policy, and bounded investigation tools. It accepts normalized `Event` and `EventWindowFeatures` records and does not assume a transaction schema or read evaluator labels.

`reflex.experiments` contains dataset adapters, synthetic generators, transaction-specific features, example lenses, benchmark labels, evaluation, and report commands. Those profiles compose the core for a particular study. New application code should import reusable operations from `reflex.core` and import a profile only when it needs that dataset or experiment.

The CLI follows the same boundary: use `reflex replay --sql-file lens.sql` for a caller-defined core Lens, and `reflex experiments ...` for prepared examples such as `reflex experiments transaction-replay` and `reflex experiments discovery demo`.

## Get started

```bash
uv sync --python 3.13 --extra dev
uv run reflex experiments demo-news --out .reflex-news-demo
uv run reflex experiments discovery demo --out .reflex-demo
```

The news demo uses synthetic data and a local simulated publisher. The discovery demo creates synthetic events, fits statistical lenses on earlier data, and scores against hidden labels only after predictions are complete. Neither demo connects to a live portfolio, messaging, or operations system.

## Run the experiments

```bash
# List the six synthetic discovery scenarios
uv run reflex experiments list

# Score exact event IDs across 30 new seeds in each scenario
uv run reflex experiments heldout --first-seed 101 --seed-count 30 \
  --entities 40 --days 60 --out reports/reflex-experiments/heldout-seeds-101-130

# Run the synthetic news and operations examples
uv run reflex experiments news --out reports/reflex-experiments/news
uv run reflex experiments operations --count 40 --out reports/reflex-experiments/operations

# Replay a selected official GDELT event archive
uv run reflex experiments gdelt --date 20261123 --limit 10000 \
  --out reports/reflex-experiments/gdelt-20261123

# Run the full automated suite
uv run pytest
```

The held-out command uses new synthetic generator seeds. It reports exact event matches and checks evidence references for input membership, same-entity alignment, and event-time order. Its labels are synthetic, not expert judgments. Results do not establish real-world performance or a win over an external agent.

## Compare REFLEX approaches

The A/B/C protocol keeps the input, objective, and resource limits fixed. The local command can run C. A and B need actual external runs; missing runs remain missing.

```bash
uv run reflex experiments portfolio-benchmark init --out reflex-benchmark.json --seeds 11,23,37
uv run reflex experiments portfolio-benchmark run-c --protocol reflex-benchmark.json --out .reflex-benchmark-c --seed 11
uv run reflex experiments portfolio-benchmark report --protocol reflex-benchmark.json
```

For the separate A–E statistical-discovery protocol, use its experiment command:

```bash
uv run reflex experiments discovery benchmark pilot --out reports/reflex-pilot --seeds 11,23,37
```

## Code map

| Location | What it contains |
|---|---|
| `src/reflex/core/models/` | Immutable event, Lens, policy, and feature types, grouped by domain |
| `src/reflex/experiments/models/` | Experiment inputs, label-separated records, adapter outputs, and reports |
| `src/reflex/core/storage.py` | Event ledger, durable state, and audit trail |
| `src/reflex/core/features.py` | Generic feature-row to matrix conversion |
| `src/reflex/core/discovery.py` | Schema-driven statistical fitting, calibration, and scoring |
| `src/reflex/core/policy.py` | Policy decisions, bounded tools, and local investigation |
| `src/reflex/experiments/transaction_features.py` | Transaction-specific event and window features |
| `src/reflex/experiments/synthetic_data.py` | Synthetic stream generation and separate evaluator labels |
| `src/reflex/experiments/evaluation.py` | Synthetic metrics and benchmark protocol records |
| `src/reflex/experiments/news_adapters.py` | Example adapter for normalized news records |
| `src/reflex/experiments/news_demo.py` | Synthetic portfolio-news Reaction example |
| `src/reflex/experiments/portfolio_benchmark.py` | Portfolio-news A/B/C protocol |
| `src/reflex/runtime.py` | Reaction execution and controlled action publishing |
| `src/reflex/contracts/` | Reaction graph contracts and schema registry types |
| `src/reflex/experiments/` | Seeded scenarios, adapters, metrics, and held-out evaluation |
| `tests/` | Event, storage, runtime, workflow, and benchmark checks |

Read the [REFLEX guide](docs/REFLEX-GUIDE.md), [core implementation guide](docs/core-guide.md), [architecture](docs/architecture.md), [benchmark protocol](docs/benchmark-protocol.md), and [experiment notes](docs/REFLEX-EXPERIMENTS.md). A simple visual overview is in [the REFLEX explainer](docs/reflex-explainer.html).

## Data and limits

Dataset adapters are available for BPI Challenge 2017, Loghub, GDELT, FNSPID-style financial news, Reuters-21578, and normalized market prices. They live in the experiment layer. Public data downloads are opt-in. News-to-market relationships are predictive diagnostics; they do not establish causation.

REFLEX is a research prototype. Synthetic results are checks of the implementation and its known test cases. Use human review and independent labels before relying on discoveries or actions in operational settings.
