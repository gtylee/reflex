# Public data selection for REFLEX experiments

Checked 24 November 2026. The current code provides row-to-event adapters, an explicit one-day GDELT archive downloader, and a public event replay command. It does not resolve licenses or run public-data relevance or market-outcome scoring. Keep raw source files, extraction dates, checksums, data-vintage metadata, and terms with every future benchmark.

## Recommended use by question

| Dataset | Best fit | What it does not establish | REFLEX adapter status |
|---|---|---|---|
| GDELT Events 2.0 | Broad chronological event replay, entity/geography/event-code lenses, scalable ingestion tests. The official project provides raw event and graph downloads, BigQuery tables, and documented codebooks. | An event-code record is not a full article. It does not by itself map a China export announcement to rare-earth or portfolio holdings, and it contains no matched price outcome label. | `download_gdelt_day` and `adapt_gdelt_rows` support a date-pinned daily archive; 10,000 events from 2026-11-23 were normalized and replayed in order in about 6.0 seconds. No GKG/article text adapter, topic-to-holding linker, relevance labels, or market outcome scoring has been run. |
| FNSPID | Financial news topic to equity-price prediction, when a matched ticker universe and dates are useful. The paper describes 29.7M stock prices and 15.7M time-aligned news records for 4,775 S&P 500 companies across 1999–2023; the author repository points to a Hugging Face dataset. | Its coverage is equities and selected news sources, not a broad commodity/rates/FX portfolio. Alignment does not prove causation. Point-in-time correctness and survivorship must be checked before claims. | `adapt_fnspid_rows` maps dated ticker/news rows. No dataset was downloaded. The Hugging Face viewer returned a schema generation error during this check, although files and a download path were present; pin and validate a specific export before use. |
| Reuters-21578 | Legacy news-topic classification and parser compatibility. The source host offers a small archive and UCI mirror; the corpus has 21,578 dated 1987 news documents. | It is not paired with market prices and is too old/narrow to evaluate modern news-to-market impact or evolving digital stories. Its text distribution is not current financial news. | `parse_reuters21578` extracts basic SGML date/title/body records. The original collection terms state research-purpose distribution; review current source terms before redistribution or commercial use. |
| CRSP | High-quality daily/monthly U.S. equity outcome tests, including active and inactive securities and corporate actions, for a licensed research environment. | It is not public free data; access and redistribution are controlled by subscription terms. It does not supply the news stream. | `adapt_market_price_rows` accepts normalized price rows; CRSP field mappings and adjusted-return construction remain to be implemented by the data owner. |
| FRED / ALFRED | Public macro, rates, and selected FX/economic series; ALFRED vintages are useful for point-in-time release-aware backtests. FRED API access requires a registered API key. | It is not a full security-level market feed and series availability/frequency varies. A time series observation alone is not a causal event outcome. | `adapt_market_price_rows` can map normalized daily price/value rows. Add series ID, units, release time, vintage, and API response metadata before production of a research dataset. |

## Suggested benchmark sequence

1. **News workflow and scale:** use a pinned, filtered GDELT export for a specific period and geography. Start with at least 10,000 chronological rows, retain source event IDs and extraction hash, and measure duplicate/story matching separately from portfolio relevance. Use a separate curated annotation set for event relevance; event codes are not the answer key.
2. **News-to-market prediction:** use FNSPID as the first candidate because it already pairs news and equities over time. Verify the specific downloadable files, text completeness, timestamps/time zones, ticker mapping, price adjustment, and license. Train chronologically, reserve a later period, and score only after the event was observable. Compare with a price-only baseline and simple topic baseline before agentic methods.
3. **Topic classification diagnostics:** use Reuters-21578 for a reproducible legacy parser/classification task, not as evidence of contemporary market relevance.
4. **Rates and FX outcomes:** use selected FRED series where available; use ALFRED vintages or another point-in-time source for releases that are revised. For security returns with inactive names and corporate actions, use CRSP if the organization has an appropriate license.

## Evaluation controls

- Record both publication time and ingestion/availability time when the source exposes them. A market return begins only after the event could have been observed by the system.
- Use chronological splits, and fit topic dictionaries, thresholds, entity maps, and models only on the training period. Keep a later validation period for curation and a still-later test period for final evaluation.
- Compute returns from a documented price field and adjustment convention. Account for trading calendars, time zones, non-trading hours, corporate actions, FX conversion, and stale marks.
- Measure relevance quality, false-positive notifications, duplicate suppression, latency, model/tool calls, tokens, and reproducibility separately. A notification count is not a quality metric without human or curated relevance labels.
- Treat topic/return results as predictive associations. Causal claims require an explicit identification design and controls for confounding, selection, and event overlap.
- Preserve terms and attribution for source records and linked articles. Public availability of metadata does not automatically grant article-text redistribution rights.

## Sources

- [GDELT official data access and scale](https://www.gdeltproject.org/data.html)
- [FNSPID author repository and download paths](https://github.com/Zdong104/FNSPID_Financial_News_Dataset), [FNSPID paper](https://arxiv.org/abs/2402.06698), and [current Hugging Face dataset card](https://huggingface.co/datasets/Zihan1004/FNSPID)
- [David D. Lewis Reuters-21578 collection page](https://www.daviddlewis.com/resources/testcollections/reuters21578/) and [Reuters-21578 Distribution 1.0 README](https://github.com/cequencer/reuters)
- [CRSP US Stock Databases](https://indexes.morningstar.com/research-data-products/crsp-us-stock-databases)
- [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/) and [series observation endpoint](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
