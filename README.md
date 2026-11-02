# MOIRA Live Testing

An independent, local experiment comparing six financial trading-agent variants. GitHub Pages displays verified **aggregate returns, risk metrics and weekly token usage**. A private Hugging Face Dataset stores released price/news files. No participant submission service or real broker integration is included.

- Dashboard: https://carolyn-jiang.github.io/MOIRA-Live-testing/
- Private data: https://huggingface.co/datasets/CarolynJiang/MOIRA-Live-testing
- Public feed contract: [docs/FEED.md](docs/FEED.md)
- Calendar and accounting: [docs/calendar-accounting.md](docs/calendar-accounting.md)
- Maintenance: [docs/maintenance.md](docs/maintenance.md)

The initial public feed deliberately contains no performance observations. “Pending” and “Unavailable” are not zero returns or measured zero tokens. A deployment or successful unit test does not establish that live collection, initialization or training has completed.

## Models

| Model ID | Backbone | Decision-memory mechanism |
| --- | --- | --- |
| `codex` | `gpt-5.3-codex` | Released observation and dated company context |
| `finmem` | `gpt-4.1-mini` | Native fixed-pair FinMem agent and layered memory |
| `moira` | `gpt-4.1-mini` | Native Moira policy components and chronological recent memory |
| `moira2-top1` | `gpt-4.1-mini` | Independent whole-reflection retrieval, top K = 1 |
| `moira2-top3` | `gpt-4.1-mini` | Independent whole-reflection retrieval, top K = 3 |
| `moira2-top5` | `gpt-4.1-mini` | Independent whole-reflection retrieval, top K = 5 |

The ten eligible tickers are NVDA, GOOG, META, MSFT, JPM, AMZN, AAPL, AMD, WMT and TSLA. Two distinct stocks and an explicit LS, SL or F action are required internally. Missing decisions, missing quotes and failed requests remain pending; they are never silently replaced with Flat.

### Moira 2.0 memory

One immutable memory unit represents one exited action, its actual and LS/SL/F counterfactual returns, a whole reflection and source/availability provenance. Each model version has an independent bank; training and live namespaces are distinct. Incomplete or future outcomes cannot enter retrieval.

The retrieval score is:

```text
cosine(query, memory) + min(raw_importance, 100) / 100
                     + exp(-elapsed_exchange_sessions / 252)
```

Exchange sessions, rather than calendar days, determine recency. Initial importance is 50. Feedback adds 18 after a positive realized return, subtracts 18 after a negative return, and leaves it unchanged at zero. Raw importance is retained without a lower clamp or hidden decay; only its scoring normalization is capped above at 100. Feedback applies only to the memories actually retrieved for that exited decision, once its outcome is available. See [live_testing/memory.py](live_testing/memory.py).

## Local setup

Use Python 3.11 and Node.js 20 or newer. There are no frontend package dependencies or build step.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The native Moira/FinMem implementation is a **separate Git checkout**. Its actual source repository and exact revision are recorded in [native-framework.lock.json](native-framework.lock.json). Do not substitute a different framework version merely because its API looks similar.

```sh
git clone https://github.com/Carolyn-Jiang/pair-trading-fix.git ../pair-trading-fix
git -C ../pair-trading-fix checkout --detach c856a3cf5102284d8e84cffdc72d527356b1e9be
export PAIR_TRADING_FIX_ROOT="$(cd ../pair-trading-fix && pwd)"
.venv/bin/python scripts/verify_native.py --framework-dir "$PAIR_TRADING_FIX_ROOT"
```

The verifier makes no network or model request. Native imports require the exact clean commit and matching hashes for 44 source/configuration files. Native code is not copied into the public data feed. The framework repository's license and notices continue to apply.

Store the runtime directory and credentials **outside this repository**, for example in an adjacent private directory with owner-only permissions. Supported bootstrap rejects an in-repository runtime directory. `config.json` in that private directory records absolute local paths to the framework, retained seed material and key file. Configure `OPENAI_API_KEY` only in the local private key file; authenticate GitHub/Hugging Face locally. Never put API credentials in `docs/`, Git history, a browser request, a dataset file or the chat.

Native live calls also require the organizer's local action adapter. The private configuration must supply `finmem_action_adapter` with its absolute `path` and verified `sha256`. The adapter must remain inside the private runtime and outside this public repository. Missing or mismatched files fail closed; its implementation and operational settings are not part of the public dashboard.

The first real call to the tokenizer may need its `cl100k_base` vocabulary. Supply an already populated `TIKTOKEN_CACHE_DIR` in offline environments. Tests inject counting stubs where appropriate; production summary validation uses the actual `cl100k_base` encoder.

## Execution and causality

All schedule values are in `America/New_York`, including daylight-saving changes. The frozen evaluation begins **2 November 2026, 09:30**, has its last data release **21 May 2027, 15:30**, and ends with valuation-only boundaries **24 May, 09:30 and 10:30**. The final action enters at 09:30 and exits at 10:30 on 24 May.

Regular-session release nodes are 09:30, 10:30, 11:30, 12:30, 14:30 and 15:30. An observation enters at the next actual node and exits at the following actual node. For example, 09:30 data determines a 10:30 entry and 11:30 exit. Holidays and half-days come from the frozen calendar; a half-day does not invent a 13:00 trading node. The live dispatcher accepts only a reached node within its 25-minute dispatch window.

Local scheduled dispatch, once explicit private bootstrap and required initialization have completed:

```sh
.venv/bin/python -m live_testing.cli --state-dir /absolute/path/to/private-runtime --due --wait-until-due
```

This command can make paid model calls and publish authorized data/results. Importing modules and running the offline tests do neither. Scheduling is managed separately; this repository does not install or enable an operating-system scheduler automatically.

### Initial migration and training

The first two explicitly authorized observations on 2 November may use retrospective point-in-time replay. Imported actions retain their genuine generation times, and every imported Moira pair transition must satisfy this experiment's lifecycle schedule: a BREAK remains pending until an actual 15:30 New York data release. An incompatible legacy transition is rejected before bootstrap copies or commits its state. The user's additional authorization covers only 2 November training-preparation gaps up to the recorded training-completion cutoff, and only where the observation was already frozen and published. It does not authorize arbitrary historical replay, future observations or recrawling a past window to replace its inputs. Moira 2.0 actions are generated after version-specific training from those corresponding frozen observations, with their **real generation times, explicit authorization and retrospective mode** retained privately. They must not be presented as contemporaneously generated live decisions or as out-of-sample results.

An explicitly authorized correction of an incompatible imported Moira transition runs in a separate shadow state. It preserves the valid decision prefix, archives the superseded records, reuses the exact published observations and immutable valuations, and exposes feedback only after its original exit boundary. Corrected decisions retain their actual new generation times and retrospective provenance. The corrected account and native state are installed only after validation; other models and the separate Task 2 competition remain unchanged. Ordinary dispatch does not grant or perform this repair automatically.

The original Moira's inherited CR-trained checkpoint requires a separate, staged Sharpe retraining before this correction. `live_testing.retrain_original_sharpe` starts from the retained default prompt with empty learned beliefs and pair memory, uses only the same frozen historical 29-release dataset and its delayed execution labels, and accepts strict sample-Sharpe improvements for at most eight cycles. It never installs a model or rewrites live results itself. Replacing a checkpoint requires regenerating the entire affected model's decision sequence, rather than retaining decisions made with the previous checkpoint. Training metrics are in-sample and never enter the live account.

Initial Moira 2.0 training uses the authorized cached **19–23 October** observations and exact delayed-entry/exit labels. Each top-K version requires its own completed training checkpoint and readiness marker before inference. Constructing a reflection bank alone is not evidence of completed version-specific training. Private manifests are the source of truth for whether actual training has finished; no result is asserted here.

Subsequent weekly jobs run Monday 08:00 New York, or Tuesday 08:00 when Monday is closed. They use only the preceding New York calendar week's observations and outcomes already known at the actual cutoff. Before retraining, each Moira version reviews finalized failed cases against its existing learned beliefs; proposed corrections or deletions require evidence and isolated counterfactual replay validation. Native baseline episodes, reflection, proposal and accept/reject operations optimize sample Sharpe without annualization, at most eight cycles per training invocation. The loop accepts strict Sharpe improvement and keeps the previous checkpoint otherwise. Training/live direction memories remain separate. Weekly context refresh and Friday pair-maintenance behavior are described in [docs/maintenance.md](docs/maintenance.md).

## News and private Dataset publication

News is selected using source **publication time**, with windows including the start and excluding the end. `dateModified` is never a substitute. Live Yahoo search is capped at 200 records per ticker; Sunday and the explicit 26 December archive supplement that cache. Partial, missing, capped, truncated and conflicted coverage remain explicit.

UUID and canonical-URL identities are deduplicated across tickers. Trusted article publication metadata takes precedence over provider timestamps. A genuinely conflicting identity component is quarantined and excluded from the released news, with a private audit and per-ticker conflict count. Available article text is used; inaccessible article bodies remain honestly labeled headline-only partial evidence.

Each article receives its **own English summary of 10–50 `cl100k_base` tokens**. A shared article is summarized once and then associated with its relevant tickers. Up to 15 independent articles share one model request, capped at 80,000 UTF-8 payload bytes; up to four batches run concurrently. Every valid article is cached immediately using its content hash and summary-protocol version. Length failures receive at most two individual revisions with actual token-count feedback. Failures never produce a placeholder summary, and restarting skips verified caches.

The publisher asserts that `CarolynJiang/MOIRA-Live-testing` remains private before each commit. One compare-and-swap commit contains a new release folder's `price.parquet`, `news.parquet`, `release.json` and updated `releases.json`. An identical retry verifies the existing bytes; different content under an existing release identity is rejected. Private raw sources, checkpoints, prompts, action records and model audit logs stay outside the repository and are not uploaded by the public website publisher.

## Returns, risk and usage

The simulator uses equal half-weight long/short legs with unit gross exposure. LS returns `0.5 × (P1_exit/P1_entry − P2_exit/P2_entry)`; SL reverses the sign; an explicit F returns zero. Equity compounds from 1. Costs, borrow fees and real execution slippage are not modeled. This is a simulated test, not a brokerage or transaction service.

Sharpe is the mean completed interval return divided by sample standard deviation (`ddof=1`), zero risk-free rate, without annualization. Zero or numerically negligible volatility produces N/A. Maximum drawdown is nonnegative `max(1 − equity/running_peak)`, including initial equity 1 and settlement observations only. Weekly views independently restart from equity 1; all-time equity carries across weeks.

Provider input, cached-input, output and total token counts are measured, not guessed. Cached input is a subset of input. Training and inference classify usage by purpose. Shared news-processing tokens are reported separately instead of inventing a seventh ranked model. Imported usage with no source evidence remains unavailable, not zero.

All models and shared processing use one project-wide **New York calendar-month** budget: $20 warning and $100 hard cap. Every request reserves a conservative bounded cost before transport. Failed or usage-unknown requests keep their reservation. An attempted reservation exceeding the limit persists a block for the rest of that month, even if later actual costs reduce the reserved total. Acknowledging a warning does not add funds or unblock the month. A new New York month resets the spending window; stale pricing verification fails closed until reviewed. This is the experiment's local cost ledger, not an account-wide billing statement.

## Public/private boundary

Only `docs/state.json` is updated by the result publisher, after strict frontend validation and independent path-metric checks. It exposes model identifiers, aggregate equity paths, CR/Sharpe/MDD, calendar status, measured usage and budget aggregates. It excludes stock pairs, actions, source prices/news, prompts, credentials, private request IDs and participant data. The browser reads the same-origin feed with credentials omitted and never receives a Hugging Face token. Dataset access requires ordinary Hugging Face sign-in and repository permission.

## Offline tests and preview

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
npm test
.venv/bin/python -m http.server 8874 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8874/ for a local-only preview. Tests use synthetic records and fake transports; native integration tests use the pinned framework when available. They do not establish real market performance. GitHub Pages serves `main` → `/docs` directly. Commit only reviewed public source/frontend files; keep all operational state and secrets outside Git.
