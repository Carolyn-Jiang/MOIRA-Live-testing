# Public website feed

GitHub Pages serves `docs/` with no build step. The page reads only the same-origin `state.json`; it never authenticates to the private Hugging Face dataset. The initial checked-in feed is a pending state with no simulated returns.

`schema` is `moira.live-testing.website/1`; `timezone` is `America/New_York`. `status` is `scheduled`, `live`, `pending` or `finished`. `updated_at` is a timezone-aware ISO timestamp, or `null` before first publication.

The fixed `period` contains `start` (`2026-11-02T09:30:00-05:00`), `last_release` (`2027-05-21T15:30:00-04:00`) and `final_settlement` (`2027-05-24T10:30:00-04:00`). `next_release` is `null` or `{timestamp, session_label, note}` from the verified market calendar. All display times use New York, including daylight saving time.

Model IDs, in stable tie-breaking order:

| ID | Display name | Backbone |
| --- | --- | --- |
| `codex` | Codex | gpt-5.3-codex |
| `finmem` | FinMem | gpt-4.1-mini |
| `moira` | Moira | gpt-4.1-mini |
| `moira2-top1` | Moira 2.0 · Top K = 1 | gpt-4.1-mini |
| `moira2-top3` | Moira 2.0 · Top K = 3 | gpt-4.1-mini |
| `moira2-top5` | Moira 2.0 · Top K = 5 | gpt-4.1-mini |

`all_time` contains exactly `completed_rounds`, `rows`, `series`. `weeks` is an array of objects with `week_id` (such as `2026-W45`), `label`, timezone-aware `start` and `end`, the same three view fields, and `tokens`. Optional `tokens_coverage` is `complete`, `partial` or `unavailable`. Weekly completed counts must sum to the all-time count.

Each metric row contains only `{model_id, cumulative_return, sharpe_ratio, max_drawdown}`. Returns and drawdowns are decimals. Sharpe is nonannualized, uses zero risk-free rate and sample standard deviation; undefined Sharpe is `null`. Drawdown is nonnegative and includes initial equity 1.

Each series contains only `{model_id, points}`; each point is exactly `{step, equity}`. Step 0 is equity 1, followed by every consecutive completed settlement. Weekly series are independently rebased to 1. Counts above zero require all six distinct models, the same number of settlement points and metrics consistent with those paths. Counts of zero require empty `rows` and `series`; pending models are never inferred to be flat.

Each week's six token rows contain exactly `model_id`, `input_tokens`, `cached_input_tokens`, `output_tokens`, `total_tokens`, `training_tokens`, `inference_tokens`. Counts are nonnegative safe integers or `null` when unavailable. Input includes cached input; total is input plus output. Training plus inference equals total when all three are known. Imported usage with no source measurements stays unavailable, rather than becoming zero.

A model token row may additionally contain `recorded_tokens` and `token_coverage`, always together. Both objects have exactly the six token field names above. Recorded counts are nonnegative safe integers or `null`; each field's coverage is `complete`, `partial` or `unavailable`. A complete field has a non-null original count and the same recorded count. A partial field keeps its original full total `null` and supplies a numeric recorded subtotal; the page shows `≥N` with an explanation that some provider usage is unavailable. An unavailable field has both counts `null`. A recorded zero may therefore appear as `≥0`, which does not claim complete zero usage. Older feeds without this metadata remain valid. Full-total arithmetic checks apply to the original counters; partial fields may cover different request subsets and need not add up. Shared processing retains its existing counter schema.

Each token row may also contain `estimated_cost_usd` and `cost_coverage` together. A cost is a nonnegative USD estimate or `null`; coverage is `complete`, `partial` or `unavailable`. Estimates aggregate the selected week's model requests, including training, inference and embeddings. Verified fees take priority; unanswered requests retain their conservative reservation. Unpriced historical requests are excluded and mark the subtotal partial. Shared news processing is excluded from each model's cost. The page displays Estimated Cost in place of the Training column; training counters remain internal to token accounting. Budget controls remain active without a public warning banner.

Optional per-week `shared_tokens` contains the same six token counters, without a model ID. Optional `shared_tokens_coverage` is `complete`, `partial` or `unavailable`. This reports common news-processing costs separately; it is never a seventh ranked model or silently charged once to every model.

Optional `data_status` contains exactly `{published_batches, expected_release_points, last_release_at, coverage}`. Expected count and latest timestamp can be `null`; coverage is `pending`, `complete`, `partial` or `unavailable`.

Optional `budget` contains exactly `{month, spent_usd, reserved_usd, warning_usd, limit_usd, status}`. Month is `YYYY-MM` in New York; verified spend/reservations are nonnegative dollar numbers or `null`; warning is 20 and hard limit 100. Status is `ok`, `warning`, `blocked` or `unavailable`.

The validator rejects extra fields at every level. No credentials, private news or prices, stock selections, actions, participant records, model prompts, source response IDs or training traces belong in this feed. Publish the aggregate JSON and frontend together when changing the schema. A failed refresh retains the last verified display; it never invents results.
