# Calendar and accounting integration

These modules are pure standard-library Python. They do not read credentials,
call models, collect prices, persist files, or create scheduled tasks. The
runtime must hold its own process/file lock and save successful mutations
atomically. All supplied timestamps must be timezone-aware.

## Calendar contract

`live_testing.calendar.DEFAULT_CALENDAR` contains 830 decision releases from
2026-11-02 09:30 through 2027-05-21 15:30 in `America/New_York`, plus exactly
two final valuation-only nodes: 2027-05-24 09:30 and 10:30. The final decision
enters at the former and exits at the latter. No new action is requested on
May 24. Each action uses the next actual node for entry and the following
actual node for exit; the midday gap, nights, weekends and holidays are real
gaps rather than invented hourly data.

Full holidays in the frozen experiment: November 26, December 25, January 1,
January 18, February 15, and March 26. November 27 and December 24 have only
09:30, 10:30, 11:30 and 12:30 nodes. No 13:00 trade is added on a half-day.

- `release_nodes`: immutable tuple of decision-release datetimes.
- `valuation_nodes`: release nodes plus the two terminal boundaries.
- `action_window(release)`: `ActionWindow(step, release_at, entry_at, exit_at)`;
  `.as_dict()` gives JSON timestamps. Terminal-only nodes cannot make actions.
- `release_schedule()`: all 830 action windows, numbered 1–830.
- `events()`: release, terminal valuation, Monday 08:00 training, Friday 16:00
  pair reselection, Sunday 09:30 cache and the December 26 09:30 cache exception.
  Training/reselection skip full holidays. The Friday model task may run on a
  half-day, but it never becomes a market node. Maintenance after the final
  decision is omitted. November 2 08:00 is before the experiment start and is
  not backfilled: the first future training event is November 9.
- `due_events(actual_now, completed_ids=..., grace=timedelta(minutes=25))`:
  returns reached, uncompleted events only. It performs no waiting or work.
  Persist event completion only after all required publication checks pass;
  use the event's stable `id` for idempotence. Expired dispatches are not replayed.
- `utc_segments()` groups actual release dates by both NY and UK UTC offsets,
  retaining each day's exact slots. This exposes the March 15–25 interval when
  US DST has changed but UK DST has not. Do not apply one fixed UTC hour across
  the experiment. `ScheduledEvent.as_dict()` supplies NY, UTC and London times.

The half-day/holiday list is explicitly scoped to this experiment. A new
experiment requires a newly verified exchange calendar.

## Independent accounts and causal mutations

```python
from live_testing.accounting import (
    new_book, record_decision, record_valuation, settle_due,
    matured_feedback, account_metrics,
)

book = new_book(six_model_ids, ten_eligible_symbols)
record_decision(book, model_id, release_at, action,
                generated_at=actual_response_time, input_asof=release_at,
                audit={"response_id": response_id, "input_sha256": input_sha256})
record_valuation(book, boundary, complete_prices,
                 observed_at=actual_collection_time, provenance=price_provenance)
report = settle_due(book, actual_now)
feedback = matured_feedback(book, model_id, actual_now,
                            observation_asof=current_release_at)
metrics = account_metrics(book, model_id)
```

Production supplies six unique model IDs. Each starts at equity 1 with empty
history, independently of the earlier competition. The utility is generic
enough for smaller synthetic fixtures, and never refers to old ledger paths.

`action` has exactly `stock1`, `stock2`, `action`; stocks must be distinct and
eligible even for a genuine `F`. `LS` is long the first stock and short the
second, `SL` reverses the legs, each weighted 0.5. A genuine recorded `F` gives
zero return. A missing action or incomplete quote stays **pending**, with no
invented zero return, and blocks only that model's cumulative prefix. Later
available rounds cannot jump over its unresolved earlier round.

Live actions require `release_at <= actual generated_at < entry_at`, with
`input_asof` equal to the release boundary. Existing genuine actions may be
copied with their original true generation times and source audit. The
explicitly authorized initial point-in-time model replays use
`mode="retrospective_point_in_time"` and must include nonempty
`audit.authorization` and a 64-character `audit.input_sha256`. Their real new
generation timestamps are retained; historical sealing is never fabricated.
This mode authorizes historical input replay, not future observations.

Only complete, positive, finite exact-symbol price snapshots are committed.
Committed price values cannot be overwritten; an identical retry keeps the
original availability/provenance. Keep incomplete collection attempts outside
this immutable valuation map. The runtime is responsible for verifying an
opening-minute open at 09:30 or the preceding completed minute's close at other
boundaries; the accounting module values only the supplied canonical boundary.

Settlement requires exit to have happened and both quotes and the decision to
have actually become available. Feedback is returned only after settlement and
exit, and `observation_asof` additionally prevents feedback from a later market
boundary from entering an earlier point-in-time decision. Use `actual_now` for
availability and `current_release_at` for the observation cutoff; legitimate
seconds of collection latency do not turn into future market information.

Every history row is independently validated against its immutable decision and
entry/exit snapshots. `settle_due` is atomic at the dictionary level and
idempotent; the caller remains responsible for disk/remote atomicity. A return
that makes equity nonpositive raises an explicit insolvency-policy error; it is
never clipped or replaced by Flat.

## Metrics and usage intervals

`risk_metrics(step_returns)` and `account_metrics(book, model_id)` return:
`completed_rounds`, `cumulative_return`, `sharpe_ratio`, and `max_drawdown`.
Equity starts at 1, returns compound, Sharpe uses the sample standard deviation
(`ddof=1`), zero risk-free return and no annualization. Undefined/constant-return
Sharpe is `None`. Maximum drawdown is a positive fraction, includes initial
equity 1, and uses the observed settlement endpoints only. No missing outcome
is silently added to the metric denominator.

`record_usage(book, model_id_or_None, call_id=..., at=actual_time,
cost_usd=..., input_tokens=..., output_tokens=..., purpose=...,
provider_model=...)` appends immutable actual usage, with `None` for shared
data preparation. Repeated identical IDs are idempotent; changed usage for an
existing ID is rejected. `weekly_usage` uses NY ISO weeks from Monday 00:00;
`monthly_usage` uses NY natural months. Both use half-open intervals, preserve
DST transitions, and report per-model plus shared totals. The separate budget
service must reserve expected spend before calls and enforce the user-approved
$20 warning / $100 monthly hard cap; these reporting functions never authorize
a model call or increase a budget.
