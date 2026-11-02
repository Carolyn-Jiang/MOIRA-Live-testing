# Scheduled maintenance contract

`live_testing.maintenance.run_maintenance(runtime, event)` is invoked by the
dispatcher while it holds the new experiment's `run.lock`. Importing the module
does no collection, credential read, model call or file write. Only actual
calendar events within the initial dispatch grace can start; it never fabricates
the past November 2 08:00 training event or runs maintenance after the final
decision. The first future scheduled weekly training is November 9.

## Raw cache

Sunday 09:30 and the explicit Saturday December 26 09:30 exception collect from
the last genuinely published release to the cache boundary. The result is saved
under the private `news/archives/` directory with the valid article-collection
schema. No summarization, model Gateway, action, valuation, trade or website
publication occurs in this job. A missing last published observation stays
pending rather than inventing a news window. Yahoo/network collection happens
only when the future job is explicitly dispatched; all unit tests inject it.

## Monday 08:00 training

Only published observations from the previous New York calendar week are used.
Canonical next-node entry/following-node exit labels come exclusively from the
book's immutable common valuations. Exit and both quote availability timestamps
must be no later than 08:00; unresolved outcomes, normally the final two Friday
actions, remain pending and cannot be graded as zero. No observations or no
known labels yields a pending job, not a successful training claim.

Original Moira and the three independent Top-K settings train in four isolated
directories with up to four concurrent workers. Every version runs its real
native-component trainer for at most eight cycles with sample-Sharpe selection.
All four must succeed before learned prompts and pair memory are installed.
Live active pairs, decisions, attempts, history, feedback, the live memory bank
and pending direction records are kept intact. Before/after source checks,
private backups and rollback protect a failed multi-file installation. This
routine is future weekly training; copying historical seed units is not counted
as initial training for a new setting.

Only after training completes are dated company-operation summaries generated
from cached previous-week per-article evidence for all models. Their source IDs,
actual generation times, partial-coverage limitations and prior context backup
are retained privately. Completed per-company responses are reused; an ambiguous
paid attempt is not silently repeated. The shared monthly budget applies to all
calls through scoped Gateways.

The dispatcher currently keeps `run.lock` during a long training job. It starts
at 08:00 with four workers to provide a 90-minute interval before the first
market release. If training exceeds that interval, the runtime needs a separate
data-first handoff; this module does not pause or rewrite the market schedule.

## Friday 16:00 FinMem pair reselection

This is a model-only task, including after an eligible half-day; no 16:00 market
price or artificial 13:00 trade node is collected. It uses the last published
observation, known current-week finalized performance, finalized feedback and
dated operating context. The existing native candidate selector is retained,
with explicit instructions about weekly realized returns and pending outcomes.

The chosen pair starts a fresh future epoch cloned from the already-trained
five-candidate warmup bank. No warmup retraining is performed. The old epoch,
sealed decisions and pair choices remain intact. Each pending action continues
to route feedback to its original `audit.native_generation`, even after the
active epoch pointer changes. New actions alone use the new epoch. Selection,
epoch readiness and pointer updates are journaled for retry safety.
