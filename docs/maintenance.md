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

## Weekly 08:00 belief review and training

The normal slot is Monday 08:00 New York. When Monday is a full market holiday,
the slot moves to Tuesday 08:00 of that same week, provided Tuesday is a trading
day. A normal Tuesday does not trigger a second job. In this experiment the
holiday fallbacks are January 19 and February 16, 2027. The source window always
remains the preceding New York Monday through Sunday; the actual Monday or
Tuesday 08:00 determines which outcomes and sources are available.

Only published observations from the previous New York calendar week are used.
Canonical next-node entry/following-node exit labels come exclusively from the
book's immutable common valuations. Exit and both quote availability timestamps
must be no later than 08:00; unresolved outcomes, normally the final two Friday
actions, remain pending and cannot be graded as zero. No observations or no
known labels yields a pending job, not a successful training claim.

Original Moira and the three independent Top-K settings train in four isolated
directories with up to four concurrent workers. Each version first reviews its
genuinely accepted prior-week actions, including explicitly authorized preparation
replays with their original mode and generation times preserved. Only finalized
losses or counterfactual regret can become failed cases. Evidence links the
existing active learned belief, actual decision rationale and dated prompt
provenance; a loss alone does not establish that a belief caused it.

A budgeted reviewer may retain, revise or delete an existing belief, citing real
failed-case evidence. Transferable revisions pass the native sanitizer. Before a
revision becomes the new training seed, separate native episodes compare the old
and proposed beliefs on the same causal observations and immutable delayed
prices. Acceptance requires a strict finite sample-Sharpe improvement and a
changed decision that improves a cited failed case. These are historical replay
checks, not proof of causal attribution or out-of-sample improvement. Rejected
proposals retain the original beliefs. Deleted active beliefs cannot silently
reactivate older hidden lessons. Private audit records retain the original
checkpoint, changes, evidence and validation metrics. Incomplete or ambiguous
review blocks the affected training instead of silently bypassing the review.

After its review completes, every version runs its real
native-component trainer for at most eight cycles with sample-Sharpe selection.
All four must succeed before learned prompts and pair memory are installed.
Live active pairs, decisions, attempts, history, feedback, the live memory bank
and pending direction records are kept intact. Before/after source checks,
private backups and rollback protect a failed multi-file installation. This
routine is future weekly training; copying historical seed units is not counted
as initial training for a new setting.

After the quarterly ranking migration, each Top-K version's readiness marker
selects the schema-2 native trainer and belief-validation episodes. The original
Moira keeps its separate native path. Weekly prompt installation also updates
the corresponding immutable training-result hash and checkpoint marker, while
preserving the live bank's individual importance values, feedback counts and
unclosed actions. A missing or inconsistent schema-2 marker holds that model
pending; it never falls back to schema 1.

Schema-2 action feedback uses the actual cited memory IDs and exact original
decision context for one-memory-removed comparisons. Each matured feedback is
applied once, together with the new finalized reflection and a private evidence
record. A failed or ambiguous comparison cannot be replaced with invented Flat
or a blanket update to all retrieved memories. Only after a complete feedback
transaction can a subsequent decision use its updated importance values.

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
