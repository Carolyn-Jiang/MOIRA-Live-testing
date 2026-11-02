# Initial Moira 2.0 training

The explicitly selected initialization archive is the 29 actual releases from
19–23 October 2026, plus their immutable delayed-execution prices. It is a
separate historical contract, not a change to the subsequent Monday 08:00
previous-week rule. The final historical action uses the actual 15:30 entry and
16:00 exit; neither valuation becomes an invented thirtieth observation.

Each Top-1, Top-3 and Top-5 version starts from its own copy of the common native
Moira checkpoint and runs actual native selection, direction, monitoring and
reflection calls. Its RRI bank starts empty in each training episode. Finalized
actual and three counterfactual outcomes enter one whole memory unit only after
their historical exit is available. Retrieval uses that observation's source
cutoff. Current company context and November observations are excluded.

The common parent already learned from this same historical period. All reported
returns, sample Sharpe and maximum drawdown therefore describe **in-sample
training evaluations**, not a held-out result. The optimization accepts only
strict improvement in defined, nonannualized sample Sharpe and records rejection
and rollback. Each invocation permits at most eight native cycles, with the
native early-stop structure. Generation timestamps and API usage are real.

`python -m live_testing.initial_training --root PRIVATE_ROOT --output STAGING_ROOT`
validates sources without calling a model. Adding `--execute` explicitly runs
the three independent versions through the shared monthly budget. The output
contains frozen inputs, parent lineage, native episode books, model-call audits,
the final trained checkpoint and each version's own retained episode CF units.
It does not install a live checkpoint, publish a website or generate live actions.
