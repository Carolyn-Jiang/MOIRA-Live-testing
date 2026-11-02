# Public projection and explicit legacy migration

`live_testing.projection.build_feed` is pure and constructs the public
`moira.live-testing.website/1` whitelist documented in `FEED.md`:

```python
feed = build_feed(
    accounting_book,
    calendar=DEFAULT_CALENDAR,
    budget_ledger=private_api_budget_or_none,
    release_manifests=verified_published_releases,
    asof=actual_now,
)
```

Only the six-model common settled prefix is public. A model that has already
completed later returns cannot pull ahead on the displayed comparison while
another model remains pending. Each completed release appears in exactly one
NY ISO week, determined by **exit time**, not action release time. Friday
actions that exit on Monday therefore belong to the new week. Each weekly
path starts at equity 1 and recomputes all metrics from only that week's
returns. The sum of weekly settlement counts equals the all-time count.

The projector accepts release manifests as a list or mapping. Each confirmed
entry must contain `timestamp` (or `release_at`), a true `published_at`, and
preferably explicit `coverage` (`complete`, `partial`, `unavailable`). Missing
coverage is unavailable. Future publications are excluded; collection-only
files must not be passed as confirmed manifests. URLs, prices, news, source
IDs, actions, pairs, prompts and private traces are never copied into the feed.

Token usage is attributed by **actual provider request start time** in New
York, independently of the historical market time an authorized replay used.
The new budget journal supplies `model_id`, `stage`, true `started_at`, status,
and actual measured input/cached/output counts. Cached tokens are a subset of
input, never added a second time. Unknown provider usage remains `null`; an
available empty project journal can establish measured zero, whereas an absent
journal cannot. An unresolved or failed request is not assumed free. Training
and inference are distinguished only where the source stage is known.

Imported measurements default to `book["imported_usage"]`; callers can supply
an explicitly audited replacement list using `imported_usage=`. Source call
identities deduplicate the same evidence if it is also present in the new
budget journal. Shared data/news processing is shown through each week's
optional `shared_tokens` / `shared_tokens_coverage`, not a seventh model row.
Monthly dollar spend/reservation fields are projected only from the new
project's budget journal; importing old usage does not silently charge or
reserve it again. If the operator chooses to count already-paid imported calls
toward this project's dollar limit, that must be an explicit audited budget
journal migration.

## Copying the two existing November 2 actions

```python
report = import_legacy_actions(
    accounting_book,
    explicit_legacy_phase3_directory,
    calendar=DEFAULT_CALENDAR,
    limit=2,
)
```

The explicit directory is required; there is no implicit path to another
project. The importer reads only the legacy state, first two cached release
JSON files, their hash-verified published Parquet bytes, and available native
runtime/provider usage evidence. It does not load/copy Qdrant or any native
checkpoint, call an API, write an alias into the old repository, or modify the
old ledger. Remote folder URLs such as `timestamp/price.parquet` map read-only
to the existing flat `timestamp_price.parquet` local file if the bare basename
does not exist. Published hashes must still match exactly.

Codex, FinMem and Moira each receive two genuine actions with the original
generation instants and `mode="live"`. Price valuations retain original
publication time as conservative measured availability. Their audit explicitly
says `legacy_input_summary_protocol`: these decisions used the original
ticker-level summaries, not newly generated per-article summaries. Original
sealed records, source observation/manifest hashes and file proofs remain
private in the new book. The three Moira 2.0 action gaps stay pending until the
separately authorized point-in-time replay fills them.

For native usage, completed accepted-attempt windows are bounded by the next
actual native attempt (or the import snapshot). Matching phase measurements
include embeddings and model calls, preserving unknown fields as unavailable.
Uninstrumented Codex calls receive explicit unknown usage markers. No token
counts are fabricated from prompt lengths or model defaults. Original
participant/competition records are never migrated.

The complete import is staged in a copy of the new accounting book, then all
read source hashes are checked again before committing the in-memory change.
An advancing source or mismatch aborts without partially changing the new
book. The returned private report is also stored as `book["legacy_import"]`;
repeating the same completed import is idempotent and does not replace its
original source proof. The root runtime owns atomic new-state persistence and
all publication operations.
