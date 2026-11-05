"""Pure aggregate-only website projection of an independently audited book."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import math

from .accounting import validate_book, risk_metrics
from .calendar import DEFAULT_CALENDAR, HALF_DAYS, NY, aware, week_bounds, month_bounds
from .registry import MODEL_IDS
from .budget import cost as provider_cost, model_rate, usage_cost, PricingUnavailable
from .usage_classification import (FIELDS as PURPOSE_FIELDS, CATEGORIES, TRAINING_STAGES,
                                   validate_catalog)

TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
                "training_tokens", "inference_tokens")
SHARED_IDS = {None, "data", "shared", "__shared__"}
MAX_SAFE_INTEGER = 2**53 - 1


def _count(value):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ValueError("A measured nonnegative safe-integer token count is required")
    return value


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("A finite nonnegative verified cost is required")
    return float(value)


def _call_cost(row, incoming, cached, outgoing, complete):
    """A public estimate never turns an unmeasured request into a free call.

    Recorded token-based estimates take priority. A known provider/token tuple
    can be priced with the verified rate table; missing cache counts use the
    conservative uncached rate. Unanswered budget bounds are reported separately.
    """
    actual = row.get("actual_cost_usd")
    if actual is not None:
        return _finite(actual), "complete"
    provider = row.get("requested_model") or row.get("response_model") or row.get("provider_model")
    if complete and provider and (provider == 'gpt-6.1-sol' or provider.startswith('gpt-6.1-sol-20')):
        # Cache writes replace ordinary input and have distinct lifetime rates.
        # Missing write evidence must not become an invented zero-write bill.
        try:
            return usage_cost(provider, row.get('usage') or {},
                              cache_write_ttl=row.get('cache_write_ttl')), 'complete'
        except (ValueError, PricingUnavailable):
            return None, 'unavailable'
    if complete and incoming is not None and outgoing is not None and provider:
        try:
            rates = model_rate(provider)
            estimate = provider_cost(provider, incoming, cached or 0, outgoing)
            return estimate, "complete" if cached is not None or rates[0] == rates[1] else "partial"
        except PricingUnavailable:
            pass
    return None, "unavailable"


def _normalise_call(identity, row):
    model = row.get("model_id")
    if model not in MODEL_IDS and model not in SHARED_IDS:
        raise ValueError("Unknown usage model; do not silently assign it to shared processing")
    started = aware(row["started_at"])
    raw = row.get("usage") or {}
    complete = row.get("status") == "completed"
    incoming = row.get("input_tokens", raw.get("input_tokens", raw.get("prompt_tokens"))) if complete else None
    outgoing = row.get("output_tokens", raw.get("output_tokens", raw.get("completion_tokens"))) if complete else None
    cached = row.get("cached_input_tokens") if complete else None
    if complete and "cached_input_tokens" not in row:
        details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
    incoming, outgoing, cached = map(_count, (incoming, outgoing, cached))
    if incoming is not None and cached is not None and cached > incoming:
        raise ValueError("Cached tokens cannot exceed total input")
    total = incoming + outgoing if incoming is not None and outgoing is not None else None
    _count(total)
    if complete and row.get("total_tokens") is not None and total is not None and _count(row["total_tokens"]) != total:
        raise ValueError("Provider token totals disagree")
    estimated_cost, cost_coverage = _call_cost(row, incoming, cached, outgoing, complete)
    reservation = (_finite(row.get("guard_cost_usd", 0))
                   if estimated_cost is None and row.get("guard_cost_usd") is not None else 0.0)
    return {"id": row.get("source_call_id", identity), "model_id": model,
            "started_at": started, "stage": row.get("stage"),
            "input_tokens": incoming, "cached_input_tokens": cached,
            "output_tokens": outgoing, "total_tokens": total,
            "estimated_cost_usd": estimated_cost, "cost_coverage": cost_coverage,
            "reserved_cost_usd": reservation}


def _calls(book, budget_ledger, imported_usage):
    result = {}
    if budget_ledger is not None:
        if budget_ledger.get("schema") != "moira.live-testing.budget/1" or not isinstance(budget_ledger.get("calls"), dict):
            raise ValueError("A verified project budget ledger is required")
        for identity, row in budget_ledger["calls"].items():
            normal = _normalise_call(identity, row)
            if normal["id"] in result and result[normal["id"]] != normal:
                raise ValueError("Conflicting usage for one provider request")
            result[normal["id"]] = normal
    imports = book.get("imported_usage", []) if imported_usage is None else imported_usage
    for row in imports:
        normal = _normalise_call(row["source_call_id"], row)
        if normal["id"] in result and result[normal["id"]] != normal:
            raise ValueError("Imported provider usage must not be counted twice or changed")
        result[normal["id"]] = normal
    return list(result.values())


def _sum_known(values):
    values = list(values)
    if any(value is None for value in values):
        return None
    return _count(sum(values))


def _token_totals(rows, *, evidence_available):
    if not rows:
        return {field: 0 if evidence_available else None for field in TOKEN_FIELDS}
    result = {field: _sum_known(row[field] for row in rows) for field in TOKEN_FIELDS[:4]}
    stages = {"training": {"training", "warmup"},
              "inference": {"inference", "selection", "replay", "backtest", "evaluation", "data_summary"}}
    unknown_stage = any(row["stage"] not in stages["training"] | stages["inference"] for row in rows)
    for category, names in stages.items():
        result[category + "_tokens"] = None if unknown_stage else _sum_known(
            row["total_tokens"] for row in rows if row["stage"] in names)
    return result


def _recorded_tokens(rows, *, evidence_available):
    """Measured lower bounds alongside the unchanged complete-total fields.

    An unanswered call contributes no invented tokens. A recorded numeric zero
    is retained, while an entirely unmeasured field remains unavailable.
    """
    stages = {"training_tokens": {"training", "warmup"},
              "inference_tokens": {"inference", "selection", "replay", "backtest", "evaluation", "data_summary"}}
    unknown_stage = any(row["stage"] not in stages["training_tokens"] | stages["inference_tokens"] for row in rows)
    recorded, coverage = {}, {}
    for field in TOKEN_FIELDS:
        if field in stages:
            values = [row["total_tokens"] for row in rows if row["stage"] in stages[field]]
            uncertain = unknown_stage
        else:
            values = [row[field] for row in rows]
            uncertain = False
        known = [value for value in values if value is not None]
        if not values and not uncertain:
            recorded[field] = 0 if evidence_available else None
            coverage[field] = "complete" if evidence_available else "unavailable"
        elif not known:
            recorded[field], coverage[field] = None, "unavailable"
        else:
            recorded[field] = _count(sum(known))
            coverage[field] = "complete" if not uncertain and len(known) == len(values) else "partial"
    return {"recorded_tokens": recorded, "token_coverage": coverage}


def _cost_totals(rows, *, evidence_available):
    if not rows:
        return {"estimated_cost_usd": 0.0 if evidence_available else None,
                "cost_coverage": "complete" if evidence_available else "unavailable",
                "reserved_cost_usd": 0.0 if evidence_available else None}
    known = [row["estimated_cost_usd"] for row in rows if row["estimated_cost_usd"] is not None]
    if not known:
        return {"estimated_cost_usd": None, "cost_coverage": "unavailable",
                "reserved_cost_usd": math.fsum(row["reserved_cost_usd"] for row in rows)}
    return {"estimated_cost_usd": math.fsum(known),
            "cost_coverage": "complete" if all(row["cost_coverage"] == "complete" for row in rows) else "partial",
            "reserved_cost_usd": math.fsum(row["reserved_cost_usd"] for row in rows)}


def _purpose_totals(rows, *, model, catalog, start, end, evidence_available):
    """Separate accepted trading decisions without dropping any other usage."""
    if catalog is None:
        return {**dict.fromkeys(PURPOSE_FIELDS),
                "recorded": dict.fromkeys(PURPOSE_FIELDS),
                "coverage": dict.fromkeys(PURPOSE_FIELDS, "unavailable")}
    groups = {name: [] for name in CATEGORIES}
    for row in rows:
        if row["stage"] in TRAINING_STAGES:
            continue
        kind = catalog["calls"].get(row["id"], {}).get("kind", "other_inference")
        groups[kind].append(row["total_tokens"])
    groups["trading"] = groups["direction_decision"] + groups["pair_reselection"]
    unresolved = catalog["unresolved"][model]
    missing = {kind: any(start <= aware(stamp) < end for stamp in unresolved.get(kind, ()))
               for kind in ("direction_decision", "pair_reselection")}
    missing["trading"] = any(missing.values())
    result, recorded, coverage = {}, {}, {}
    for kind, values in groups.items():
        field = kind + "_tokens"
        known = [value for value in values if value is not None]
        uncertain = missing.get(kind, False) or len(known) != len(values)
        if not evidence_available:
            result[field] = recorded[field] = None; coverage[field] = "unavailable"
        elif uncertain:
            result[field] = None
            # A missing proof may contribute no known rows; zero remains only a lower bound.
            recorded[field] = sum(known)
            coverage[field] = "partial"
        else:
            result[field] = recorded[field] = _count(sum(known)); coverage[field] = "complete"
    return {**result, "recorded": recorded, "coverage": coverage}


def _coverage(rows):
    values = [row["token_coverage"][field] if "token_coverage" in row
              else "complete" if row[field] is not None else "unavailable"
              for row in rows for field in (*TOKEN_FIELDS, *(PURPOSE_FIELDS if 'trading_tokens' in row else ()))]
    return "complete" if all(value == "complete" for value in values) else "unavailable" if all(value == "unavailable" for value in values) else "partial"


def _view(histories, indices):
    indices = list(indices)
    if not indices:
        return {"completed_rounds": 0, "rows": [], "series": []}
    rows, series = [], []
    for model in MODEL_IDS:
        returns = [histories[model][i]["step_return"] for i in indices]
        metrics = risk_metrics(returns)
        rows.append({"model_id": model, **{field: metrics[field] for field in
            ("cumulative_return", "sharpe_ratio", "max_drawdown")}})
        points, equity = [{"step": 0, "equity": 1.0}], 1.0
        for step, value in enumerate(returns, 1):
            equity *= 1 + value
            points.append({"step": step, "equity": equity})
        series.append({"model_id": model, "points": points})
    rows.sort(key=lambda row: (-row["cumulative_return"], MODEL_IDS.index(row["model_id"])))
    return {"completed_rounds": len(indices), "rows": rows, "series": series}


def _public_releases(manifests, asof, calendar):
    manifests = list(manifests.values()) if isinstance(manifests, dict) else list(manifests)
    result = {}
    for record in manifests:
        at = aware(record.get("timestamp", record.get("release_at")))
        published = aware(record["published_at"])
        if at not in calendar.release_nodes or published < at:
            raise ValueError("Only an actual published release can enter the public manifest count")
        if published > asof:
            continue
        coverage = record.get("coverage", "unavailable")
        if coverage not in {"complete", "partial", "unavailable"}:
            raise ValueError("Published release coverage must be explicit")
        value = {"at": at, "coverage": coverage}
        if at in result and result[at] != value:
            raise ValueError("Conflicting release manifest")
        result[at] = value
    return result


def _budget(ledger, asof):
    month = aware(asof).strftime("%Y-%m")
    if ledger is None:
        return {"month": month, "spent_usd": None, "reserved_usd": None,
                "warning_usd": 20, "limit_usd": 100, "status": "unavailable"}
    if ledger.get("limit_usd") != 100 or ledger.get("warning_usd") != 20:
        raise ValueError("The fixed monthly budget policy changed")
    start, end = month_bounds(asof)
    rows = [row for row in ledger["calls"].values() if start <= aware(row["started_at"]) < end and aware(row["started_at"]) <= asof]
    for row in rows:
        if row.get("month") != month:
            raise ValueError("Budget month differs from the actual NY request time")
    spent = math.fsum(_finite(row["actual_cost_usd"]) for row in rows if row.get("actual_cost_usd") is not None)
    guard = math.fsum(_finite(row["guard_cost_usd"]) for row in rows)
    if guard + 1e-9 < spent:
        raise ValueError("Budget reservations cannot be below verified spend")
    return {"month": month, "spent_usd": spent, "reserved_usd": max(0.0, guard - spent),
            "warning_usd": 20, "limit_usd": 100,
            "status": "blocked" if guard >= 100 or ledger.get("blocks", {}).get(month) else "warning" if spent >= 20 else "ok"}


def build_feed(book, *, calendar=DEFAULT_CALENDAR, budget_ledger=None,
               release_manifests=(), asof, imported_usage=None, usage_catalog=None) -> dict:
    """Construct a strict public whitelist; callers publish the returned JSON.

    Performance weeks use actual exit boundaries. Usage weeks use actual request
    start times. Never expose a model's returns beyond the six-model common prefix.
    """
    validate_book(book, calendar=calendar)
    if set(book["models"]) != set(MODEL_IDS):
        raise ValueError("The public comparison requires exactly the registered six models")
    asof = aware(asof)
    histories = {model: book["accounts"][model]["history"] for model in MODEL_IDS}
    visible = []
    for model in MODEL_IDS:
        count = 0
        for row in histories[model]:
            if aware(row["exit_at"]) > asof or aware(row["settled_at"]) > asof:
                break
            count += 1
        visible.append(count)
    common = min(visible)
    for i in range(common):
        reference = histories[MODEL_IDS[0]][i]
        if any(any(histories[model][i][field] != reference[field] for field in
                   ("step", "release_at", "entry_at", "exit_at")) for model in MODEL_IDS):
            raise ValueError("Models do not share one completed release prefix")
    all_time = _view(histories, range(common))
    calls = [row for row in _calls(book, budget_ledger, imported_usage) if row["started_at"] <= asof]
    if usage_catalog is not None:
        validate_catalog(usage_catalog, book, _calls(book, budget_ledger, imported_usage), ledger=budget_ledger)
    grouped = defaultdict(list)
    for i in range(common):
        local = aware(histories[MODEL_IDS[0]][i]["exit_at"])
        grouped[local.isocalendar()[:2]].append(i)
    first_week, _ = week_bounds(calendar.start_at)
    last_week, _ = week_bounds(min(asof, calendar.valuation_nodes[-1]))
    last_week = max(first_week, last_week)
    weeks, start = [], first_week
    while start <= last_week:
        end = start + timedelta(days=7)
        key = start.isocalendar()[:2]
        scoped = [row for row in calls if start <= row["started_at"] < end]
        tokens = []
        for model in MODEL_IDS:
            model_calls = [row for row in scoped if row["model_id"] == model]
            row = {"model_id": model, **_token_totals(model_calls,
                evidence_available=budget_ledger is not None), **_recorded_tokens(model_calls,
                evidence_available=budget_ledger is not None), **_cost_totals(model_calls,
                evidence_available=budget_ledger is not None)}
            purpose = _purpose_totals(model_calls, model=model, catalog=usage_catalog, start=start, end=end,
                                      evidence_available=budget_ledger is not None)
            row.update({field: purpose[field] for field in PURPOSE_FIELDS})
            row["recorded_tokens"].update(purpose["recorded"])
            row["token_coverage"].update(purpose["coverage"])
            tokens.append(row)
        shared = _token_totals([row for row in scoped if row["model_id"] in SHARED_IDS],
                               evidence_available=budget_ledger is not None)
        weeks.append({"week_id": f"{key[0]:04d}-W{key[1]:02d}",
            "label": f"Week of {start.date().isoformat()}", "start": start.isoformat(), "end": end.isoformat(),
            **_view(histories, grouped[key]), "tokens": tokens, "tokens_coverage": _coverage(tokens),
            "shared_tokens": shared, "shared_tokens_coverage": _coverage([shared])})
        start = end
    published = _public_releases(release_manifests, asof, calendar)
    next_release = next((node for node in calendar.release_nodes if node > asof), None)
    upcoming = None if next_release is None else {"timestamp": next_release.isoformat(),
        "session_label": "NYSE shortened session" if next_release.date() in HALF_DAYS else "NYSE regular session",
        "note": "Only 09:30–12:30 observation nodes run on this half-day." if next_release.date() in HALF_DAYS
                else "Next actual observation node; exchange holidays and overnight gaps follow the verified calendar."}
    expected_settled = sum(calendar.action_window(at).exit_at <= asof for at in calendar.release_nodes)
    missing_decision = any(at <= asof and calendar.action_window(at).entry_at <= asof
        and any(at.isoformat() not in book["decisions"][model] for model in MODEL_IDS) for at in calendar.release_nodes)
    missing_release = any(at <= asof and at not in published for at in calendar.release_nodes)
    status = ("scheduled" if asof < calendar.start_at else "finished" if common == len(calendar.release_nodes)
              else "pending" if common < expected_settled or missing_decision or missing_release else "live")
    coverages = {row["coverage"] for row in published.values()}
    coverage = "pending" if not published else "complete" if coverages == {"complete"} else "unavailable" if coverages == {"unavailable"} else "partial"
    return {"schema": "moira.live-testing.website/1", "status": status, "timezone": "America/New_York",
        "updated_at": asof.isoformat(), "period": {"start": calendar.start_at.isoformat(),
            "last_release": calendar.last_release_at.isoformat(), "final_settlement": calendar.valuation_nodes[-1].isoformat()},
        "next_release": upcoming, "all_time": all_time, "weeks": weeks,
        "data_status": {"published_batches": len(published), "expected_release_points": len(calendar.release_nodes),
            "last_release_at": max(published).isoformat() if published else None, "coverage": coverage},
        "budget": _budget(budget_ledger, asof)}
