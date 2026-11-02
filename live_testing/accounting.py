"""Pure, append-only delayed-execution accounting; no I/O, API or clock reads.

Callers supply actual generation/observation times and persist atomically under
their own lock. Missing decisions/quotes remain pending, never synthetic Flat.
"""
from __future__ import annotations

import copy
from datetime import datetime
import json
import math
import re
import statistics
from typing import Iterable, Mapping

from .calendar import DEFAULT_CALENDAR, ExperimentCalendar, aware, month_bounds, week_bounds

SCHEMA = "moira.live-testing.accounting/1"


def _number(value, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("A finite numeric value is required")
    return float(value)


def _stamp(value):
    return aware(value).isoformat()


def _metadata(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("JSON-object audit metadata required")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("Audit metadata must be finite JSON data") from error


def new_book(models: Iterable[str], symbols: Iterable[str]) -> dict:
    models, symbols = list(models), list(symbols)
    for names in (models, symbols):
        if not names or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names):
            raise ValueError("Nonempty exact model/symbol names required")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Duplicate model/symbol names")
    if len(symbols) < 2:
        raise ValueError("At least two eligible symbols are required")
    if "__shared__" in models:
        raise ValueError("The shared-cost reporting key is reserved")
    return {"schema": SCHEMA, "models": models, "symbols": symbols,
            "accounts": {model: {"equity": 1.0, "history": []} for model in models},
            "decisions": {model: {} for model in models}, "valuations": {}, "usage": {}}


def _prices(book, prices):
    if not isinstance(prices, Mapping) or set(prices) != set(book["symbols"]):
        raise ValueError("A complete exact-symbol valuation is required")
    result = {symbol: _number(prices[symbol]) for symbol in book["symbols"]}
    if any(value <= 0 for value in result.values()):
        raise ValueError("Every quote must be positive; missing prices stay pending")
    return result


def _action(book, action):
    if (not isinstance(action, Mapping) or set(action) != {"stock1", "stock2", "action"}
            or action["stock1"] not in book["symbols"] or action["stock2"] not in book["symbols"]
            or action["stock1"] == action["stock2"] or action["action"] not in {"LS", "SL", "F"}):
        raise ValueError("Two distinct eligible symbols and one LS/SL/F action required")
    return dict(action)


def pair_return(action, entry: Mapping[str, float], exit: Mapping[str, float]) -> float:
    """Equal half-weight legs, unit gross exposure; no costs or annualization."""
    if (not isinstance(action, Mapping) or set(action) != {"stock1", "stock2", "action"}
            or action["stock1"] == action["stock2"] or action["action"] not in {"LS", "SL", "F"}):
        raise ValueError("An explicit valid pair action is required; missing is never Flat")
    a, b = action["stock1"], action["stock2"]
    try:
        values = [_number(value) for value in (entry[a], entry[b], exit[a], exit[b])]
    except (KeyError, TypeError) as error:
        raise ValueError("Both pair legs need exact quotes") from error
    if any(value <= 0 for value in values):
        raise ValueError("Positive exact entry and exit prices required")
    if action["action"] == "F":
        return 0.0
    value = 0.5 * (values[2] / values[0] - values[3] / values[1])
    if action["action"] == "SL":
        value = -value
    elif action["action"] != "LS":
        raise ValueError("Unknown pair action")
    return value


def _decision(book, model, release_at, action, generated_at, input_asof, mode, audit, calendar):
    if model not in book["models"]:
        raise ValueError("Unknown independent model account")
    window = calendar.action_window(release_at)
    generated, supplied = aware(generated_at), aware(input_asof)
    if supplied != window.release_at or generated < window.release_at:
        raise ValueError("Only the same released observation may produce this action")
    if mode == "live":
        if generated >= window.entry_at:
            raise ValueError("A live decision cannot be generated at/after entry")
    elif mode == "retrospective_point_in_time":
        if (not isinstance(audit, Mapping) or not isinstance(audit.get("authorization"), str)
                or not audit["authorization"].strip() or not isinstance(audit.get("input_sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", audit["input_sha256"])):
            raise ValueError("Explicit replay authorization and frozen input SHA256 required")
    else:
        raise ValueError("Unknown execution mode")
    return {**window.as_dict(), "generated_at": generated.isoformat(), "input_asof": supplied.isoformat(),
            "mode": mode, "action": _action(book, action), "audit": _metadata(audit)}


def record_decision(book, model: str, release_at, action, *, generated_at, input_asof,
                    mode="live", audit=None, calendar=DEFAULT_CALENDAR) -> dict:
    validate_book(book, calendar=calendar)
    record = _decision(book, model, release_at, action, generated_at, input_asof, mode, audit, calendar)
    key = record["release_at"]
    if key in book["decisions"][model] and book["decisions"][model][key] != record:
        raise ValueError("An accepted decision and its true timestamps are immutable")
    book["decisions"][model][key] = copy.deepcopy(record)
    return copy.deepcopy(record)


def record_valuation(book, at, prices, *, observed_at, provenance=None,
                     calendar=DEFAULT_CALENDAR) -> dict:
    validate_book(book, calendar=calendar)
    at, observed = aware(at), aware(observed_at)
    if at not in calendar.valuation_nodes or observed < at:
        raise ValueError("A reached canonical valuation node is required")
    record = {"at": at.isoformat(), "observed_at": observed.isoformat(),
              "prices": _prices(book, prices), "provenance": _metadata(provenance)}
    existing = book["valuations"].get(record["at"])
    if existing is not None:
        # A later fetch of identical values may not rewrite the original audit.
        if existing["prices"] != record["prices"]:
            raise ValueError("An immutable complete valuation cannot be replaced")
        return copy.deepcopy(existing)
    book["valuations"][record["at"]] = copy.deepcopy(record)
    return copy.deepcopy(record)


def _settlement(book, decision, previous_equity, asof):
    entry = book["valuations"].get(decision["entry_at"])
    exit = book["valuations"].get(decision["exit_at"])
    if entry is None or exit is None:
        return None, "missing_valuation"
    available = max(aware(decision["exit_at"]), aware(decision["generated_at"]),
                    aware(entry["observed_at"]), aware(exit["observed_at"]))
    if available > asof:
        return None, "valuation_or_decision_not_yet_available"
    ret = pair_return(decision["action"], entry["prices"], exit["prices"])
    equity = previous_equity * (1 + ret)
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("Nonpositive equity requires an explicit insolvency policy; never clip or Flat")
    return {key: decision[key] for key in ("step", "release_at", "entry_at", "exit_at")} | {
        "settled_at": asof.isoformat(), "step_return": ret, "equity": equity,
        "cumulative_return": equity - 1}, None


def validate_book(book, *, calendar=DEFAULT_CALENDAR) -> None:
    if not isinstance(book, dict) or book.get("schema") != SCHEMA:
        raise ValueError("Wrong independent accounting schema")
    shape = new_book(book["models"], book["symbols"])
    if set(book["accounts"]) != set(shape["accounts"]) or set(book["decisions"]) != set(shape["decisions"]):
        raise ValueError("Independent model account set changed")
    _validate_usage(book)
    for key, quote in book["valuations"].items():
        at = aware(key)
        if (key != at.isoformat() or quote["at"] != key or at not in calendar.valuation_nodes
                or aware(quote["observed_at"]) < at):
            raise ValueError("Invalid immutable valuation timestamp")
        _prices(book, quote["prices"])
    for model in book["models"]:
        for key, record in book["decisions"][model].items():
            rebuilt = _decision(book, model, key, record["action"], record["generated_at"],
                record["input_asof"], record["mode"], record["audit"], calendar)
            if rebuilt != record:
                raise ValueError("A decision's causal interval or timestamp changed")
        account = book["accounts"][model]
        equity = 1.0
        if not isinstance(account["history"], list) or len(account["history"]) > len(calendar.release_nodes):
            raise ValueError("Invalid history length")
        for index, actual in enumerate(account["history"]):
            key = calendar.release_nodes[index].isoformat()
            decision = book["decisions"][model].get(key)
            if decision is None:
                raise ValueError("History cannot turn a missing decision into Flat")
            expected, pending = _settlement(book, decision, equity, aware(actual["settled_at"]))
            if pending or set(actual) != set(expected):
                raise ValueError("Missing or prematurely settled history")
            for field in expected:
                if field in {"step_return", "equity", "cumulative_return"}:
                    if not math.isclose(_number(actual[field]), expected[field], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Return or cumulative equity differs from immutable evidence")
                elif actual[field] != expected[field]:
                    raise ValueError("Settlement timeline differs from its decision")
            equity = expected["equity"]
        if not math.isclose(_number(account["equity"]), equity, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Independent equity differs from its settled prefix")


def settle_due(book, asof, *, calendar=DEFAULT_CALENDAR) -> dict:
    """Atomically settle each model's complete prefix; stop at its first gap."""
    validate_book(book, calendar=calendar)
    instant = aware(asof)
    candidate = copy.deepcopy(book)
    report = {"settlements": [], "pending": {}}
    for model in candidate["models"]:
        account = candidate["accounts"][model]
        for release in calendar.release_nodes[len(account["history"]):]:
            window = calendar.action_window(release)
            if window.exit_at > instant:
                break
            decision = candidate["decisions"][model].get(release.isoformat())
            if decision is None:
                report["pending"][model] = {"release_at": release.isoformat(), "reason": "missing_decision"}
                break
            settlement, reason = _settlement(candidate, decision, account["equity"], instant)
            if reason:
                report["pending"][model] = {"release_at": release.isoformat(), "reason": reason}
                break
            account["history"].append(settlement)
            account["equity"] = settlement["equity"]
            report["settlements"].append({"model": model, **copy.deepcopy(settlement)})
    validate_book(candidate, calendar=calendar)
    book.clear(); book.update(candidate)
    return report


def matured_feedback(book, model, asof, *, observation_asof=None, calendar=DEFAULT_CALENDAR) -> list[dict]:
    validate_book(book, calendar=calendar)
    if model not in book["models"]:
        raise ValueError("Unknown independent model")
    instant, rows = aware(asof), []
    cutoff = instant if observation_asof is None else aware(observation_asof)
    if cutoff > instant:
        raise ValueError("An observation cutoff cannot be future information")
    for row in book["accounts"][model]["history"]:
        if aware(row["exit_at"]) > cutoff or aware(row["settled_at"]) > instant:
            continue
        decision = book["decisions"][model][row["release_at"]]
        rows.append({**copy.deepcopy(row), "action": copy.deepcopy(decision["action"]),
                     "entry_prices": copy.deepcopy(book["valuations"][row["entry_at"]]["prices"]),
                     "exit_prices": copy.deepcopy(book["valuations"][row["exit_at"]]["prices"])})
    return rows


def risk_metrics(step_returns: Iterable[float]) -> dict:
    returns = [_number(value) for value in step_returns]
    equity = peak = 1.0
    maximum = 0.0
    for value in returns:
        equity *= 1 + value
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("Metrics require a positive finite equity curve")
        peak = max(peak, equity)
        maximum = max(maximum, 1 - equity / peak)
    deviation = statistics.stdev(returns) if len(returns) > 1 else 0.0
    scale = max((abs(value) for value in returns), default=0.0)
    sharpe = None if deviation <= max(1e-15, scale * 1e-12) else statistics.mean(returns) / deviation
    return {"completed_rounds": len(returns), "cumulative_return": equity - 1,
            "sharpe_ratio": sharpe, "max_drawdown": maximum}


def account_metrics(book, model, *, asof=None, calendar=DEFAULT_CALENDAR) -> dict:
    validate_book(book, calendar=calendar)
    if model not in book["models"]:
        raise ValueError("Unknown independent model")
    rows = book["accounts"][model]["history"]
    if asof is not None:
        instant = aware(asof)
        rows = [row for row in rows if aware(row["settled_at"]) <= instant]
    return risk_metrics(row["step_return"] for row in rows)


def _usage_record(book, model, call_id, at, cost_usd, input_tokens, output_tokens, purpose, provider_model):
    if model is not None and model not in book["models"]:
        raise ValueError("Unknown usage account; use None for shared data preparation")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("An immutable provider/request identity is required")
    if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens)):
        raise ValueError("Nonnegative integer token counts required")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("Usage purpose required")
    if provider_model is not None and (not isinstance(provider_model, str) or not provider_model.strip()):
        raise ValueError("A provider-model name must be text")
    return {"model": model, "at": _stamp(at), "cost_usd": _number(cost_usd, nonnegative=True),
              "input_tokens": input_tokens, "output_tokens": output_tokens,
              "purpose": purpose, "provider_model": provider_model}


def _validate_usage(book):
    if not isinstance(book["usage"], dict):
        raise ValueError("Immutable usage records must be keyed by call ID")
    for call_id, row in book["usage"].items():
        rebuilt = _usage_record(book, row["model"], call_id, row["at"], row["cost_usd"],
                               row["input_tokens"], row["output_tokens"], row["purpose"], row["provider_model"])
        if row != rebuilt:
            raise ValueError("Usage fields or NY timestamp changed")


def record_usage(book, model, *, call_id: str, at, cost_usd, input_tokens=0,
                 output_tokens=0, purpose="inference", provider_model=None,
                 calendar=DEFAULT_CALENDAR) -> dict:
    validate_book(book, calendar=calendar)
    record = _usage_record(book, model, call_id, at, cost_usd, input_tokens, output_tokens, purpose, provider_model)
    if call_id in book["usage"] and book["usage"][call_id] != record:
        raise ValueError("Provider usage cannot be overwritten or double-counted")
    book["usage"][call_id] = record
    return copy.deepcopy(record)


def _usage(book, at, bounds):
    _validate_usage(book)
    start, end = bounds(at)
    rows = [row for row in book["usage"].values() if start <= aware(row["at"]) < end]
    by_model = {}
    for model in [*book["models"], None]:
        selected = [row for row in rows if row["model"] == model]
        by_model[model if model is not None else "__shared__"] = {
            "calls": len(selected), "cost_usd": math.fsum(row["cost_usd"] for row in selected),
            "input_tokens": sum(row["input_tokens"] for row in selected),
            "output_tokens": sum(row["output_tokens"] for row in selected)}
    return {"start": start.isoformat(), "end_exclusive": end.isoformat(), "calls": len(rows),
            "cost_usd": math.fsum(row["cost_usd"] for row in rows), "by_model": by_model}


def weekly_usage(book, at) -> dict:
    result = _usage(book, at, week_bounds)
    result["iso_year"], result["iso_week"], _ = aware(at).isocalendar()
    return result


def monthly_usage(book, at) -> dict:
    return _usage(book, at, month_bounds)
