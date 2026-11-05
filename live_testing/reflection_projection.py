"""Pure, news-free views of finalized action counterfactuals.

Rendering creates one complete string, never a memory write or an LLM call.
Old reflection/reason text, source articles, retrieval scores and audit fields
are deliberately not inputs to the representation. The caller retains them.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import re

_ACTIONS = {'LS':'LS', 'SL':'SL', 'F':'F',
    'LONG_SHORT':'LS', 'SHORT_LONG':'SL', 'FLAT':'F'}
_LIFECYCLES = {'HEALTHY', 'WARNING', 'BREAK', 'HISTORICAL_SEED', 'UNKNOWN'}
_SYMBOL = re.compile(r'[A-Z][A-Z0-9.-]{0,14}\Z')
PROTOCOL = 'moira.news-free-cf-reflection/1'


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('A finalized finite numeric outcome is required')
    return float(value)


def _action(value):
    if not isinstance(value, str) or value not in _ACTIONS:
        raise ValueError('A recorded executed LS/SL/F action is required')
    return _ACTIONS[value]


def _pair(value):
    if isinstance(value, str): value = value.split('/')
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not isinstance(v, str) or _SYMBOL.fullmatch(v) is None for v in value)
            or value[0] == value[1]):
        raise ValueError('Two distinct recorded symbol identifiers are required')
    return tuple(value)


def _outcomes(value):
    if not isinstance(value, Mapping) or set(value) not in (
            {'LS', 'SL', 'F'}, {'LONG_SHORT', 'SHORT_LONG', 'FLAT'}):
        raise ValueError('All three explicit finalized counterfactual outcomes are required')
    result = {_action(k): _number(v) for k, v in value.items()}
    if result['F'] != 0.:
        raise ValueError('The simulated Flat action must have zero gross return')
    return result


def _percent(value):
    # Significant digits preserve tiny nonzero evidence instead of printing 0.
    return f'{value * 100:+.12g}%'


def build_reflection(action, pair, actual_return, counterfactual_returns,
                     long_z=None, trading_z=None, lifecycle=None):
    """Return a full outcome-based experience, without dated/news reasoning.

    Returns are decimal gross returns for the SAME finalized execution interval.
    Optional regime inputs are numerical z scores and a fixed lifecycle enum.
    They are descriptive observations, never a causal attribution or a mandate
    to repeat a profitable action. No input is sliced, summarized or persisted.
    """
    chosen = _action(action); a, b = _pair(pair)
    actual = _number(actual_return); cf = _outcomes(counterfactual_returns)
    if not math.isclose(actual, cf[chosen], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Actual return disagrees with the executed action counterfactual')
    z_long = None if long_z is None else _number(long_z)
    z_trade = None if trading_z is None else _number(trading_z)
    if lifecycle is not None and lifecycle not in _LIFECYCLES:
        raise ValueError('Only a known abstract lifecycle category may be projected')
    best_value = max(cf.values()); best = [k for k in ('LS', 'SL', 'F') if cf[k] == best_value]
    parts = [f'Completed experience for {a}/{b}: LS is long {a}/short {b}; '
        f'SL reverses those legs; F stays flat. The executed {chosen} action returned {_percent(actual)}.',
        'On the same frozen entry and exit prices, the gross counterfactual returns were '
        + ', '.join(f'{k} {_percent(cf[k])}' for k in ('LS', 'SL', 'F')) + '.']
    regime = []
    if z_long is not None: regime.append(f'long-horizon z={z_long:+.12g}')
    if z_trade is not None: regime.append(f'trading-window z={z_trade:+.12g}')
    if lifecycle is not None: regime.append(f'lifecycle={lifecycle}')
    if regime: parts.append('Recorded abstract state: ' + '; '.join(regime) + '.')
    if len(best) == 3:
        parts.append('All three alternatives tied in this interval. This outcome supplies no '
            'directional payoff discrimination; neither a directional rule nor an abstention '
            'rule is established by the tie.')
    elif chosen in best:
        parts.append('The executed action attained the highest gross counterfactual return'
            + (' jointly with ' + '/'.join(k for k in best if k != chosen) if len(best) > 1 else '') + '. '
            'Treat the observed success as a conditional experience to compare with later '
            'matching states, not as sufficient reason to repeat the action.')
    else:
        gap = best_value - cf[chosen]
        parts.append(f'The highest gross-return alternative was {"/".join(best)}, '
            f'which exceeded the chosen action by {_percent(gap)}. '
            + ('Flat forfeited a directional gross opportunity in this completed interval. '
                if chosen == 'F' else 'The chosen direction underperformed its available alternatives. ')
            + 'For a comparable future state, explicitly compare the opposite direction and '
            'abstention before repeating this choice; the appropriate action still requires '
            'evidence available before entry.')
    if z_trade is not None and z_trade != 0. and len(best) != 3:
        implied = 'SL' if z_trade > 0 else 'LS'
        parts.append(f'A simple mean-reversion reading of the recorded trading-window z '
            f'would point to {implied}; this '
            + ('matched' if implied in best else 'did not match')
            + ' the interval\'s highest gross-return alternative. This is a descriptive '
            'agreement check, not proof of reversion or continuation.')
    parts.append('Reusable constraint: compare both directional actions with Flat under '
        'comparable pre-entry conditions and wait for genuine exit feedback. These '
        'counterfactuals reuse one observed price path; they are not independent trades '
        'and do not identify which input caused the result. One episode does not establish '
        'a causal or generally profitable policy, and gross payoffs omit unmodeled execution costs.')
    return ' '.join(parts)


def render_unit_reflection(unit):
    """Render a MemoryUnit, flat retrieval hit, or serialized ``{'unit':...}``."""
    if isinstance(unit, Mapping):
        if 'unit' in unit and 'stock1' not in unit: unit = unit['unit']
        get = unit.get if isinstance(unit, Mapping) else lambda k, default=None: getattr(unit, k, default)
    else:
        if hasattr(unit, 'unit'): unit = unit.unit
        get = lambda k, default=None: getattr(unit, k, default)
    provenance = get('provenance'); provenance = provenance if isinstance(provenance, Mapping) else {}
    protocol = provenance.get('reflection_protocol')
    if protocol is not None and protocol != PROTOCOL:
        raise ValueError('An unknown structured reflection protocol cannot be trusted')
    regime = provenance.get('reflection_regime', {}) if protocol == PROTOCOL else {}
    if not isinstance(regime, Mapping) or set(regime) - {'long_z', 'trading_z', 'lifecycle'}:
        raise ValueError('Only structured abstract regime fields may enter the reflection')
    text = build_reflection(get('action'), (get('stock1'), get('stock2')),
        get('actual_return'), get('counterfactual_returns'), **regime)
    if protocol == PROTOCOL and get('reflection') != text:
        raise ValueError('The stored news-free reflection differs from its structured evidence')
    return text


def render_direction_reflection(record, *, decision=None):
    """Render a finalized native DirectionMemory record, with optional context.

    A full record supplies ``selected_pair``. If the native ``recent`` view
    omitted that field, its SAME historical decision must supply the pair.
    Never substitute the currently active pair for a different past action.
    """
    if not isinstance(record, Mapping) or decision is not None and not isinstance(decision, Mapping):
        raise ValueError('An immutable structured native record is required')
    decision = decision or {}; accepted = decision.get('action')
    accepted = accepted if isinstance(accepted, Mapping) else {}
    pair = record.get('selected_pair')
    supplied_pair = (accepted.get('stock1'), accepted.get('stock2')) if accepted else None
    if pair is None: pair = supplied_pair
    elif supplied_pair is not None and _pair(pair) != _pair(supplied_pair):
        raise ValueError('Historical native record and decision refer to different pairs')
    action = record.get('final_action', record.get('action'))
    if record.get('action') is not None and _action(record['action']) != _action(action):
        raise ValueError('Executed action labels disagree')
    if accepted and _action(accepted.get('action')) != _action(action):
        raise ValueError('Historical native record and accepted action disagree')
    actual = record.get('actual_pnl', record.get('realized_pnl'))
    cf = {'LS': record.get('cf_long_short'), 'SL': record.get('cf_short_long'), 'F': record.get('cf_flat')}
    monitor = decision.get('monitor'); monitor = monitor if isinstance(monitor, Mapping) else {}
    lifecycle = record.get('lifecycle_state', monitor.get('lifecycle'))
    return build_reflection(action, pair, actual, cf, long_z=decision.get('long_z'),
        trading_z=decision.get('trading_z'), lifecycle=lifecycle)
