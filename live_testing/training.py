"""General-length, causal weekly native-component training orchestration.

The adapter supplies REAL native select/action/reflect/propose/accept/reject
operations. This core never invents an action, reward, response, or checkpoint.
Only the previous NY calendar week's observations and already-matured labels
are retained. The objective is sample Sharpe, without annualization.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import statistics
from typing import Mapping, Protocol

from .calendar import NY, aware, week_bounds, DEFAULT_CALENDAR
from .accounting import pair_return
from .memory import Namespace


@dataclass(frozen=True)
class Observation:
    action_id: str
    timestamp: datetime
    prices: Mapping[str, float]
    news: Mapping[str, list[str]]
    available_at: datetime

    @classmethod
    def from_dict(cls, row):
        at = aware(row['timestamp'])
        return cls(row.get('round_id', row.get('action_id', at.isoformat())), at,
            copy.deepcopy(row['prices']), copy.deepcopy(row['news']), aware(row.get('available_at', row['timestamp'])))

    def as_dict(self):
        return {'round_id': self.action_id, 'timestamp': self.timestamp.isoformat(),
                'prices': copy.deepcopy(self.prices), 'news': copy.deepcopy(self.news)}


@dataclass(frozen=True)
class Execution:
    action_id: str
    entry_at: datetime
    exit_at: datetime
    available_at: datetime
    entry_prices: Mapping[str, float]
    exit_prices: Mapping[str, float]
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class WeeklyDataset:
    training_at: datetime
    observations: tuple[Observation, ...]
    executions: Mapping[str, Execution]
    pending_action_ids: tuple[str, ...]

    @classmethod
    def build(cls, observations, executions, *, training_at):
        cutoff = aware(training_at)
        if cutoff.weekday() != 0 or (cutoff.hour, cutoff.minute, cutoff.second, cutoff.microsecond) != (8, 0, 0, 0):
            raise ValueError('Weekly training has the explicit Monday 08:00 New York cutoff')
        current_week, _ = week_bounds(cutoff)
        previous_week, _ = week_bounds(current_week-timedelta(days=1))
        rows = tuple(copy.deepcopy(row) for row in observations if previous_week <= aware(row.timestamp) < current_week)
        if not rows or tuple(sorted(row.timestamp for row in rows)) != tuple(row.timestamp for row in rows):
            raise ValueError('Chronological observations from the previous week are required')
        if len({row.action_id for row in rows}) != len(rows) or len({row.timestamp for row in rows}) != len(rows):
            raise ValueError('Duplicate historical observation')
        for row in rows:
            DEFAULT_CALENDAR.action_window(row.timestamp)
            if (not row.action_id or aware(row.available_at) < aware(row.timestamp) or aware(row.available_at) > cutoff
                    or not row.prices or set(row.prices) != set(row.news)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in row.prices.values())
                    or any(not isinstance(v, list) or not all(isinstance(s, str) for s in v) for v in row.news.values())):
                raise ValueError('Complete, available historical prices/news required')
        known, ids = {}, {row.action_id: row for row in rows}
        prices_at = {aware(row.timestamp): row.prices for row in rows}
        for item in executions:
            if item.action_id not in ids: continue
            if aware(item.exit_at) > cutoff or aware(item.available_at) > cutoff:
                continue  # Drop future price labels entirely, never replace with zero.
            if item.action_id in known: raise ValueError('Duplicate outcome label')
            if not ids[item.action_id].timestamp <= aware(item.entry_at) < aware(item.exit_at) <= aware(item.available_at):
                raise ValueError('Canonical ordered delayed outcome evidence required')
            canonical = DEFAULT_CALENDAR.action_window(ids[item.action_id].timestamp)
            if aware(item.entry_at) != canonical.entry_at or aware(item.exit_at) != canonical.exit_at:
                raise ValueError('Training label must use the actual canonical delayed execution grid')
            # Validate all prices now; no label is passed to an earlier decision.
            symbols = set(ids[item.action_id].prices)
            if set(item.entry_prices) != symbols or set(item.exit_prices) != symbols or not item.provenance:
                raise ValueError('Complete immutable delayed price snapshots/provenance required')
            for values in (item.entry_prices, item.exit_prices):
                if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values.values()):
                    raise ValueError('Finite positive outcome prices required')
            if ((aware(item.entry_at) in prices_at and item.entry_prices != prices_at[aware(item.entry_at)])
                    or (aware(item.exit_at) in prices_at and item.exit_prices != prices_at[aware(item.exit_at)])):
                raise ValueError('Outcome prices differ from immutable shared published observations')
            known[item.action_id] = copy.deepcopy(item)
        return cls(cutoff, rows, known, tuple(row.action_id for row in rows if row.action_id not in known))


@dataclass(frozen=True)
class FinalizedFeedback:
    action_id: str
    decision: Mapping[str, object]
    observation_at: datetime
    execution: Execution
    actual_return: float
    counterfactual_returns: Mapping[str, float]


@dataclass(frozen=True)
class EpisodeResult:
    decisions: tuple[Mapping[str, object], ...]
    finalized: tuple[FinalizedFeedback, ...]
    pending_action_ids: tuple[str, ...]
    checkpoint_after: Mapping[str, object]
    metrics: Mapping[str, float | None]


def objective(returns):
    values = list(returns)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Only finite realized returns can be scored')
    sharpe = None
    if len(values) > 1:
        sd = statistics.stdev(values)
        if sd > 8*math.ulp(1.0)*max(1., max(map(abs, values))): sharpe = statistics.mean(values)/sd
    equity = peak = 1.; drawdown = 0.
    for ret in values:
        equity *= 1+ret
        if equity <= 0 or not math.isfinite(equity): raise ValueError('Nonpositive training equity cannot be clipped')
        peak = max(peak, equity); drawdown = max(drawdown, 1-equity/peak)
    return {'sharpe_ratio': sharpe, 'cumulative_return': equity-1, 'max_drawdown': drawdown}


class EpisodePolicy(Protocol):
    def decide(self, observation: Observation) -> Mapping[str, object]: ...
    def on_feedback(self, feedback: FinalizedFeedback) -> None: ...
    def checkpoint(self) -> Mapping[str, object]: ...


class NativeAdapter(Protocol):
    def begin_episode(self, checkpoint, namespace: Namespace) -> EpisodePolicy: ...
    def reflect(self, finalized: tuple[FinalizedFeedback, ...], checkpoint) -> Mapping: ...
    def propose(self, checkpoint, reflection, cycle: int, baseline_metrics) -> Mapping: ...
    def accept(self, candidate_checkpoint, reference_checkpoint, reflection, cycle: int, baseline_metrics, candidate_metrics) -> Mapping: ...
    def reject(self, candidate_checkpoint, reference_checkpoint, reflection, cycle: int, baseline_metrics, candidate_metrics, reason: str) -> Mapping: ...


def active_prompt(checkpoint):
    return checkpoint['prompt_memory']['active_prompt']


def simulate_episode(adapter, checkpoint, dataset: WeeklyDataset, namespace: Namespace):
    if namespace.scope != 'training': raise ValueError('Training must not use a live memory bank')
    policy = adapter.begin_episode(copy.deepcopy(checkpoint), namespace)
    decisions, available, finished = {}, {}, {}
    def mature(market_at, knowledge_at):
        for row in dataset.observations:
            key = row.action_id; label = dataset.executions.get(key)
            if (key in finished or key not in decisions or label is None
                    or aware(label.exit_at) > market_at or aware(label.available_at) > knowledge_at): continue
            action = {key: decisions[row.action_id][key] for key in ('stock1', 'stock2', 'action')}
            cf = {name: pair_return({**action, 'action': name}, label.entry_prices, label.exit_prices) for name in ('LS', 'SL', 'F')}
            feedback = FinalizedFeedback(row.action_id, copy.deepcopy(decisions[row.action_id]), row.timestamp,
                label, cf[action['action']], cf)
            policy.on_feedback(copy.deepcopy(feedback)); finished[key] = feedback
    for row in dataset.observations:
        mature(aware(row.timestamp), aware(row.available_at))
        action = policy.decide(copy.deepcopy(row))
        pair_return({key: action[key] for key in ('stock1', 'stock2', 'action')}, row.prices, row.prices)
        decisions[row.action_id] = copy.deepcopy(action)
    # Final known outcomes only; no terminal observation or fake decision.
    mature(dataset.training_at, dataset.training_at)
    finalized = tuple(finished[row.action_id] for row in dataset.observations if row.action_id in finished)
    after = policy.checkpoint()
    if active_prompt(after) != active_prompt(checkpoint): raise RuntimeError('An episode must not update the frozen prompt')
    return EpisodeResult(tuple(decisions[row.action_id] for row in dataset.observations), finalized,
        tuple(row.action_id for row in dataset.observations if row.action_id not in finished), copy.deepcopy(after),
        objective(item.actual_return for item in finalized))


def train_native(adapter: NativeAdapter, dataset: WeeklyDataset, checkpoint, *, version: str,
                 run_id: str, max_cycles=8):
    """Native-component loop; strict sample-Sharpe accept, explicit rollback.

    Direction banks are fresh per episode; native persistent pair memory is
    returned by the adapter. Pending actions never reach reflection. Early-stop
    structure follows native: two rejects or two <5% relative objective changes,
    gated on at least one accepted update. Undefined Sharpe is never made zero.
    """
    if type(max_cycles) is not int or not 1 <= max_cycles <= 8: raise ValueError('At most eight native cycles per invocation')
    current = copy.deepcopy(checkpoint); cycles = []; accepted = rejects = 0; previous = None; changes = []
    for cycle in range(1, max_cycles+1):
        baseline = simulate_episode(adapter, current, dataset, Namespace(version, 'training', f'{run_id}:{cycle}:baseline'))
        current = baseline.checkpoint_after
        trades = sum(item.decision['action'] != 'F' for item in baseline.finalized)
        item = {'cycle': cycle, 'baseline_metrics': dict(baseline.metrics), 'matured_labels': len(baseline.finalized),
            'pending_labels': len(baseline.pending_action_ids), 'trade_count': trades, 'accepted': False, 'early_stop': False}
        if not trades:
            item['reason'] = 'NO_EVIDENCE'; cycles.append(item); continue
        reflection = adapter.reflect(copy.deepcopy(baseline.finalized), copy.deepcopy(current))
        item['reflection'] = copy.deepcopy(reflection)
        if trades < 2:
            item['reason'] = 'LOW_EVIDENCE_analysis_only'; cycles.append(item); continue
        candidate_seed = adapter.propose(copy.deepcopy(current), reflection, cycle, baseline.metrics)
        candidate = simulate_episode(adapter, candidate_seed, dataset, Namespace(version, 'training', f'{run_id}:{cycle}:candidate'))
        before, after = baseline.metrics['sharpe_ratio'], candidate.metrics['sharpe_ratio']
        improve = before is not None and after is not None and after > before
        reason = 'strict_sample_sharpe_improvement' if improve else 'undefined_sharpe_comparison' if before is None or after is None else 'candidate_sharpe_not_strictly_better'
        if improve:
            current = adapter.accept(candidate.checkpoint_after, current, reflection, cycle, baseline.metrics, candidate.metrics)
            accepted += 1; rejects = 0
        else:
            prior_prompt = active_prompt(current)
            current = adapter.reject(candidate.checkpoint_after, current, reflection, cycle, baseline.metrics, candidate.metrics, reason)
            if active_prompt(current) != prior_prompt: raise RuntimeError('Rejected candidate was not rolled back')
            rejects += 1
        if before is not None:
            if previous is not None: changes.append(abs(before-previous)/max(abs(previous), 1e-9))
            previous = before
        would_stop = rejects >= 2 or (len(changes) >= 2 and changes[-1] < .05 and changes[-2] < .05)
        stop = bool(accepted and would_stop)
        item.update(candidate_metrics=dict(candidate.metrics), accepted=improve, reason=reason,
            accepted_updates=accepted, consecutive_rejections=rejects, early_stop=stop,
            early_stop_blocked=bool(would_stop and not accepted))
        cycles.append(item)
        if stop: break
    return {'checkpoint': copy.deepcopy(current), 'cycles': cycles, 'completed_cycles': len(cycles),
        'max_cycles': max_cycles, 'objective': 'sample_sharpe_without_annualization',
        'training_at': dataset.training_at.isoformat(), 'observation_count': len(dataset.observations),
        'matured_label_count': len(dataset.executions), 'pending_action_ids': list(dataset.pending_action_ids),
        'accepted_updates': accepted}
