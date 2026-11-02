"""Causal, version-isolated Moira 2.0 counterfactual reflection retrieval.

One memory is one exited action's actual/LS/SL/F returns and whole reflection.
FinMem-style rank = cosine + min(raw_importance, 100)/100 + exp(-sessions/252).
The native configurable feedback mechanism is retained: initial importance 50
is this bank's engineering default; step 18 matches the checked fixed-pair
FinMem configuration. Raw importance is never silently clamped or decayed.

No implicit embeddings/calendar, network, SDK, persistence, or model call exists.
Historical facts' availability and real generation audit belong in provenance;
backfill uses an explicit knowledge-as-of distinct from its market observation.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence


ACTIONS = frozenset({'LS', 'SL', 'F'})


def instant(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('A timezone-aware timestamp is required')
    return value.astimezone(timezone.utc)


def number(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite numeric evidence is required')
    return float(value)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Namespace:
    version: str
    scope: str
    run_id: str

    def __post_init__(self):
        if self.scope not in {'training', 'live'} or not self.version.strip() or not self.run_id.strip():
            raise ValueError('Explicit version, training/live scope, and run ID are required')


class TradingSessionClock:
    """Explicit exchange opens; holidays absent, half-days count as one session.

    Before Monday's opening, the current ordinal is Friday's. The application
    supplies a verified exchange calendar; weekdays are never guessed here.
    """
    def __init__(self, opens: Sequence[datetime]):
        self.opens = tuple(instant(value) for value in opens)
        if not self.opens or tuple(sorted(set(self.opens))) != self.opens:
            raise ValueError('Strictly ordered explicit trading-session opens required')

    def ordinal(self, at: datetime) -> int:
        index = bisect_right(self.opens, instant(at)) - 1
        if index < 0: raise ValueError('Timestamp precedes the supplied exchange calendar')
        return index

    def age(self, then: datetime, as_of: datetime) -> int:
        if instant(then) > instant(as_of): raise ValueError('Future session age is forbidden')
        return self.ordinal(as_of) - self.ordinal(then)


@dataclass(frozen=True)
class MemoryUnit:
    unit_id: str
    action_id: str
    stock1: str
    stock2: str
    action: str
    decision_at: datetime
    entry_at: datetime
    exit_at: datetime
    available_at: datetime
    actual_return: float
    counterfactual_returns: Mapping[str, float]
    reflection: str
    entities: tuple[str, ...]
    provenance: Mapping[str, object]

    def __post_init__(self):
        if (not self.unit_id or not self.action_id or not self.stock1 or self.stock1 == self.stock2
                or not self.stock2 or self.action not in ACTIONS or not self.reflection.strip()):
            raise ValueError('One complete exited action and one whole reflection are required')
        for name in ('decision_at', 'entry_at', 'exit_at', 'available_at'):
            object.__setattr__(self, name, instant(getattr(self, name)))
        if not self.decision_at <= self.entry_at < self.exit_at <= self.available_at:
            raise ValueError('Counterfactual memory cannot precede its executed exit')
        if set(self.counterfactual_returns) != ACTIONS:
            raise ValueError('All three LS/SL/F counterfactual outcomes are required')
        outcomes = {key: number(value) for key, value in self.counterfactual_returns.items()}
        actual = number(self.actual_return)
        if not math.isclose(actual, outcomes[self.action], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Actual outcome must match the executed action counterfactual')
        entities = tuple(sorted(set(self.entities)))
        if not {self.stock1, self.stock2}.issubset(entities):
            raise ValueError('Memory entity provenance must include both traded stocks')
        if not self.provenance: raise ValueError('Explicit source provenance is required')
        json.dumps(self.provenance, allow_nan=False)
        object.__setattr__(self, 'counterfactual_returns', outcomes)
        object.__setattr__(self, 'actual_return', actual)
        object.__setattr__(self, 'entities', entities)
        object.__setattr__(self, 'provenance', copy.deepcopy(dict(self.provenance)))

    def to_prompt(self) -> dict:
        return {'unit_id': self.unit_id, 'action_id': self.action_id,
            'stock1': self.stock1, 'stock2': self.stock2, 'action': self.action,
            **{key: getattr(self, key).isoformat() for key in ('decision_at', 'entry_at', 'exit_at', 'available_at')},
            'actual_return': self.actual_return, 'counterfactual_returns': dict(self.counterfactual_returns),
            'reflection': self.reflection, 'entities': list(self.entities), 'provenance': copy.deepcopy(self.provenance)}


@dataclass(frozen=True)
class RetrievalHit:
    unit: MemoryUnit
    score: float
    relevance: float
    recency: float
    raw_importance: float
    normalized_importance: float
    session_age: int

    def to_prompt(self):
        return {**self.unit.to_prompt(), 'retrieval': {key: getattr(self, key) for key in (
            'score', 'relevance', 'recency', 'raw_importance', 'normalized_importance', 'session_age')}}


@dataclass(frozen=True)
class RetrievalBatch:
    retrieval_id: str
    namespace: Namespace
    as_of: datetime
    source_cutoff: datetime
    top_k: int
    hits: tuple[RetrievalHit, ...]
    embedding_id: str


def cosine(first, second) -> float:
    if len(first) != len(second) or not first: raise ValueError('Equal nonempty embedding dimensions required')
    a, b = tuple(map(number, first)), tuple(map(number, second))
    x, y = math.sqrt(math.fsum(v*v for v in a)), math.sqrt(math.fsum(v*v for v in b))
    if not x or not y: raise ValueError('A zero vector is not a valid semantic embedding')
    # Only floating-point cosine overshoot is bounded; negative relevance stays.
    return max(-1.0, min(1.0, math.fsum(v*w for v, w in zip(a, b))/(x*y)))


class CounterfactualMemoryBank:
    def __init__(self, namespace: Namespace, *, session_age: Callable[[datetime, datetime], int],
                 embedding_id: str, dimension: int, embed: Callable[[str], Sequence[float]] | None = None,
                 initial_importance=50.0, feedback_step=18.0, tau_sessions=252.0):
        if not embedding_id or type(dimension) is not int or dimension <= 0:
            raise ValueError('An explicit embedding provider/model identity and dimension are required')
        if number(tau_sessions) != 252.0:
            raise ValueError('This experiment fixes the recency time constant to 252 trading sessions')
        if number(feedback_step) <= 0: raise ValueError('Positive native feedback step required')
        self.namespace, self.session_age, self.embedding_id = namespace, session_age, embedding_id
        self.dimension, self.embed = dimension, embed
        self.initial_importance, self.feedback_step, self.tau = number(initial_importance), number(feedback_step), number(tau_sessions)
        self._units, self._action_ids, self._vectors = {}, {}, {}
        self._retrievals, self._feedback = {}, {}

    def _vector(self, text, supplied):
        if supplied is None:
            if self.embed is None: raise RuntimeError('An explicit embedding provider or precomputed vector is required')
            supplied = self.embed(text)
        vector = tuple(map(number, supplied))
        if len(vector) != self.dimension: raise ValueError('Embedding dimension changed')
        cosine(vector, vector)
        return vector

    def add_finalized(self, unit: MemoryUnit, *, as_of: datetime, embedding=None):
        as_of = instant(as_of)
        if unit.available_at > as_of: raise ValueError('Unfinished or not-yet-available feedback cannot enter memory')
        if unit.unit_id in self._units:
            if self._units[unit.unit_id].to_prompt() != unit.to_prompt(): raise ValueError('An immutable memory unit changed')
            return unit.unit_id
        if unit.action_id in self._action_ids: raise ValueError('One exited action may form only one memory unit')
        vector = self._vector(json.dumps(unit.to_prompt(), sort_keys=True), embedding)
        self._units[unit.unit_id], self._vectors[unit.unit_id] = copy.deepcopy(unit), vector
        self._action_ids[unit.action_id] = unit.unit_id
        return unit.unit_id

    def importance_at(self, unit_id, as_of, *, source_cutoff=None):
        as_of = instant(as_of)
        cutoff = instant(source_cutoff) if source_cutoff is not None else as_of
        if unit_id not in self._units: raise KeyError(unit_id)
        return self.initial_importance + self.feedback_step * sum(event['direction'] for event in self._feedback.values()
            if unit_id in event['unit_ids'] and event['available_at'] <= as_of and event['exit_at'] <= cutoff)

    def retrieve(self, query: str, *, as_of: datetime, top_k: int, retrieval_id: str,
                 entities: Sequence[str] | None = None, query_embedding=None, source_cutoff=None) -> RetrievalBatch:
        as_of = instant(as_of)
        cutoff = instant(source_cutoff) if source_cutoff is not None else as_of
        if cutoff > as_of: raise ValueError('Market cutoff cannot exceed knowledge availability')
        if top_k not in (1, 3, 5) or type(top_k) is not int or not retrieval_id or not query.strip():
            raise ValueError('Explicit query/retrieval ID and top K in 1/3/5 required')
        entity_filter = tuple(sorted(set(entities or ())))
        signature = (query, as_of, top_k, entity_filter, self.embedding_id, cutoff)
        if retrieval_id in self._retrievals:
            old = self._retrievals[retrieval_id]
            if old['signature'] != signature or (query_embedding is not None and tuple(query_embedding) != old['vector']):
                raise ValueError('An accepted retrieval query cannot change')
            return copy.deepcopy(old['batch'])
        vector = self._vector(query, query_embedding)
        batch = RetrievalBatch(retrieval_id, self.namespace, as_of, cutoff, top_k,
            self._rank(vector, as_of, top_k, entity_filter, cutoff), self.embedding_id)
        self._retrievals[retrieval_id] = {'signature': signature, 'vector': vector, 'batch': copy.deepcopy(batch)}
        return batch

    def _rank(self, vector, as_of, top_k, entity_filter, cutoff):
        hits = []
        for key, unit in self._units.items():
            if unit.available_at > as_of or unit.exit_at > cutoff or (entity_filter and not set(entity_filter).intersection(unit.entities)):
                continue
            age = self.session_age(unit.exit_at, cutoff)
            if type(age) is not int or age < 0: raise ValueError('Causal integer exchange-session age required')
            recency = math.exp(-age/self.tau)
            importance = self.importance_at(key, as_of, source_cutoff=cutoff)
            normalized = min(importance, 100.0)/100.0
            relevance = cosine(vector, self._vectors[key])
            hits.append(RetrievalHit(copy.deepcopy(unit), relevance + normalized + recency,
                                     relevance, recency, importance, normalized, age))
        hits.sort(key=lambda h: (-h.score, -h.unit.exit_at.timestamp(), h.unit.unit_id))
        return tuple(hits[:top_k])

    def apply_outcome_feedback(self, retrieval_id: str, *, actual_return: float, exit_at: datetime,
                               available_at: datetime, as_of: datetime):
        exit_at, available_at, as_of = map(instant, (exit_at, available_at, as_of))
        if retrieval_id not in self._retrievals: raise ValueError('Feedback must reference an actual retrieval')
        batch = self._retrievals[retrieval_id]['batch']
        if not batch.source_cutoff <= exit_at <= available_at <= as_of:
            raise ValueError('Only already-exited, available feedback can update retrieved evidence')
        actual = number(actual_return)
        event = {'actual_return': actual, 'exit_at': exit_at, 'available_at': available_at,
            'direction': 1 if actual > 0 else -1 if actual < 0 else 0,
            'unit_ids': tuple(hit.unit.unit_id for hit in batch.hits)}
        if retrieval_id in self._feedback:
            if self._feedback[retrieval_id] != event: raise ValueError('Accepted outcome feedback is immutable')
            return
        self._feedback[retrieval_id] = event

    def export_units(self):
        """Explicit caller-controlled transfer only; another bank is never queried."""
        return tuple(copy.deepcopy(unit) for unit in self._units.values())

    def to_dict(self):
        """Finite JSON snapshot only; the application owns atomic private I/O."""
        result = {'schema': 'moira.counterfactual-memory/1', 'namespace': vars(self.namespace),
            'embedding_id': self.embedding_id, 'dimension': self.dimension,
            'initial_importance': self.initial_importance, 'feedback_step': self.feedback_step,
            'tau_sessions': self.tau,
            'units': [{'unit': unit.to_prompt(), 'vector': list(self._vectors[key])} for key, unit in self._units.items()],
            'retrievals': [], 'feedback': []}
        for key, record in self._retrievals.items():
            query, as_of, top_k, entities, _, cutoff = record['signature']
            result['retrievals'].append({'retrieval_id': key, 'query': query, 'as_of': as_of.isoformat(),
                'source_cutoff': cutoff.isoformat(), 'top_k': top_k, 'entities': list(entities), 'query_vector': list(record['vector']),
                'hits': [{'unit_id': hit.unit.unit_id, **{name: getattr(hit, name) for name in (
                    'score', 'relevance', 'recency', 'raw_importance', 'normalized_importance', 'session_age')}}
                    for hit in record['batch'].hits]})
        for key, event in self._feedback.items():
            result['feedback'].append({'retrieval_id': key, 'actual_return': event['actual_return'],
                'exit_at': event['exit_at'].isoformat(), 'available_at': event['available_at'].isoformat()})
        json.dumps(result, allow_nan=False)
        return copy.deepcopy(result)

    @classmethod
    def from_dict(cls, data, *, namespace, session_age, embed=None, embedding_id=None):
        if data.get('schema') != 'moira.counterfactual-memory/1' or Namespace(**data['namespace']) != namespace:
            raise ValueError('Memory snapshots cannot cross version/scope/run namespaces')
        if embedding_id is not None and data['embedding_id'] != embedding_id:
            raise ValueError('Embedding provider/model identity changed')
        bank = cls(namespace, session_age=session_age, embedding_id=data['embedding_id'],
            dimension=data['dimension'], embed=embed, initial_importance=data['initial_importance'],
            feedback_step=data['feedback_step'], tau_sessions=data['tau_sessions'])
        for item in data['units']:
            raw = copy.deepcopy(item['unit'])
            for name in ('decision_at', 'entry_at', 'exit_at', 'available_at'):
                raw[name] = datetime.fromisoformat(raw[name])
            raw['entities'] = tuple(raw['entities'])
            unit = MemoryUnit(**raw)
            bank.add_finalized(unit, as_of=unit.available_at, embedding=item['vector'])
        saved = {}
        for row in data['retrievals']:
            key = row['retrieval_id']
            if key in saved: raise ValueError('Duplicate retrieval provenance')
            as_of = instant(datetime.fromisoformat(row['as_of']))
            cutoff = instant(datetime.fromisoformat(row.get('source_cutoff', row['as_of'])))
            if cutoff > as_of: raise ValueError('Stored market cutoff exceeds availability')
            entities = tuple(sorted(set(row['entities'])))
            vector = bank._vector(row['query'], row['query_vector'])
            hits = []
            for raw in row['hits']:
                unit = bank._units[raw['unit_id']]
                if unit.available_at > as_of or unit.exit_at > cutoff: raise ValueError('A stored retrieval includes future memory')
                hits.append(RetrievalHit(copy.deepcopy(unit), **{name: raw[name] for name in (
                    'score', 'relevance', 'recency', 'raw_importance', 'normalized_importance', 'session_age')}))
            batch = RetrievalBatch(key, namespace, as_of, cutoff, row['top_k'], tuple(hits), bank.embedding_id)
            bank._retrievals[key] = {'signature': (row['query'], as_of, row['top_k'], entities, bank.embedding_id, cutoff),
                                     'vector': vector, 'batch': batch}
            saved[key] = row
        for row in data['feedback']:
            available = datetime.fromisoformat(row['available_at'])
            bank.apply_outcome_feedback(row['retrieval_id'], actual_return=row['actual_return'],
                exit_at=datetime.fromisoformat(row['exit_at']), available_at=available, as_of=available)
        for key, record in bank._retrievals.items():
            _, as_of, top_k, entities, _, cutoff = record['signature']
            expected = bank._rank(record['vector'], as_of, top_k, entities, cutoff)
            if expected != record['batch'].hits:
                raise ValueError('Stored retrieval scores/order disagree with causal source evidence')
        return bank
