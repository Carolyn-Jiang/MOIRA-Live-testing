"""Offline memory causality and exact native-compound arithmetic."""
from datetime import datetime, timedelta
import math
import copy
import json
import unittest
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from live_testing.memory import CounterfactualMemoryBank, MemoryUnit, Namespace, TradingSessionClock, cosine

NY = ZoneInfo('America/New_York')
BASE = datetime(2026, 11, 25, 12, 30, tzinfo=NY)


def unit(key='a', *, exit_at=BASE, available_at=None, stock1='NVDA', stock2='GOOG', **changes):
    values = dict(unit_id=key, action_id='action-'+key, stock1=stock1, stock2=stock2, action='LS',
        decision_at=exit_at-timedelta(hours=2), entry_at=exit_at-timedelta(hours=1),
        exit_at=exit_at, available_at=available_at or exit_at+timedelta(minutes=1), actual_return=.02,
        counterfactual_returns={'LS': .02, 'SL': -.02, 'F': 0.}, reflection='One whole actual and counterfactual reflection.',
        entities=(stock1, stock2), provenance={'source_kind': 'synthetic-finalized-price-pair'})
    values.update(changes)
    return MemoryUnit(**values)


def bank(*, version='moira2-k3', scope='live', age=lambda _a, _b: 0, embed=None):
    return CounterfactualMemoryBank(Namespace(version, scope, 'test'), session_age=age,
        embedding_id='injected-synthetic-vector-v1', dimension=2, embed=embed)


class MemoryTests(unittest.TestCase):
    def test_native_compound_tau_252_is_divisor_not_multiplier(self):
        b = bank(age=lambda _a, _b: 252); b.add_finalized(unit(), as_of=BASE+timedelta(minutes=2), embedding=[1., 0.])
        hit = b.retrieve('query', as_of=BASE+timedelta(days=400), top_k=1, retrieval_id='one', query_embedding=[2., 0.]).hits[0]
        self.assertAlmostEqual(hit.relevance, 1.)
        self.assertAlmostEqual(hit.recency, math.exp(-1))
        self.assertAlmostEqual(hit.score, 1 + .5 + math.exp(-1))
        self.assertEqual(set(hit.unit.to_prompt()['counterfactual_returns']), {'LS', 'SL', 'F'})

    def test_holidays_weekends_preopen_and_halfday_use_session_count(self):
        opens = [datetime(2026, 11, d, 9, 30, tzinfo=NY) for d in (25, 27, 30)]
        clock = TradingSessionClock(opens)
        self.assertEqual(clock.age(BASE, datetime(2026, 11, 27, 12, 30, tzinfo=NY)), 1)
        self.assertEqual(clock.age(BASE, datetime(2026, 11, 30, 8, tzinfo=NY)), 1)
        self.assertEqual(clock.age(BASE, datetime(2026, 11, 30, 9, 30, tzinfo=NY)), 2)
        with self.assertRaises(ValueError): clock.age(BASE, BASE-timedelta(days=1))

    def test_unfinished_feedback_and_missing_cf_never_call_embedding(self):
        provider = Mock(return_value=[1., 0.]); b = bank(embed=provider)
        with self.assertRaises(ValueError): b.add_finalized(unit(), as_of=BASE)
        provider.assert_not_called()
        with self.assertRaises(ValueError): unit(counterfactual_returns={'LS': .02})
        with self.assertRaises(ValueError): unit(actual_return=.03)
        self.assertEqual(b.export_units(), ())

    def test_no_future_unit_and_entity_scope_in_retrieval(self):
        b = bank(); b.add_finalized(unit('old'), as_of=BASE+timedelta(minutes=2), embedding=[1., 0.])
        b.add_finalized(unit('future', exit_at=BASE+timedelta(hours=2)), as_of=BASE+timedelta(hours=3), embedding=[1., 0.])
        first = b.retrieve('query', as_of=BASE+timedelta(minutes=2), top_k=5, retrieval_id='first', query_embedding=[1., 0.])
        self.assertEqual([h.unit.unit_id for h in first.hits], ['old'])
        empty = b.retrieve('query', as_of=BASE+timedelta(minutes=2), top_k=3, retrieval_id='other', entities=['WMT'], query_embedding=[1., 0.])
        self.assertEqual(empty.hits, ())

    def test_feedback_step18_only_retrieved_evidence_and_no_future_importance(self):
        b = bank(); at = BASE+timedelta(minutes=2)
        for key, vector in [('a', [1., 0.]), ('b', [0., 1.])]: b.add_finalized(unit(key), as_of=at, embedding=vector)
        original = b.retrieve('query', as_of=at, top_k=1, retrieval_id='decision', query_embedding=[1., 0.])
        exited, available = at+timedelta(hours=1), at+timedelta(hours=1, minutes=1)
        with self.assertRaises(ValueError): b.apply_outcome_feedback('decision', actual_return=.01, exit_at=exited, available_at=available, as_of=exited)
        b.apply_outcome_feedback('decision', actual_return=.01, exit_at=exited, available_at=available, as_of=available)
        self.assertEqual(b.importance_at('a', available), 68)
        self.assertEqual(b.importance_at('b', available), 50)
        self.assertEqual(b.importance_at('a', at), 50)
        self.assertEqual(b.retrieve('query', as_of=at, top_k=1, retrieval_id='decision', query_embedding=[1., 0.]), original)
        with self.assertRaises(ValueError): b.apply_outcome_feedback('decision', actual_return=-.01, exit_at=exited, available_at=available, as_of=available)

    def test_late_initialization_knowledge_does_not_leak_later_market_outcomes(self):
        b = bank(); knowledge = BASE+timedelta(days=3)
        # A genuine old training unit can be initialized later than its source
        # market cutoff, while a current-day future exit remains unavailable.
        b.add_finalized(unit('seed', available_at=knowledge), as_of=knowledge, embedding=[1.,0.])
        b.add_finalized(unit('future', exit_at=BASE+timedelta(hours=2), available_at=knowledge), as_of=knowledge, embedding=[1.,0.])
        first = b.retrieve('query', as_of=knowledge, source_cutoff=BASE+timedelta(minutes=2),
            top_k=5, retrieval_id='backfill', query_embedding=[1.,0.])
        self.assertEqual([h.unit.unit_id for h in first.hits], ['seed'])
        b.apply_outcome_feedback('backfill', actual_return=1., exit_at=BASE+timedelta(hours=1),
            available_at=knowledge, as_of=knowledge)
        replay = b.retrieve('query', as_of=knowledge, source_cutoff=BASE+timedelta(minutes=3),
            top_k=1, retrieval_id='earlier-prefix', query_embedding=[1.,0.])
        self.assertEqual(replay.hits[0].raw_importance, 50)
        self.assertEqual(b.importance_at('seed', knowledge, source_cutoff=BASE+timedelta(hours=1)), 68)

    def test_raw_importance_is_not_clamped_and_zero_feedback_is_neutral(self):
        b = bank(); b.add_finalized(unit(), as_of=BASE+timedelta(minutes=2), embedding=[1., 0.])
        for i in range(5):
            at = BASE+timedelta(hours=i+1)
            b.retrieve('query', as_of=at, top_k=1, retrieval_id=str(i), query_embedding=[1., 0.])
            b.apply_outcome_feedback(str(i), actual_return=1., exit_at=at+timedelta(minutes=1), available_at=at+timedelta(minutes=2), as_of=at+timedelta(minutes=2))
        hit = b.retrieve('query', as_of=BASE+timedelta(hours=7), top_k=1, retrieval_id='last', query_embedding=[1., 0.]).hits[0]
        self.assertEqual(hit.raw_importance, 140)
        self.assertEqual(hit.normalized_importance, 1.)
        b.apply_outcome_feedback('last', actual_return=0., exit_at=BASE+timedelta(hours=8), available_at=BASE+timedelta(hours=8), as_of=BASE+timedelta(hours=8))
        self.assertEqual(b.importance_at('a', BASE+timedelta(hours=8)), 140)

    def test_stable_topk_ties_and_no_duplicate_action_units(self):
        b = bank(); at = BASE+timedelta(minutes=2)
        for key in ('c', 'b', 'a'): b.add_finalized(unit(key), as_of=at, embedding=[1., 0.])
        hits = b.retrieve('query', as_of=at, top_k=3, retrieval_id='stable', query_embedding=[1., 0.]).hits
        self.assertEqual([h.unit.unit_id for h in hits], ['a', 'b', 'c'])
        with self.assertRaises(ValueError): b.add_finalized(unit('other', action_id='action-a'), as_of=at, embedding=[1., 0.])

    def test_namespace_and_explicit_embedding_provider_isolation(self):
        first, live, variant = bank(scope='training'), bank(), bank(version='moira2-k5')
        first.add_finalized(unit(), as_of=BASE+timedelta(minutes=2), embedding=[1., 0.])
        self.assertEqual(live.export_units(), ()); self.assertEqual(variant.export_units(), ())
        with self.assertRaises(RuntimeError): live.add_finalized(unit(), as_of=BASE+timedelta(minutes=2))
        with self.assertRaises(ValueError): live.add_finalized(unit(), as_of=BASE+timedelta(minutes=2), embedding=[0., 0.])
        self.assertAlmostEqual(cosine([1., 0.], [-1., 0.]), -1.)
        with self.assertRaises(ValueError): cosine([1.], [1., 2.])

    def test_snapshot_roundtrip_is_offline_and_rejects_namespace_or_score_tamper(self):
        b = bank(); at = BASE+timedelta(minutes=2)
        b.add_finalized(unit(), as_of=at, embedding=[1., 0.])
        b.retrieve('query', as_of=at, top_k=1, retrieval_id='one', query_embedding=[1., 0.])
        exited = at+timedelta(hours=1)
        b.apply_outcome_feedback('one', actual_return=.01, exit_at=exited, available_at=exited, as_of=exited)
        snapshot = json.loads(json.dumps(b.to_dict()))
        forbidden = Mock(side_effect=AssertionError('Restoring known vectors must not embed'))
        restored = CounterfactualMemoryBank.from_dict(snapshot, namespace=b.namespace, session_age=lambda _a, _b: 0, embed=forbidden)
        self.assertEqual(restored.to_dict(), snapshot)
        forbidden.assert_not_called()
        with self.assertRaises(ValueError): CounterfactualMemoryBank.from_dict(snapshot, namespace=Namespace('other', 'live', 'test'), session_age=lambda _a, _b: 0)
        bad = copy.deepcopy(snapshot); bad['retrievals'][0]['hits'][0]['score'] += 1
        with self.assertRaises(ValueError): CounterfactualMemoryBank.from_dict(bad, namespace=b.namespace, session_age=lambda _a, _b: 0)


if __name__ == '__main__': unittest.main()
