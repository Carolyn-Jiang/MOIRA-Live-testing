import copy
from datetime import datetime,time,timedelta
import json
import math
import random
import unittest

from live_testing.calendar import NY,aware,is_trading_day
from live_testing.fast_memory import FastCounterfactualMemoryBank,IndexedExchangeSessionAge
from live_testing.memory import CounterfactualMemoryBank,MemoryUnit,Namespace


BASE=datetime(2026,10,19,11,30,tzinfo=NY)

def unit(index,*,available=None,exit_at=None,pair=('NVDA','GOOG')):
    exited=exit_at or BASE+timedelta(hours=index)
    return MemoryUnit(unit_id=f'u{index:04}',action_id=f'a{index:04}',stock1=pair[0],stock2=pair[1],action='LS',
        decision_at=exited-timedelta(hours=2),entry_at=exited-timedelta(hours=1),exit_at=exited,
        available_at=available or exited+timedelta(minutes=1),actual_return=.01,
        counterfactual_returns={'LS':.01,'SL':-.01,'F':0.},reflection='One complete synthetic reflection.',
        entities=pair,provenance={'synthetic':True})

def naive_age(then,as_of):
    then,as_of=aware(then),aware(as_of)
    if then>as_of:raise ValueError('Future age')
    day=then.date();result=0
    while day<=as_of.date():
        if is_trading_day(day)and then<datetime.combine(day,time(9,30),NY)<=as_of:result+=1
        day+=timedelta(days=1)
    return result

def banks(dimension=32,namespace=None,age=None):
    namespace=namespace or Namespace('moira2-top3','live','synthetic')
    return [kind(namespace,session_age=age or naive_age,embedding_id='synthetic-exact-vector',dimension=dimension)
        for kind in (CounterfactualMemoryBank,FastCounterfactualMemoryBank)]


class FastMemoryTests(unittest.TestCase):
    def test_random_queries_feedback_and_restores_are_bit_exact(self):
        rng=random.Random(3102);original,fast=banks();knowledge=BASE+timedelta(days=5)
        for index in range(35):
            value=unit(index,pair=('NVDA','GOOG')if index%2 else('WMT','JPM'))
            vector=[rng.uniform(-1.,1.)for _ in range(32)]
            for bank in (original,fast):bank.add_finalized(value,as_of=knowledge,embedding=vector)
        for step in range(25):
            at=knowledge+timedelta(hours=step);vector=[rng.uniform(-1.,1.)for _ in range(32)]
            options={'as_of':at,'top_k':(1,3,5)[step%3],'retrieval_id':str(step),'query_embedding':vector,
                'entities':['WMT']if step%4==0 else None}
            self.assertEqual(original.retrieve('actual query',**options),fast.retrieve('actual query',**options))
            reward=(.02,-.01,0.)[step%3]
            for bank in (original,fast):bank.apply_outcome_feedback(str(step),actual_return=reward,
                exit_at=at+timedelta(minutes=1),available_at=at+timedelta(minutes=2),as_of=at+timedelta(minutes=2))
        snapshot=original.to_dict();self.assertEqual(snapshot,fast.to_dict())
        restored=FastCounterfactualMemoryBank.from_dict(snapshot,namespace=original.namespace,session_age=IndexedExchangeSessionAge())
        self.assertEqual(restored.to_dict(),snapshot)
        self.assertLess(restored._fast_diagnostics['exact_cosines'],restored._fast_diagnostics['eligible_units'])

    def test_point_in_time_availability_and_market_cutoff_both_remain_required(self):
        original,fast=banks(2);knowledge=BASE+timedelta(days=4)
        for bank in (original,fast):
            bank.add_finalized(unit(0,available=knowledge),as_of=knowledge,embedding=[1.,0.])
            bank.add_finalized(unit(1,available=knowledge,exit_at=BASE+timedelta(hours=3)),as_of=knowledge,embedding=[1.,0.])
        for bank in (original,fast):
            first=bank.retrieve('historical prefix',as_of=knowledge,source_cutoff=BASE+timedelta(minutes=2),
                retrieval_id='r',top_k=5,query_embedding=[1.,0.])
            self.assertEqual([h.unit.unit_id for h in first.hits],['u0000'])
            bank.apply_outcome_feedback('r',actual_return=-1.,exit_at=BASE+timedelta(hours=2),available_at=knowledge,as_of=knowledge)
            self.assertEqual(bank.importance_at('u0000',knowledge,source_cutoff=BASE),50.)
            self.assertEqual(bank.importance_at('u0000',knowledge,source_cutoff=BASE+timedelta(hours=2)),32.)
        self.assertEqual(original.to_dict(),fast.to_dict())
        self.assertEqual(FastCounterfactualMemoryBank.from_dict(original.to_dict(),namespace=fast.namespace,
            session_age=IndexedExchangeSessionAge()).to_dict(),original.to_dict())

    def test_exact_near_ties_include_the_whole_boundary_and_keep_original_tie_breakers(self):
        original,fast=banks(2);at=BASE+timedelta(days=2)
        for index,angle in [(3,0.),(2,1e-6),(1,0.),(0,0.)]:
            for bank in (original,fast):bank.add_finalized(unit(index,exit_at=BASE),as_of=at,
                embedding=[math.cos(angle),math.sin(angle)])
        for bank in (original,fast):result=bank.retrieve('tie',as_of=at,top_k=3,retrieval_id='ties',query_embedding=[1.,0.])
        self.assertEqual(original.to_dict(),fast.to_dict())
        self.assertEqual([h.unit.unit_id for h in result.hits],['u0000','u0001','u0003'])
        self.assertEqual(fast._fast_diagnostics['exact_cosines'],4)

    def test_full_causal_snapshot_validation_still_rejects_score_order_and_namespace_tampering(self):
        original,fast=banks(2);at=BASE+timedelta(days=2)
        for bank in (original,fast):
            for index in range(3):bank.add_finalized(unit(index),as_of=at,embedding=[1.,float(index)])
            bank.retrieve('query',as_of=at,top_k=3,retrieval_id='actual',query_embedding=[1.,0.])
        snapshot=original.to_dict()
        for mutate in (lambda d:d['retrievals'][0]['hits'][0].update(score=123.),
                       lambda d:d['retrievals'][0]['hits'].reverse(),
                       lambda d:d['namespace'].update(scope='training')):
            bad=copy.deepcopy(snapshot);mutate(bad)
            with self.assertRaises(ValueError):FastCounterfactualMemoryBank.from_dict(bad,
                namespace=fast.namespace,session_age=IndexedExchangeSessionAge())

    def test_exported_training_units_need_explicit_new_live_namespace_and_do_not_transfer_feedback(self):
        training=FastCounterfactualMemoryBank(Namespace('moira2-top1','training','epoch'),session_age=naive_age,
            embedding_id='synthetic-exact-vector',dimension=2)
        at=BASE+timedelta(days=2);training.add_finalized(unit(0),as_of=at,embedding=[1.,0.])
        training.retrieve('q',as_of=at,top_k=1,retrieval_id='r',query_embedding=[1.,0.])
        training.apply_outcome_feedback('r',actual_return=1.,exit_at=at,available_at=at,as_of=at)
        original,fast=banks(2,namespace=Namespace('moira2-top1','live','new-epoch'))
        for bank in (original,fast):
            for value in training.export_units():bank.add_finalized(value,as_of=at,embedding=[1.,0.])
            self.assertEqual(bank.importance_at('u0000',at),50.)
            bank.retrieve('q',as_of=at,top_k=1,retrieval_id='new',query_embedding=[1.,0.])
        self.assertEqual(original.to_dict(),fast.to_dict())
        with self.assertRaises(ValueError):FastCounterfactualMemoryBank.from_dict(training.to_dict(),
            namespace=fast.namespace,session_age=IndexedExchangeSessionAge())

    def test_extreme_numerical_scale_uses_exact_fallback_and_add_invalidates_matrix(self):
        original,fast=banks(2);at=BASE+timedelta(days=2)
        for bank in (original,fast):
            bank.add_finalized(unit(0),as_of=at,embedding=[1e-120,0.])
            bank.retrieve('q',as_of=at,top_k=1,retrieval_id='one',query_embedding=[1e-120,0.])
            bank.add_finalized(unit(1),as_of=at,embedding=[0.,1e-120])
            bank.retrieve('q2',as_of=at,top_k=3,retrieval_id='two',query_embedding=[0.,1e-120])
        self.assertEqual(original.to_dict(),fast.to_dict());self.assertEqual(fast._fast_diagnostics['fallback_ranks'],2)

    def test_supplied_session_callback_is_not_replaced_and_only_repeated_exact_pairs_are_cached(self):
        calls=[]
        def callback(then,asof):calls.append((then,asof));return 123
        _original,fast=banks(2,age=callback);at=BASE+timedelta(days=2)
        fast.add_finalized(unit(0),as_of=at,embedding=[1.,0.])
        for key in ('r1','r2'):fast.retrieve('q',as_of=at,top_k=1,retrieval_id=key,query_embedding=[1.,0.])
        self.assertEqual(len(calls),1);self.assertEqual(fast.to_dict()['retrievals'][0]['hits'][0]['session_age'],123)

    def test_indexed_open_ordinals_match_native_calendar_on_holidays_preopen_and_dst(self):
        cached=IndexedExchangeSessionAge();rng=random.Random(999)
        dates=[datetime(2026,11,25,h,m,tzinfo=NY)for h,m in [(9,29),(9,30),(16,0)]]
        dates += [datetime(2026,11,day,9,30,tzinfo=NY)for day in (26,27,30)]
        dates += [datetime(2027,3,day,h,30,tzinfo=NY)for day,h in [(12,15),(15,8),(15,9),(26,9)]]
        for _ in range(70):dates.append(BASE+timedelta(days=rng.randrange(220),minutes=rng.randrange(1440)))
        for first in dates:
            for last in dates[::9]:
                if first<=last:self.assertEqual(cached(first,last),naive_age(first,last))
        with self.assertRaises(ValueError):cached(BASE+timedelta(days=1),BASE)


if __name__=='__main__':unittest.main()
