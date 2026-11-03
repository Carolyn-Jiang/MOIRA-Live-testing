"""Pure quarter-clock, bounded scores, used-memory EMA and restore checks."""
import copy
from dataclasses import replace
from datetime import date,datetime,time,timedelta,timezone
import math
import unittest

from live_testing.calendar import NY,is_trading_day
from live_testing.memory import MemoryUnit,Namespace
from live_testing.memory_v2 import (CounterfactualMemoryBankV2 as Bank,
    NYSEMarketHoursClock,TAU_MARKET_HOURS,SCHEMA)


def at(day='2026-10-19',hour=11,minute=30):
    return datetime.combine(date.fromisoformat(day),time(hour,minute),NY)


def unit(uid,*,exit_at=None,available_at=None):
    exit_at=exit_at or at();available_at=available_at or exit_at
    return MemoryUnit(uid,'action-'+uid,'NVDA','GOOG','LS',exit_at-timedelta(hours=2),
        exit_at-timedelta(hours=1),exit_at,available_at,.01,{'LS':.01,'SL':-.01,'F':0.},
        'One whole finalized synthetic counterfactual reflection.',('NVDA','GOOG'),{'source':'explicit fixture'})


def bank(scope='training',version='moira2-top3'):
    return Bank(Namespace(version,scope,'synthetic'),embedding_id='explicit-fixture-vectors',dimension=2)


def add(bank,uid='u1',vector=None,**kwargs):
    value=unit(uid,**kwargs);bank.add_finalized(value,as_of=value.available_at,embedding=vector or [1.,0.]);return value


def feedback(bank,event,query,used,labels,*,exit_at=None,available_at=None,as_of=None,cutoff=None):
    exit_at=exit_at or at(hour=12);available_at=available_at or exit_at;as_of=as_of or available_at
    return bank.apply_attributed_feedback(event,retrieval_id=query,used_memory_ids=used,
        attributions={uid:{'label':labels[uid],'evidence_sha256':'a'*64}for uid in used},
        action_evidence_sha256='b'*64,outcome_evidence_sha256='c'*64,exit_at=exit_at,
        available_at=available_at,as_of=as_of,source_cutoff=cutoff or exit_at)


class QuarterClockTests(unittest.TestCase):
    def setUp(self):self.clock=NYSEMarketHoursClock()
    def test_regular_session_and_same_day_hour_resolution(self):
        self.assertEqual(self.clock.hours(at(hour=9),at(hour=16,minute=0)),6.5)
        self.assertEqual(self.clock.hours(at(),at(hour=12)),1.)
        self.assertEqual(self.clock.hours(at(hour=15),at(hour=16,minute=0)),.5)
        self.assertEqual(self.clock.hours(at(hour=16,minute=0),at('2026-10-20',9,30)),0.)
    def test_holiday_halfday_and_weekend_contribute_only_actual_open_hours(self):
        self.assertEqual(self.clock.hours(at('2026-11-27',9,30),at('2026-11-27',16,0)),3.5)
        self.assertEqual(self.clock.hours(at('2026-11-27',12,30),at('2026-11-27',14,30)),.5)
        self.assertEqual(self.clock.hours(at('2026-11-26',9,30),at('2026-11-26',16,0)),0.)
        self.assertEqual(self.clock.hours(at('2026-10-23',15,0),at('2026-10-26',10,0)),1.5)
        self.assertEqual(self.clock.hours(at('2026-12-24',12,0),at('2026-12-28',10,0)),1.5)
    def test_quarter_is_63_regular_sessions_and_efold_not_half_life(self):
        days=[];day=date(2027,1,19)
        while len(days)<63:
            if is_trading_day(day):days.append(day)
            day+=timedelta(days=1)
        hours=self.clock.hours(datetime.combine(days[0],time(9,30),NY),datetime.combine(days[-1],time(16),NY))
        self.assertEqual(hours,409.5);self.assertEqual(hours,TAU_MARKET_HOURS)
        self.assertAlmostEqual(math.exp(-hours/TAU_MARKET_HOURS),1/math.e)
    def test_dst_closed_hours_and_clock_coverage_are_explicit(self):
        begin=at('2027-03-12',15,30);end=at('2027-03-15',10,30)
        self.assertEqual(self.clock.hours(begin,end),1.5)
        self.assertEqual(self.clock.hours(begin.astimezone(timezone.utc),end.astimezone(timezone.utc)),1.5)
        for first,last in [(end,begin),(at('2026-10-18'),at()),(at(),at('2027-05-26'))]:
            with self.assertRaises(ValueError):self.clock.hours(first,last)


class MemoryV2Tests(unittest.TestCase):
    def test_explicit_cross_variant_unit_provenance_is_rejected(self):
        b=bank();value=replace(unit('other'),provenance={'model_id':'moira2-top5'})
        with self.assertRaises(ValueError):b.add_finalized(value,as_of=at(),embedding=[1.,0.])
        self.assertEqual(b.to_dict()['units'],[])
    def test_readding_identical_finalized_unit_never_reembeds(self):
        calls=[]
        b=Bank(Namespace('v2','training','unit'),embedding_id='fixture',dimension=2,
            embed=lambda text:(calls.append(text)or[1.,0.]))
        value=unit('same');b.add_finalized(value,as_of=at());b.add_finalized(value,as_of=at())
        self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError):b.add_finalized(value,as_of=at(),embedding=[-1.,0.])
    def test_each_score_component_is_bounded_including_opposite_cosine(self):
        b=bank();add(b,'positive',[1.,0.]);add(b,'negative',[-1.,0.])
        batch=b.retrieve('actual query',as_of=at(),source_cutoff=at(),top_k=3,retrieval_id='q',query_embedding=[1.,0.])
        self.assertEqual([h.raw_cosine for h in batch.hits],[1.,-1.])
        self.assertEqual([h.relevance for h in batch.hits],[1.,0.])
        self.assertEqual([h.score for h in batch.hits],[2.5,1.5])
        for h in batch.hits:
            self.assertTrue(all(0<=v<=1 for v in (h.relevance,h.recency,h.importance)))
            self.assertTrue(0<=h.score<=3)
            self.assertEqual(h.to_prompt()['retrieval']['clock']['half_day_close'],'13:00')
    def test_same_day_recency_uses_economic_exit_not_knowledge_timestamp(self):
        b=bank();value=add(b,available_at=at(hour=13))
        first=b.retrieve('first',as_of=at(hour=14),source_cutoff=at(hour=12),top_k=1,retrieval_id='q1',query_embedding=[1.,0.]).hits[0]
        second=b.retrieve('second',as_of=at(hour=15),source_cutoff=at(hour=13),top_k=1,retrieval_id='q2',query_embedding=[1.,0.]).hits[0]
        self.assertEqual(first.market_hours_age,1.);self.assertEqual(second.market_hours_age,2.)
        self.assertAlmostEqual(first.recency,math.exp(-1/409.5));self.assertGreater(first.recency,second.recency)
    def test_three_positive_negative_and_neutral_ema_updates(self):
        for label,expected in [('helpful',1-.5*.98**3),('harmful',.5*.96**3),('neutral',.5)]:
            b=bank();add(b)
            for i in range(3):
                source=at()+timedelta(minutes=10*i);exit=source+timedelta(minutes=5)
                b.retrieve('q'+str(i),as_of=source,source_cutoff=source,top_k=1,retrieval_id='q'+str(i),query_embedding=[1.,0.])
                self.assertTrue(feedback(b,'f'+str(i),'q'+str(i),['u1'],{'u1':label},exit_at=exit))
            state=b.importance_state_at('u1',exit,source_cutoff=exit)
            self.assertAlmostEqual(state['importance'],expected);self.assertEqual(state['feedback_count'],3)
            self.assertEqual(state['updated_at'],exit.isoformat())
    def test_only_explicitly_used_memory_updates_and_empty_used_is_recorded(self):
        b=bank();add(b,'a');add(b,'b');b.retrieve('q',as_of=at(),source_cutoff=at(),top_k=3,retrieval_id='q',query_embedding=[1.,0.])
        feedback(b,'f','q',['a'],{'a':'harmful'})
        self.assertEqual(b.importance_at('a',at(hour=12)),.48)
        self.assertEqual(b.importance_at('b',at(hour=12)),.5)
        b.retrieve('q2',as_of=at(hour=12),source_cutoff=at(hour=12),top_k=3,retrieval_id='q2',query_embedding=[1.,0.])
        feedback(b,'f2','q2',[],{},exit_at=at(hour=13))
        self.assertEqual(b.importance_state_at('a',at(hour=13))['feedback_count'],1)
        self.assertEqual(b.importance_state_at('b',at(hour=13))['feedback_count'],0)
        self.assertEqual(len(b.to_dict()['feedback']),2)
    def test_feedback_is_atomic_idempotent_and_requires_real_evidence_ids(self):
        b=bank();add(b);b.retrieve('q',as_of=at(),source_cutoff=at(),top_k=1,retrieval_id='q',query_embedding=[1.,0.]);before=b.to_dict()
        with self.assertRaises(ValueError):feedback(b,'bad','q',['unprovided'],{'unprovided':'helpful'})
        self.assertEqual(b.to_dict(),before)
        self.assertTrue(feedback(b,'f','q',['u1'],{'u1':'helpful'}));snapshot=b.to_dict()
        self.assertFalse(feedback(b,'f','q',['u1'],{'u1':'helpful'},as_of=at(hour=13)))
        self.assertEqual(b.to_dict(),snapshot)
        with self.assertRaises(ValueError):feedback(b,'f','q',['u1'],{'u1':'harmful'})
        with self.assertRaises(ValueError):feedback(b,'other','q',['u1'],{'u1':'helpful'})
        with self.assertRaises(ValueError):
            b.apply_attributed_feedback('bad2',retrieval_id='q',used_memory_ids=['u1'],attributions={'u1':{'label':'helpful','evidence_sha256':'not-a-hash'}},
                action_evidence_sha256='b'*64,outcome_evidence_sha256='c'*64,exit_at=at(hour=12),available_at=at(hour=12),as_of=at(hour=12),source_cutoff=at(hour=12))
    def test_future_source_and_knowledge_guards_never_invent_flat_or_credit(self):
        b=bank();add(b)
        with self.assertRaises(ValueError):b.retrieve('q',as_of=at(),source_cutoff=at(hour=12),top_k=1,retrieval_id='q',query_embedding=[1.,0.])
        future=unit('future',exit_at=at(hour=12),available_at=at(hour=13))
        with self.assertRaises(ValueError):b.add_finalized(future,as_of=at(hour=12),embedding=[1.,0.])
        b.add_finalized(future,as_of=at(hour=13),embedding=[1.,0.])
        batch=b.retrieve('q',as_of=at(),source_cutoff=at(),top_k=3,retrieval_id='q',query_embedding=[1.,0.])
        self.assertEqual(len(batch.hits),1);before=b.to_dict()
        with self.assertRaises(ValueError):feedback(b,'f','q',['u1'],{'u1':'helpful'},exit_at=at(hour=12),as_of=at(),cutoff=at())
        with self.assertRaises(ValueError):feedback(b,'f','q',['u1'],{'u1':'helpful'},exit_at=at(hour=12),available_at=at(hour=13),as_of=at(hour=12))
        self.assertEqual(b.to_dict(),before)
    def test_equal_timestamp_feedback_revision_does_not_rewrite_accepted_query(self):
        b=bank();add(b);b.retrieve('q0',as_of=at(),source_cutoff=at(),top_k=1,retrieval_id='q0',query_embedding=[1.,0.])
        old=b.retrieve('q1',as_of=at(hour=12),source_cutoff=at(hour=12),top_k=1,retrieval_id='q1',query_embedding=[1.,0.])
        feedback(b,'f','q0',['u1'],{'u1':'helpful'})
        new=b.retrieve('q2',as_of=at(hour=12),source_cutoff=at(hour=12),top_k=1,retrieval_id='q2',query_embedding=[1.,0.])
        self.assertEqual(old.hits[0].importance,.5);self.assertEqual(new.hits[0].importance,.51)
        restored=Bank.from_dict(b.to_dict(),namespace=b.namespace)
        self.assertEqual(restored.to_dict(),b.to_dict())
    def test_feedback_index_matches_full_causal_event_scan_for_every_revision(self):
        b=bank();add(b,'a');add(b,'b')
        for i,(uid,label)in enumerate([('a','helpful'),('b','harmful'),('a','neutral')]):
            stamp=at()+timedelta(minutes=i*20)
            b.retrieve(str(i),as_of=stamp,source_cutoff=stamp,top_k=3,retrieval_id=str(i),query_embedding=[1.,0.])
            feedback(b,'f'+str(i),str(i),[uid],{uid:label},exit_at=stamp+timedelta(minutes=10))
        events=b.to_dict()['feedback']
        for uid in ('a','b'):
            for revision in range(4):
                for cutoff in [at()+timedelta(minutes=10),at()+timedelta(minutes=30),at()+timedelta(minutes=50)]:
                    value=.5;count=0
                    for event in events[:revision]:
                        if uid in event['attributions'] and datetime.fromisoformat(event['exit_at'])<=cutoff:
                            target,rate={'helpful':(1.,.02),'harmful':(0.,.04),'neutral':(.5,.01)}[event['attributions'][uid]['label']]
                            value=(1-rate)*value+rate*target;count+=1
                    actual=b.importance_state_at(uid,at(hour=13),source_cutoff=cutoff,feedback_revision=revision)
                    self.assertEqual(actual['importance'],value);self.assertEqual(actual['feedback_count'],count)
    def test_scores_sort_deterministically_and_support_all_three_topks(self):
        b=bank()
        for uid in ['b','a','c','d','e']:add(b,uid)
        for k in (1,3,5):
            batch=b.retrieve('q'+str(k),as_of=at(),source_cutoff=at(),top_k=k,retrieval_id='q'+str(k),query_embedding=[1.,0.])
            self.assertEqual([h.unit.unit_id for h in batch.hits],sorted(['b','a','c','d','e'])[:k])
        with self.assertRaises(ValueError):b.retrieve('invalid',as_of=at(),source_cutoff=at(),top_k=2,retrieval_id='bad',query_embedding=[1.,0.])
    def test_snapshot_restore_rejects_schema_namespace_policy_score_and_ema_tamper(self):
        b=bank();add(b);b.retrieve('q',as_of=at(),source_cutoff=at(),top_k=1,retrieval_id='q',query_embedding=[1.,0.]);feedback(b,'f','q',['u1'],{'u1':'helpful'});snapshot=b.to_dict()
        self.assertEqual(snapshot['schema'],SCHEMA);self.assertEqual(Bank.from_dict(snapshot,namespace=b.namespace).to_dict(),snapshot)
        mutations=[lambda s:s.update(schema='moira.counterfactual-memory/1'),
            lambda s:s['clock'].update(half_day_close='16:00'),lambda s:s.update(tau_market_hours=252),
            lambda s:s['retrievals'][0]['hits'][0].update(relevance=-1),
            lambda s:s['units'][0]['state'].update(importance=.99),
            lambda s:s['feedback'][0]['unit_updates']['u1'].update(after=.99),
            lambda s:s['retrievals'][0].update(feedback_revision=9)]
        for mutate in mutations:
            changed=copy.deepcopy(snapshot);mutate(changed)
            with self.assertRaises(ValueError):Bank.from_dict(changed,namespace=b.namespace)
        with self.assertRaises(ValueError):Bank.from_dict(snapshot,namespace=Namespace('other','training','synthetic'))
        self.assertEqual(b.to_dict(),snapshot)
    def test_verified_training_seed_preserves_learned_ema_without_new_embeddings(self):
        trained=bank();add(trained);trained.retrieve('q',as_of=at(),source_cutoff=at(),top_k=1,retrieval_id='q',query_embedding=[1.,0.]);feedback(trained,'f','q',['u1'],{'u1':'helpful'})
        frozen=trained.to_dict();live=bank('live');completion=at('2026-11-02',8,0);cutoff=at('2026-10-23',16,0)
        basis=live.seed_from_training_bank(frozen,checkpoint_sha256='d'*64,completed_at=completion,source_cutoff=cutoff,as_of=completion)
        self.assertEqual(live.importance_at('u1',completion),.51)
        self.assertEqual(live.importance_state_at('u1',completion)['feedback_count'],1)
        self.assertEqual(live.to_dict()['units'][0]['state']['updated_at'],completion.isoformat())
        self.assertEqual(trained.to_dict(),frozen)
        self.assertEqual(Bank.from_dict(live.to_dict(),namespace=live.namespace).to_dict(),live.to_dict())
        with self.assertRaises(ValueError):live.importance_at('u1',completion-timedelta(seconds=1))
        changed=live.to_dict();changed['units'][0]['initial_state']['importance']=.9
        with self.assertRaises(ValueError):Bank.from_dict(changed,namespace=live.namespace)
        with self.assertRaises(ValueError):bank('live','another-variant').seed_from_training_bank(frozen,checkpoint_sha256='d'*64,completed_at=completion,source_cutoff=cutoff,as_of=completion)


if __name__=='__main__':unittest.main()
