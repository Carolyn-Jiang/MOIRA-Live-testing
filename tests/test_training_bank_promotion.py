import copy
from datetime import datetime,timedelta
import unittest

from live_testing.calendar import NY
from live_testing.memory import CounterfactualMemoryBank,MemoryUnit,Namespace
from live_testing.training_bank_promotion import promote_training_bank


class PromotionTests(unittest.TestCase):
    def fixture(self):
        knowledge=datetime(2026,11,2,11,40,tzinfo=NY);age=lambda _a,_b:0
        bank=CounterfactualMemoryBank(Namespace('moira2-top1','training','real-episode'),session_age=age,
            embedding_id='explicit-synthetic',dimension=2)
        times=[datetime(2026,10,19,h,30,tzinfo=NY) for h in (11,12,14)]
        for i,at in enumerate(times):
            unit=MemoryUnit(str(i),'action-'+str(i),'NVDA','GOOG','LS',at-timedelta(hours=2),at-timedelta(hours=1),
                at,at,.01,{'LS':.01,'SL':-.01,'F':0.},'One whole finalized actual and counterfactual reflection.',
                ('NVDA','GOOG'),{'model_id':'moira2-top1','reflection_recorded_at':knowledge.isoformat()})
            bank.add_finalized(unit,as_of=knowledge,embedding=[1.,0.])
        for i,at in enumerate(times[:2]):
            bank.retrieve('query',as_of=knowledge,source_cutoff=at,top_k=1,retrieval_id=str(i),query_embedding=[1.,0.])
            bank.apply_outcome_feedback(str(i),actual_return=1.,exit_at=times[i+1],available_at=times[i+1],as_of=knowledge)
        return bank,knowledge,age

    def promote(self,snapshot,knowledge,age):
        return promote_training_bank(snapshot,model_id='moira2-top1',completed_at=knowledge+timedelta(minutes=20),
            training_window_end=datetime(2026,10,23,16,tzinfo=NY),live_source_cutoff=datetime(2026,11,2,9,30,tzinfo=NY),
            session_age=age,expected_units=3)

    def test_preserves_actual_feedback_scores_vectors_and_causal_source_cutoffs(self):
        bank,knowledge,age=self.fixture();snapshot=bank.to_dict();unchanged=copy.deepcopy(snapshot)
        result=self.promote(snapshot,knowledge,age);promoted=result['bank'];proof=result['proof']
        self.assertEqual(snapshot,unchanged);self.assertTrue(proof['causal_rank_recomputed'])
        self.assertTrue(proof['all_raw_importance_preserved']);self.assertEqual(proof['new_model_or_embedding_calls'],0)
        self.assertEqual(promoted['namespace'],{'version':'moira2-top1','scope':'live','run_id':'experiment'})
        self.assertEqual([r['source_cutoff'] for r in promoted['retrievals']],[r['source_cutoff'] for r in snapshot['retrievals']])
        self.assertEqual([x['vector'] for x in promoted['units']],[x['vector'] for x in snapshot['units']])
        self.assertGreater(max(proof['learned_raw_importance'].values()),50)
        self.assertEqual(len(promoted['feedback']),2)

    def test_actual_completion_prevents_false_earlier_knowledge_and_allows_disclosed_replay(self):
        bank,knowledge,age=self.fixture();result=self.promote(bank.to_dict(),knowledge,age)
        restored=CounterfactualMemoryBank.from_dict(result['bank'],namespace=Namespace('moira2-top1','live','experiment'),session_age=age)
        early=restored.retrieve('q',as_of=knowledge,source_cutoff=datetime(2026,11,2,9,30,tzinfo=NY),top_k=1,retrieval_id='early',query_embedding=[1.,0.])
        self.assertFalse(early.hits)
        later=restored.retrieve('q',as_of=knowledge+timedelta(minutes=20),source_cutoff=datetime(2026,11,2,9,30,tzinfo=NY),top_k=1,retrieval_id='replay',query_embedding=[1.,0.])
        self.assertEqual(len(later.hits),1)

    def test_wrong_variant_future_source_and_score_tamper_rejected(self):
        bank,knowledge,age=self.fixture();snapshot=bank.to_dict()
        for kind in ('variant','future','score','size'):
            bad=copy.deepcopy(snapshot)
            if kind=='variant':bad['namespace']['version']='moira2-top5'
            if kind=='future':bad['units'][0]['unit']['exit_at']='2026-11-02T11:30:00-05:00'
            if kind=='score':bad['retrievals'][0]['hits'][0]['raw_importance']+=18
            if kind=='size':bad['units'].pop()
            with self.assertRaises(ValueError):self.promote(bad,knowledge,age)


if __name__=='__main__':unittest.main()
