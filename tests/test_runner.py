"""Failure-path integration with real accounting and fake external transports."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,PropertyMock,patch

from live_testing import runner
from live_testing.accounting import new_book,record_decision,record_valuation
from live_testing.budget import BudgetExceeded
from live_testing.calendar import DEFAULT_CALENDAR
from live_testing.registry import MODEL_IDS,TICKERS
from live_testing.storage import read_json


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.at=DEFAULT_CALENDAR.release_nodes[2];self.instant=self.at+timedelta(minutes=1)
        self.values={ticker:100. for ticker in TICKERS};self.values['NVDA']=110.
        self.rows=[{'ticker':ticker,'price':value,'timestamp':self.at.isoformat()} for ticker,value in self.values.items()]
        self.runtime=runner.Runtime.__new__(runner.Runtime);self.runtime.root=self.root;self.runtime.config={}
        self.runtime.book=new_book(MODEL_IDS,TICKERS)
        self.runtime.state={'published':{},'completed_events':[],'feedback_applied':{m:[] for m in MODEL_IDS},
                            'model_attempts':{},'errors':{}}
        self.runtime._gateway=SimpleNamespace(scoped=Mock())
        self.runtime.feed=Mock(return_value={'verified':True})
        self.runtime.decide=Mock(side_effect=AssertionError('No real/model inference permitted in this fixture'))
        self.data={'timestamp':self.at.isoformat(),'prices':self.values,'price_rows':self.rows,
            'observed_at':self.instant.isoformat(),'news_rows':[],'coverage':{},'input_sha256':'a'*64}
        instant=self.instant
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)
        self.clock=Clock

    def action(self):return {'stock1':'NVDA','stock2':'GOOG','action':'LS'}
    def seed_current(self,models):
        for model in models:record_decision(self.runtime.book,model,self.at,self.action(),generated_at=self.instant,input_asof=self.at)
    def seed_first_interval(self):
        first=DEFAULT_CALENDAR.release_nodes[0];entry=DEFAULT_CALENDAR.action_window(first).entry_at
        for model in MODEL_IDS:record_decision(self.runtime.book,model,first,self.action(),generated_at=first+timedelta(minutes=1),input_asof=first)
        record_valuation(self.runtime.book,entry,{t:100. for t in TICKERS},observed_at=entry)
    def release(self,**extra):
        with patch.object(runner,'datetime',self.clock),patch.object(runner,'now',return_value=self.instant.isoformat()), \
                patch.object(runner,'prices',return_value=(self.values,self.rows)), \
                patch.object(runner,'collect_release',return_value=self.data), \
                patch.object(runner,'publish',return_value={'published_at':self.instant.isoformat()}):
            return self.runtime.release(self.at,**extra)

    def test_budget_failure_still_freezes_exit_prices_and_settles_existing_actions(self):
        self.seed_first_interval()
        with patch.object(runner,'datetime',self.clock),patch.object(runner,'now',return_value=self.instant.isoformat()), \
                patch.object(runner,'prices',return_value=(self.values,self.rows)), \
                patch.object(runner,'collect_release',side_effect=BudgetExceeded('Cap reached')) as collect:
            result=self.runtime.release(self.at)
        self.assertEqual(result['status'],'pending');self.assertFalse(result['private_data_verified'])
        self.assertIn(self.at.isoformat(),self.runtime.book['valuations'])
        self.assertTrue(all(len(a['history'])==1 for a in self.runtime.book['accounts'].values()))
        self.assertTrue(all(abs(a['equity']-1.05)<1e-12 for a in self.runtime.book['accounts'].values()))
        self.assertEqual(collect.call_args.kwargs['valuation']['prices'],self.values)
        self.runtime.decide.assert_not_called();self.runtime.feed.assert_called_once()

    def test_untrained_v2_is_pending_before_any_paid_attempt_is_recorded(self):
        self.seed_current(('codex','finmem','moira'));self.runtime.feedback=Mock(return_value={})
        result=self.release();self.assertEqual(result['status'],'pending')
        self.assertEqual(set(result['pending_models']),{'moira2-top1','moira2-top3','moira2-top5'})
        self.assertEqual(self.runtime.state['model_attempts'],{})
        self.runtime.decide.assert_not_called();self.runtime._gateway.scoped.assert_not_called()

    def test_killed_in_progress_codex_attempt_cannot_be_reissued(self):
        self.seed_current(tuple(model for model in MODEL_IDS if model!='codex'));self.runtime.feedback=Mock(return_value={})
        key='codex:'+self.at.isoformat();original={'status':'in_progress','input_sha256':'b'*64}
        self.runtime.state['model_attempts'][key]=dict(original)
        result=self.release();self.assertEqual(result['pending_models']['codex'],'prior_ambiguous_attempt_requires_audit')
        self.assertEqual(self.runtime.state['model_attempts'][key],original);self.runtime.decide.assert_not_called()

    def test_feedback_embedding_failure_does_not_undo_accounting_or_allow_new_model_decision(self):
        self.seed_first_interval();self.seed_current(tuple(model for model in MODEL_IDS if model!='moira'))
        first=DEFAULT_CALENDAR.release_nodes[0];record={'release_at':first.isoformat()}
        def feedback(_book,model,*args,**kwargs):return [record] if model=='moira' else []
        with patch.object(runner,'matured_feedback',side_effect=feedback), \
                patch('live_testing.moira_native.apply_feedback',side_effect=BudgetExceeded('No embedding budget')):
            result=self.release()
        self.assertEqual(result['pending_models']['moira'],'BudgetExceeded')
        self.assertTrue(all(len(a['history'])==1 for a in self.runtime.book['accounts'].values()))
        self.assertEqual(self.runtime.state['feedback_applied']['moira'],[]);self.runtime.decide.assert_not_called()

    def test_failed_website_deployment_recovers_exact_projection_outside_node_window_without_inputs_or_models(self):
        self.runtime.feed=runner.Runtime.feed.__get__(self.runtime)
        document={'schema':'synthetic-safe-aggregate','all_time':{'completed_rounds':0,'rows':[],'series':[]}}
        with patch.object(runner,'build_feed',return_value=document),patch.object(runner,'publish_feed',side_effect=RuntimeError('Pages pending')):
            with self.assertRaisesRegex(RuntimeError,'Pages pending'):self.runtime.feed()
        self.assertTrue(read_json(self.root/'runtime.json')['website_pending'])
        self.assertEqual(read_json(self.root/'website-feed-pending.json'),document)
        late=self.at+timedelta(hours=1,minutes=30)
        class LateClock(datetime):
            @classmethod
            def now(cls,tz=None):return late.astimezone(tz) if tz else late.replace(tzinfo=None)
        with patch.object(runner,'datetime',LateClock),patch.object(runner,'Runtime',return_value=self.runtime), \
                patch.object(runner,'publish_feed',return_value={'verified':True}) as publish, \
                patch.object(runner,'prices') as prices,patch.object(runner,'collect_release') as collect, \
                patch.object(type(self.runtime),'gateway',new_callable=PropertyMock,side_effect=AssertionError('No API key/client should be opened')):
            result=runner.dispatch(self.root)
        self.assertEqual(result,{'status':'website_recovered','website_verified':True})
        publish.assert_called_once_with(document,self.root);prices.assert_not_called();collect.assert_not_called()
        self.assertFalse(read_json(self.root/'runtime.json')['website_pending'])


if __name__=='__main__':unittest.main()
