import copy
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from live_testing import maintenance as m
from live_testing.calendar import DEFAULT_CALENDAR as C,NY,ScheduledEvent
from live_testing.accounting import new_book,record_valuation
from live_testing.registry import MODEL_IDS,TICKERS
from live_testing.storage import write_private,read_json,file_hash,digest


def when(day,hour,minute=0):return datetime.fromisoformat(f'{day}T{hour:02d}:{minute:02d}:00').replace(tzinfo=NY)
def event(day,hour,kind,minute=0):return next(e for e in C.events() if e.kind==kind and e.at==when(day,hour,minute))


class Gateway:
    key='synthetic-test-key'
    def __init__(self,label=None):self.label=label;self.budget=object()
    def scoped(self,model,stage,event):return Gateway((model,stage,event))


class Runtime:
    def __init__(self,root):
        self.root=root;self.book=new_book(MODEL_IDS,TICKERS);self.state={'published':{},'completed_events':[],'errors':{}}
        self._gateway=Gateway();self.saved=self.fed=self.feedback_calls=0
    @property
    def gateway(self):return self._gateway
    def save(self):self.saved+=1
    def feed(self):self.fed+=1;return {'verified':True}
    def feedback(self,cutoff):self.feedback_calls+=1


def publish(runtime,at):
    prices={t:100. for t in TICKERS};observed=at+timedelta(seconds=10)
    record_valuation(runtime.book,at,prices,observed_at=observed)
    rows=[{'ticker':t,'article_id':f'{at.date()}-{at.hour}-{t}','published_at':(at-timedelta(minutes=1)).isoformat(),
           'summary':'Synthetic company evidence.','summary_generated_at':observed.isoformat(),'title':'Synthetic company article.'} for t in TICKERS]
    data={'timestamp':at.isoformat(),'observed_at':observed.isoformat(),'prices':prices,'news_rows':rows,'coverage':{}}
    write_private(runtime.root/'releases'/(_name(at)+'.json'),data)
    runtime.state['published'][at.isoformat()]={'published_at':(at+timedelta(seconds=20)).isoformat()}


def _name(at):return at.strftime('%Y-%m-%d_%H-%M_NY')


def reviewed_checkpoint(directory,checkpoint):
    result={'status':'complete','checkpoint':copy.deepcopy(checkpoint),'checkpoint_sha256':digest(checkpoint)}
    write_private(Path(directory)/'review-result.json',result)
    return result


def initialize_models(root):
    for model in m.TRAINED_MODELS:
        text='Old learned policy '+model
        prepared={'model':'gpt-4.1-mini','status':'ready','pair_memory':[{'old':model}],
                  'prompt_memory':{'active_prompt':text},'frozen_prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
        live={'active_pair':list(TICKERS[:2]),'pair_memory':[{'live':model}],
              'prepared_prompt_sha256':prepared['frozen_prompt_sha256'],'decisions':{'pending':{'fixed':'untouched'}},
              'direction_memory':[{'actual_pnl':None,'realized_pnl':None}],
              'memory_bank':{'whole_unit':'must survive'},'history':{'old':'retained'},'feedback':{},'attempts':{}}
        write_private(root/'models'/model/'prepared.json',prepared);write_private(root/'models'/model/'live-state.json',live)
    write_private(root/'seed/moira-bundle.json',{'candidates':[]})
    write_private(root/'company-context.json',{'old':'context remains until post-training summary'})


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.runtime=Runtime(self.root)

    def test_sunday_and_special_saturday_cache_never_access_gateway_or_trade(self):
        class CacheRuntime(Runtime):
            @property
            def gateway(self):raise AssertionError('Cache must not load a key or model gateway')
        for e,prior in [(event('2026-11-08',9,'cache',30),when('2026-11-06',15,30)),
                        (event('2026-12-26',9,'cache',30),when('2026-12-24',12,30))]:
            rt=CacheRuntime(self.root);rt.state['published']={prior.isoformat():{'published_at':prior.isoformat()}}
            collection={'schema':'moira.live-testing.articles/1','collected_at':e.at.isoformat(),
                'window_start':prior.isoformat(),'window_end':e.at.isoformat(),'articles':[{'published_at':(prior+timedelta(minutes=1)).isoformat(),'title':'Synthetic cached headline','uuid':'source-1','tickers':[TICKERS[0]]}],'coverage':{}}
            with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.news.collect_articles',return_value=collection) as fetch:
                before=copy.deepcopy(rt.book);result=m.run_maintenance(rt,e)
            self.assertEqual(result['model_calls'],0);self.assertEqual(rt.book,before);self.assertEqual(rt.fed,0)
            self.assertEqual(fetch.call_args.args[:2],(prior,e.at));self.assertTrue((self.root/'news/archives'/(_name(e.at)+'.json')).exists())

    def test_previous_week_training_has_28_matured_labels_and_two_pending(self):
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        dataset,sources=m.build_weekly_dataset(self.runtime,when('2026-11-09',8))
        self.assertEqual(len(dataset.observations),30);self.assertEqual(len(dataset.executions),28)
        self.assertEqual(len(dataset.pending_action_ids),2)
        self.assertTrue(all(x.exit_at<=dataset.training_at and x.available_at<=dataset.training_at for x in dataset.executions.values()))
        self.assertEqual(len(sources),30)

    def test_empty_or_holiday_week_never_claims_training_success(self):
        e=event('2026-11-09',8,'training')
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.moira_native.train_week') as train:
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'pending');train.assert_not_called();self.assertNotIn(e.id,self.runtime.state['completed_events'])
        skipped=ScheduledEvent('training',when('2027-01-18',8))
        self.assertEqual(m.run_maintenance(self.runtime,skipped)['status'],'skipped')

    def test_all_four_parallel_training_installs_only_learned_fields_then_context(self):
        initialize_models(self.root)
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        e=event('2026-11-09',8,'training');before={model:read_json(self.root/'models'/model/'live-state.json') for model in m.TRAINED_MODELS};barrier=threading.Barrier(4);completed=[]
        def train(dataset,prepared,bundle,directory,gateway,model,max_cycles):
            self.assertEqual(max_cycles,8);self.assertEqual(gateway.label[:2],(model,'training'));barrier.wait(timeout=3)
            text='New learned policy '+model;learned=copy.deepcopy(prepared)
            learned['prompt_memory']['active_prompt']=text;learned['frozen_prompt_sha256']=hashlib.sha256(text.encode()).hexdigest();learned['pair_memory']=[{'learned':model}]
            completed.append(model)
            return {'checkpoint':learned,'training_at':e.at.isoformat()}
        def context(*args):
            self.assertEqual(set(completed),set(m.TRAINED_MODELS))
            return {'as_of':e.at.isoformat(),'generated_at':e.at.isoformat(),'companies':{}}
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.moira_native.train_week',side_effect=train),patch.object(m,'_company_context',side_effect=context):
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'completed');self.assertEqual(result['pending_labels'],2)
        for model in m.TRAINED_MODELS:
            after=read_json(self.root/'models'/model/'live-state.json')
            for key in set(before[model])-{'prepared_prompt_sha256','pair_memory'}:self.assertEqual(before[model][key],after[key])
            self.assertEqual(after['pair_memory'],[{'learned':model}])
        self.assertEqual(self.runtime.fed,1);self.assertIn(e.id,self.runtime.state['completed_events'])

    def test_one_training_failure_leaves_all_live_files_unchanged(self):
        initialize_models(self.root)
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        before={str(path):path.read_bytes() for path in (self.root/'models').rglob('*.json')};e=event('2026-11-09',8,'training')
        def train(dataset,prepared,bundle,directory,gateway,model,max_cycles):
            if model=='moira2-top3':raise RuntimeError('Synthetic failure')
            return {'checkpoint':prepared,'training_at':e.at.isoformat()}
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.moira_native.train_week',side_effect=train),patch.object(m,'_company_context') as summary:
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'pending');summary.assert_not_called()
        self.assertEqual({str(path):path.read_bytes() for path in (self.root/'models').rglob('*.json')},before)

    def test_each_version_reviews_beliefs_before_training_from_reviewed_checkpoint(self):
        initialize_models(self.root)
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        e=event('2026-11-09',8,'training');reviewed=set();trained=set();guard=threading.Lock()
        def review(dataset,prepared,bundle,directory,gateway,model_id,*,live_state,live_decisions):
            self.assertEqual(gateway.label[:2],(model_id,'training'))
            self.assertEqual(live_state['pair_memory'],[{'live':model_id}])
            self.assertEqual(live_decisions,{})
            checkpoint=copy.deepcopy(prepared);text='Reviewed belief '+model_id
            checkpoint['prompt_memory']['active_prompt']=text
            checkpoint['frozen_prompt_sha256']=hashlib.sha256(text.encode()).hexdigest()
            with guard:reviewed.add(model_id)
            return reviewed_checkpoint(directory,checkpoint)
        def train(dataset,prepared,bundle,directory,gateway,model,max_cycles):
            with guard:self.assertIn(model,reviewed);trained.add(model)
            self.assertEqual(max_cycles,8)
            self.assertEqual(prepared['prompt_memory']['active_prompt'],'Reviewed belief '+model)
            return {'checkpoint':prepared,'training_at':e.at.isoformat()}
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.belief_review.review_weekly_beliefs',side_effect=review),patch('live_testing.moira_native.train_week',side_effect=train),patch.object(m,'_company_context',return_value={'generated_at':e.at.isoformat()}):
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(reviewed,set(m.TRAINED_MODELS));self.assertEqual(trained,reviewed)
        for model in reviewed:
            saved=read_json(self.root/'training'/'2026-W46'/model/'training-result.json')
            self.assertTrue(saved['belief_review']['completed_before_training'])

    def test_failed_review_blocks_its_training_and_all_live_installation(self):
        initialize_models(self.root)
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        e=event('2026-11-09',8,'training');trained=[]
        before={str(path):path.read_bytes() for path in (self.root/'models').rglob('*.json')}
        def review(dataset,prepared,bundle,directory,gateway,model_id,**kwargs):
            if model_id=='moira2-top3':raise RuntimeError('Synthetic ambiguous review')
            return reviewed_checkpoint(directory,prepared)
        def train(dataset,prepared,bundle,directory,gateway,model,max_cycles):
            trained.append(model);return {'checkpoint':prepared,'training_at':e.at.isoformat()}
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.belief_review.review_weekly_beliefs',side_effect=review),patch('live_testing.moira_native.train_week',side_effect=train),patch.object(m,'_company_context') as summary:
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'pending');self.assertNotIn('moira2-top3',trained)
        summary.assert_not_called()
        self.assertEqual({str(path):path.read_bytes() for path in (self.root/'models').rglob('*.json')},before)

    def test_holiday_monday_training_runs_tuesday_with_same_previous_week(self):
        initialize_models(self.root)
        for at in C.release_nodes:
            if at.date().isoformat()>='2027-01-11' and at.date().isoformat()<='2027-01-15':publish(self.runtime,at)
        e=event('2027-01-19',8,'training');seen=[]
        def review(dataset,prepared,bundle,directory,gateway,model_id,**kwargs):
            self.assertEqual(dataset.training_at,e.at)
            self.assertEqual(len(dataset.observations),30)
            self.assertTrue(all('2027-01-11'<=row.timestamp.date().isoformat()<='2027-01-15' for row in dataset.observations))
            seen.append(model_id);return reviewed_checkpoint(directory,prepared)
        def train(dataset,prepared,bundle,directory,gateway,model,max_cycles):
            return {'checkpoint':prepared,'training_at':e.at.isoformat()}
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.belief_review.review_weekly_beliefs',side_effect=review),patch('live_testing.moira_native.train_week',side_effect=train),patch.object(m,'_company_context',return_value={'generated_at':e.at.isoformat()}):
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'completed');self.assertEqual(set(seen),set(m.TRAINED_MODELS))
        self.assertEqual(result['matured_labels'],28);self.assertEqual(result['pending_labels'],2)

    def test_unpersisted_belief_review_cannot_start_any_native_training(self):
        initialize_models(self.root)
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        e=event('2026-11-09',8,'training')
        def review(dataset,prepared,*args,**kwargs):
            return {'status':'complete','checkpoint':prepared,'checkpoint_sha256':digest(prepared)}
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.belief_review.review_weekly_beliefs',side_effect=review),patch('live_testing.moira_native.train_week') as train,patch.object(m,'_company_context') as summary:
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'pending');train.assert_not_called();summary.assert_not_called()

    def test_saved_training_without_completed_review_proof_is_rejected(self):
        initialize_models(self.root)
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        e=event('2026-11-09',8,'training')
        for model in m.TRAINED_MODELS:
            write_private(self.root/'training'/'2026-W46'/model/'training-result.json',
                {'training_at':e.at.isoformat(),'checkpoint':read_json(self.root/'models'/model/'prepared.json')})
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.belief_review.review_weekly_beliefs') as review,patch('live_testing.moira_native.train_week') as train,patch.object(m,'_company_context') as summary:
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'pending');review.assert_not_called();train.assert_not_called();summary.assert_not_called()

    def test_halfday_friday_reselection_has_no_trade_and_retains_old_epoch(self):
        e=event('2026-11-27',16,'pair_reselection');publish(self.runtime,when('2026-11-27',12,30));old=self.root/'models/finmem/epochs/old'
        write_private(old/'prepared.json',{'status':'ready','pairs':{}});write_private(old/'live-state.json',{'decisions':{'pending':'unchanged'}})
        write_private(self.root/'models/finmem/active-epoch.json',{'directory':str(old)});write_private(self.root/'seed/finmem-bundle.json',{'candidates':[1,2,3,4,5]})
        before=copy.deepcopy(self.runtime.book);old_bytes=(old/'live-state.json').read_bytes()
        with patch.object(m,'now',return_value=e.at.isoformat()),patch('live_testing.finmem_native.select_pair',return_value=(list(TICKERS[:2]),{'generated_at':e.at.isoformat()})) as choose,patch('live_testing.finmem_native.cloned_live_checkpoint') as clone,patch('live_testing.news.collect_articles') as collect:
            result=m.run_maintenance(self.runtime,e)
        self.assertEqual(result['status'],'completed');self.assertEqual(self.runtime.book,before)
        self.assertEqual((old/'live-state.json').read_bytes(),old_bytes);collect.assert_not_called();clone.assert_called_once()
        obs=choose.call_args.args[1];self.assertEqual(obs['timestamp'],when('2026-11-27',12,30).isoformat());self.assertIn('weekly_reselection_context',obs);self.assertEqual(obs['weekly_reselection_context']['known_labels'],0)
        new=Path(read_json(self.root/'models/finmem/active-epoch.json')['directory']);self.assertNotEqual(new,old)
        self.assertEqual(read_json(new/'live-state.json')['decisions'],{});self.assertFalse(result['warmup_retrained'])

    def test_future_maintenance_rejected_before_model_or_collection(self):
        e=event('2026-11-09',8,'training')
        with patch.object(m,'now',return_value=(e.at-timedelta(seconds=1)).isoformat()),patch('live_testing.moira_native.train_week') as train:
            with self.assertRaises(ValueError):m.run_maintenance(self.runtime,e)
        train.assert_not_called()

    def test_mid_install_write_failure_rolls_back_all_four_exact_bytes(self):
        initialize_models(self.root)
        paths=[self.root/'models'/model/name for model in m.TRAINED_MODELS for name in ['prepared.json','live-state.json']]
        before={str(path):path.read_bytes() for path in paths};expected={str(path):file_hash(path) for path in paths};results={}
        for model in m.TRAINED_MODELS:
            learned=read_json(self.root/'models'/model/'prepared.json');text='New synthetic '+model
            learned['prompt_memory']['active_prompt']=text;learned['frozen_prompt_sha256']=hashlib.sha256(text.encode()).hexdigest()
            results[model]={'checkpoint':learned}
        original=m.write_private;calls=[]
        def fail(path,value):
            calls.append(str(path))
            if len(calls)==2:raise OSError('Synthetic write failure')
            return original(path,value)
        with patch.object(m,'write_private',side_effect=fail):
            with self.assertRaises(OSError):m.install_training_results(self.root,results,expected,self.root/'rollback-backup')
        self.assertEqual({str(path):path.read_bytes() for path in paths},before)

    def test_historical_exit_with_late_quote_availability_stays_pending(self):
        for at in C.release_nodes[:30]:publish(self.runtime,at)
        cutoff=when('2026-11-09',8);exit_at=C.action_window(C.release_nodes[0]).exit_at
        self.runtime.book['valuations'][exit_at.isoformat()]['observed_at']=(cutoff+timedelta(hours=1)).isoformat()
        dataset,_=m.build_weekly_dataset(self.runtime,cutoff)
        self.assertEqual(len(dataset.executions),26)
        self.assertEqual(len(dataset.pending_action_ids),4)
        self.assertNotIn(dataset.observations[0].action_id,dataset.executions)


if __name__=='__main__':unittest.main()
