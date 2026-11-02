"""Synthetic single-model correction; no provider or production file access."""
import copy
from datetime import datetime,timedelta
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from live_testing import repair_moira_lifecycle as repair
from live_testing.accounting import new_book,record_decision,record_valuation,settle_due
from live_testing.calendar import DEFAULT_CALENDAR,NY,START_AT,aware
from live_testing.data import observation,release_id,round_id
from live_testing.registry import MODEL_IDS,TICKERS
from live_testing.storage import read_json,write_private,file_hash,digest


NOW=datetime(2026,11,2,17,tzinfo=NY).isoformat()


class Gateway:
    def __init__(self,root,label=None):self.root=root;self.label=label
    def scoped(self,*label):return Gateway(self.root,label)


class Native:
    def __init__(self,root,*,fail_after_accept=None,ambiguous=None):
        self.root=root;self.calls=[];self.seen=[];self.fail_after_accept=fail_after_accept;self.ambiguous=ambiguous
    def decide(self,obs,model,directory,bundle,gateway,*,asof,replay):
        at=aware(obs['timestamp']);rid=obs['round_id'];n=read_json(directory/'live-state.json')
        if rid in n['attempts']:raise RuntimeError('An ambiguous request cannot be repeated')
        assert gateway.root==self.root and gateway.label[:2]==('moira','replay') and replay
        assert max(aware(x) for x in n['history'])<=at
        assert all(aware(f['exit_at'])<=at for f in n['feedback'].values())
        if at.hour==10:
            assert n['direction_memory'][0]['realized_pnl'] is None
            assert 'actual_pnl' not in n['direction_memory'][0]
            assert 'cf_long_short' not in n['direction_memory'][0]
        self.seen.append((at.hour,len(n['feedback'])))
        n['attempts'][rid]={'status':'in_progress','started_at':NOW,'input_sha256':digest(obs)}
        write_private(directory/'live-state.json',n)
        self.calls.append(at.hour);budget=read_json(self.root/'api-budget.json')
        budget['calls']['new-'+str(len(self.calls))]={'status':'completed','actual_cost_usd':.01}
        write_private(self.root/'api-budget.json',budget)
        if self.ambiguous==at.hour:raise RuntimeError('Synthetic ambiguous provider failure')
        active=n['active_pair'];monitor=None;switched=False
        if active is None:active=list(TICKERS[:2])
        else:
            n['pending_break']=True;monitor={'lifecycle':'BREAK','pending_break':True}
            if at.hour==15:active=list(TICKERS[2:4]);switched=True;n['pending_break']=False
        action={'stock1':active[0],'stock2':active[1],'action':'LS','model':'gpt-4.1-mini','generated_at':NOW,'reason':'Synthetic true request.'}
        n['history'][at.isoformat()]={'price':obs['prices']};index=len(n['direction_memory'])
        n['direction_memory'].append({'selected_pair':'/'.join(active),'action':'LONG_SHORT','reason':'Synthetic true request.',
            'realized_pnl':None,'decision_index':index,'decision_date':at.isoformat(),'lifecycle_state':'BREAK',
            'original_action':None,'final_action':'LONG_SHORT'})
        n['active_pair']=active;n['decisions'][rid]={'action':action,'memory_index':index,'timestamp':at.isoformat(),
            'input_sha256':digest(obs),'monitor':monitor,'pair_switched':switched};n['attempts'][rid]['status']='complete'
        write_private(directory/'live-state.json',n)
        if self.fail_after_accept==at.hour:
            self.fail_after_accept=None;raise RuntimeError('Synthetic crash after durable provider acceptance')
        return action
    def apply_feedback(self,feedback,directory,bundle,gateway,model):
        n=read_json(directory/'live-state.json');rid=round_id(feedback['release_at'])
        if rid in n['feedback']:
            assert n['feedback'][rid]==feedback;return
        index=n['decisions'][rid]['memory_index'];n['direction_memory'][index].update(
            realized_pnl=feedback['step_return'],actual_pnl=feedback['step_return'],cf_long_short=feedback['step_return'])
        n['feedback'][rid]=copy.deepcopy(feedback);write_private(directory/'live-state.json',n)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()/'runtime';self.root.mkdir()
        root=self.root;self.times=DEFAULT_CALENDAR.release_nodes[:6];book=new_book(MODEL_IDS,TICKERS)
        runtime={'published':{},'completed_events':[],'feedback_applied':{m:[] for m in MODEL_IDS},
            'model_attempts':{},'errors':{},'website_pending':False}
        text='Initial frozen learned prompt';self.prepared={'model':'gpt-4.1-mini','status':'ready',
            'prompt_memory':{'active_prompt':text},'pair_memory':[{'historical':'learned only'}],
            'frozen_prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
        native={'history':{'2026-10-30T16:00:00-04:00':{'price':{t:100. for t in TICKERS}}},'decisions':{},
            'direction_memory':[],'feedback':{},'pair_memory':self.prepared['pair_memory']+[{'bad_later_break':1},{'bad_later_break':2}],
            'active_pair':None,'pending_break':False,'attempts':{}}
        for i,at in enumerate(self.times):
            prices={t:100.+i*(j+1)/10 for j,t in enumerate(TICKERS)}
            record_valuation(book,at,prices,observed_at=at+timedelta(minutes=1),provenance={'source':'synthetic exact'})
            data={'timestamp':at.isoformat(),'prices':prices,'price_rows':[], 'news_rows':[],'coverage':'partial'}
            write_private(root/'releases'/(release_id(at)+'.json'),data);write_private(root/'inputs'/(release_id(at)+'.json'),data)
            folder='data/'+release_id(at);hashes={}
            for name in ('price.parquet','news.parquet'):
                path=root/'news/hf-staging'/folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((name+str(i)).encode());hashes[name]=file_hash(path)
            runtime['published'][at.isoformat()]={'published_at':(at+timedelta(minutes=2)).isoformat(),'folder':folder,'file_sha256':hashes}
            runtime['completed_events'].append('release:'+at.isoformat())
            for m in MODEL_IDS:
                pair=TICKERS[:2] if m!='moira' or i==0 else TICKERS[2:4] if i<5 else TICKERS[4:6]
                action={'stock1':pair[0],'stock2':pair[1],'action':'LS'}
                d=record_decision(book,m,at,action,generated_at=at+timedelta(minutes=3),input_asof=at,audit={'original':True})
                runtime['model_attempts'][m+':'+at.isoformat()]={'status':'complete','started_at':d['generated_at']}
                if m=='moira':
                    record={**action,'generated_at':d['generated_at'],'model':'gpt-4.1-mini','reason':'Original accepted action'}
                    native['decisions'][round_id(at)]={'action':record,'memory_index':i,'timestamp':at.isoformat(),'monitor':None}
                    native['history'][at.isoformat()]={'price':prices}
                    native['direction_memory'].append({'selected_pair':'/'.join(pair),'action':'LONG_SHORT','reason':'Original accepted action',
                        'realized_pnl':99.,'decision_index':i,'decision_date':at.isoformat(),'lifecycle_state':'HEALTHY',
                        'original_action':None,'final_action':'LONG_SHORT','actual_pnl':99.,'cf_long_short':99.,'price_stock1':999.,'delta_stock1':999.})
                    native['active_pair']=list(pair)
        settle_due(book,aware(NOW))
        for m in MODEL_IDS:runtime['feedback_applied'][m]=[h['release_at'] for h in book['accounts'][m]['history']]
        for m in MODEL_IDS:
            write_private(root/'models'/m/'sentinel.json',{'model':m});(root/'models'/m/'runtime.lock').touch()
        source=copy.deepcopy(native);source['decisions']={k:v for k,v in list(source['decisions'].items())[:2]};source['direction_memory']=source['direction_memory'][:2]
        source['history']={k:v for k,v in source['history'].items() if aware(k)<=self.times[1]};self.copy=root/'source/initial-native-copy.json';write_private(self.copy,source)
        write_private(root/'models/moira/live-state.json',native);write_private(root/'models/moira/prepared.json',self.prepared)
        write_private(root/'models/moira/training-inputs.json',{'frozen':True});write_private(root/'seed/moira-bundle.json',{'candidates':[{}]*5})
        write_private(root/'company-context.json',{'dated':'known before first observation'});write_private(root/'config.json',{'only':'synthetic'})
        write_private(root/'bootstrap.json',{'source_files':{'old-native':{'copy':str(root/'models/moira/live-state.json'),'copy_sha256':file_hash(self.copy)}}})
        write_private(root/'book.json',book);write_private(root/'runtime.json',runtime);write_private(root/'api-budget.json',{'calls':{'old':{'actual_cost_usd':1.25,'status':'completed'}}})
        self.native=Native(root);self.gateway=Gateway(root)
    def authorize(self,*,preserve=True,replacement=None):
        auth=repair.authorization(self.root,original_native_copy=self.copy,statement='Explicit synthetic correction authorization.',
            preserve_first=preserve,replacement_prepared=replacement)
        path=self.root/'authorization.json';write_private(path,auth);return path
    def run_repair(self,path):
        with patch.object(repair,'now',return_value=NOW):return repair.repair(self.root,path,self.gateway,native=self.native)
    def test_restored_first_prefix_has_no_future_outcome_or_break_pollution(self):
        original=read_json(self.copy);accepted=read_json(self.root/'book.json')['decisions']['moira'][START_AT.isoformat()]
        state=repair.restored_prefix(original,self.prepared,accepted)
        self.assertEqual(state['pair_memory'],self.prepared['pair_memory']);self.assertEqual(len(state['decisions']),1)
        self.assertEqual(state['feedback'],{});self.assertFalse(state['pending_break'])
        self.assertIsNone(state['direction_memory'][0]['realized_pnl'])
        self.assertFalse({'actual_pnl','cf_long_short','price_stock1','delta_stock1'}&set(state['direction_memory'][0]))
        self.assertEqual(state['decisions'][round_id(START_AT)],original['decisions'][round_id(START_AT)])
    def test_five_replay_preserves_first_other_models_prices_and_old_costs(self):
        before=read_json(self.root/'book.json');other=repair._other_models(self.root);path=self.authorize();result=self.run_repair(path)
        after=read_json(self.root/'book.json');self.assertEqual(result['status'],'installed');self.assertEqual(result['replayed_decisions'],5)
        self.assertEqual(after['decisions']['moira'][START_AT.isoformat()],before['decisions']['moira'][START_AT.isoformat()])
        self.assertEqual(after['valuations'],before['valuations']);self.assertEqual(repair._other_models(self.root),other)
        self.assertEqual(self.native.seen,[(10,0),(11,1),(12,2),(14,3),(15,4)])
        self.assertEqual(len(after['accounts']['moira']['history']),4);self.assertEqual(len(self.native.calls),5)
        self.assertEqual(read_json(self.root/'api-budget.json')['calls']['old'],{'actual_cost_usd':1.25,'status':'completed'})
    def test_new_checkpoint_full_six_replay_has_no_inherited_first_action(self):
        old_calls=self.root/'models/moira/model-calls.jsonl';old_calls.write_text('{"synthetic_superseded_call":true}\n')
        original_calls=old_calls.read_bytes()
        new=copy.deepcopy(self.prepared);new['prompt_memory']['active_prompt']='New Sharpe-trained prompt';new['frozen_prompt_sha256']=hashlib.sha256(new['prompt_memory']['active_prompt'].encode()).hexdigest();new['pair_memory']=[{'new':'Sharpe-trained'}]
        path=self.root/'new-prepared.json';write_private(path,new)
        with self.assertRaises(ValueError):self.authorize(replacement=path)
        authorization=self.authorize(preserve=False,replacement=path);result=self.run_repair(authorization)
        self.assertEqual(result['replayed_decisions'],6);self.assertEqual(self.native.seen,[(9,0),(10,0),(11,1),(12,2),(14,3),(15,4)])
        book=read_json(self.root/'book.json');self.assertTrue(all(d['mode']=='retrospective_point_in_time' for d in book['decisions']['moira'].values()))
        self.assertEqual(read_json(self.root/'models/moira/prepared.json'),new)
        self.assertEqual(read_json(self.root/'models/moira/live-state.json')['prepared_sha256'],file_hash(path))
        self.assertEqual((Path(result['operation'])/'archive/moira/model-calls.jsonl').read_bytes(),original_calls)
        self.assertFalse((self.root/'models/moira/model-calls.jsonl').exists())
    def test_accepted_provider_response_survives_crash_and_resume_without_repeat(self):
        self.native.fail_after_accept=12;path=self.authorize(preserve=False);before=file_hash(self.root/'book.json')
        with self.assertRaises(RuntimeError):self.run_repair(path)
        self.assertEqual(file_hash(self.root/'book.json'),before)
        result=self.run_repair(path);self.assertEqual(result['status'],'installed');self.assertEqual(self.native.calls,[9,10,11,12,14,15])
        again=self.run_repair(path);self.assertEqual(again,result);self.assertEqual(len(self.native.calls),6)
    def test_ambiguous_request_is_not_cleared_or_repeated(self):
        self.native.ambiguous=9;path=self.authorize(preserve=False);before=file_hash(self.root/'book.json')
        for _ in range(2):
            with self.assertRaises(RuntimeError):self.run_repair(path)
        self.assertEqual(self.native.calls,[9]);self.assertEqual(file_hash(self.root/'book.json'),before)
    def test_changed_pin_or_wrong_original_copy_refuses_before_any_request(self):
        path=self.authorize();write_private(self.root/'company-context.json',{'changed':True})
        with self.assertRaises(ValueError):self.run_repair(path)
        self.assertEqual(self.native.calls,[])
        self.copy.write_text('{}')
        with self.assertRaises(ValueError):self.authorize()
    def test_first_day_repair_cannot_drop_a_later_genuine_decision(self):
        book=read_json(self.root/'book.json');at=DEFAULT_CALENDAR.release_nodes[6]
        record_decision(book,'moira',at,{'stock1':TICKERS[0],'stock2':TICKERS[1],'action':'F'},
            generated_at=at+timedelta(minutes=1),input_asof=at)
        write_private(self.root/'book.json',book);before=file_hash(self.root/'book.json')
        with self.assertRaises(ValueError):self.authorize(preserve=False)
        self.assertEqual(file_hash(self.root/'book.json'),before);self.assertEqual(self.native.calls,[])
    def test_atomic_install_failure_restores_old_model_book_runtime(self):
        path=self.authorize();old={name:(self.root/name).read_bytes() for name in ('book.json','runtime.json')};tree=repair._tree(self.root/'models/moira');original=repair.write_private
        def fail(target,value):
            if Path(target)==self.root/'runtime.json':raise OSError('Synthetic mid-install failure')
            return original(target,value)
        with patch.object(repair,'write_private',side_effect=fail),self.assertRaises(OSError):self.run_repair(path)
        self.assertEqual({name:(self.root/name).read_bytes() for name in old},old);self.assertEqual(repair._tree(self.root/'models/moira'),tree)
    def test_interrupted_commit_blocks_dispatch_and_restores_then_reuses_shadow(self):
        path=self.authorize();auth=read_json(path);operation=self.root/'repairs/moira-lifecycle'/digest(auth)[:24]
        self.native.fail_after_accept=15
        with self.assertRaises(RuntimeError):self.run_repair(path)
        journal=read_json(operation/'status.json');journal['status']='committing';write_private(operation/'status.json',journal)
        (self.root/'models/moira').rename(operation/'live-before');(operation/'shadow/moira').rename(self.root/'models/moira')
        write_private(self.root/'book.json',read_json(operation/'book.json'))
        with self.assertRaises(RuntimeError):repair.assert_installation_consistent(self.root)
        result=self.run_repair(path);self.assertEqual(result['status'],'installed');self.assertEqual(self.native.calls,[10,11,12,14,15])
        repair.assert_installation_consistent(self.root)
    def test_real_runtime_refuses_interrupted_install_before_config_or_gateway(self):
        from live_testing import runner
        operation=self.root/'repairs/moira-lifecycle/interrupt'
        write_private(operation/'status.json',{'status':'committing'})
        write_private(self.root/'repairs/moira-lifecycle/current.json',{'operation':str(operation)})
        with (patch.object(runner,'read_json',side_effect=AssertionError('Configuration/book must not load')),
              patch.object(runner,'create_guarded_gateway') as gateway):
            with self.assertRaisesRegex(RuntimeError,'interrupted single-model'):
                runner.Runtime(self.root)
            gateway.assert_not_called()


if __name__=='__main__':unittest.main()
