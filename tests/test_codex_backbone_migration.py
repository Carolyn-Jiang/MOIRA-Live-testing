import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from live_testing import codex_backbone_migration as migration
from live_testing.accounting import new_book, record_decision, record_valuation, settle_due, validate_book, pair_return
from live_testing.calendar import DEFAULT_CALENDAR as C, aware
from live_testing.data import observation, release_id
from live_testing.registry import MODEL_IDS, TICKERS
from live_testing.storage import digest, file_hash, read_json, write_private, append_private

NOW='2026-11-05T13:00:00+00:00'


class Gateway:
    def __init__(self,root,*,failure=None,callback=None):
        self.budget=SimpleNamespace(root=root);self.calls=[];self.scopes=[];self.failure=failure;self.callback=callback
    def scoped(self,*args):self.scopes.append(args);return self
    def json(self,**kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.callback:self.callback(len(self.calls))
        if len(self.calls)==self.failure:raise RuntimeError('Synthetic ambiguous transport')
        action={'stock1':'META','stock2':'JPM','action':'LS' if len(self.calls)%2 else 'SL'}
        append_private(self.budget.root/'api-audit/model-calls.jsonl',{'model_id':'codex','stage':'replay',
            'event_id':self.scopes[-1][2],'requested_model':kwargs['model'],'payload':kwargs['payload'],
            'payload_sha256':digest(kwargs['payload']),'instructions':kwargs['instructions'],
            'response':action,'started_at':NOW,'generated_at':NOW})
        return action


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.patch=patch.object(migration,'now',return_value=NOW);self.patch.start();self.addCleanup(self.patch.stop)
        self.book=new_book(MODEL_IDS,TICKERS);self.nodes=C.release_nodes[:18]
        self.runtime={'published':{},'feedback_applied':{m:[] for m in MODEL_IDS},'model_attempts':{},
            'completed_events':[],'website_pending':False,'errors':{}}
        self.audits=[];self.budget={'calls':{}}
        for i,at in enumerate(self.nodes):
            prices={ticker:100+(j+1)*i/100 for j,ticker in enumerate(TICKERS)}
            record_valuation(self.book,at,prices,observed_at=at+timedelta(seconds=1),provenance={'frozen':i})
            data={'timestamp':at.isoformat(),'prices':prices,'coverage':{},'news_rows':[]}
            write_private(self.root/'releases'/(release_id(at)+'.json'),data)
            write_private(self.root/'inputs'/(release_id(at)+'.json'),{'frozen_input':i})
            self.runtime['published'][at.isoformat()]={'published_at':(at+timedelta(seconds=10)).isoformat(),'folder':release_id(at),'file_sha256':{}}
            action={'stock1':'NVDA','stock2':'GOOG','action':'LS'};generated=at+timedelta(seconds=30)
            for model in MODEL_IDS:
                record_decision(self.book,model,at,action,generated_at=generated,input_asof=at,audit={'original':True})
                self.runtime['model_attempts'][model+':'+at.isoformat()]={'status':'complete','original':i}
            kwargs={'model':'gpt-5.3-codex','payload':{'observation':observation(data),'company_operations_context':{'historical':True}},
                'instructions':'Exact original synthetic point-in-time instructions.','schema':copy.deepcopy(migration.ACTION_SCHEMA),
                'name':'codex_pair_action','max_output_tokens':2048}
            body=migration._wire(kwargs,kwargs['model'])
            self.audits.append({'model_id':'codex','requested_model':'gpt-5.3-codex','payload':kwargs['payload'],
                'payload_sha256':digest(kwargs['payload']),'instructions':kwargs['instructions'],'response':action,
                'response_id':'synthetic-response-'+str(i),'generated_at':generated.isoformat()})
            identity='source-'+str(i);self.budget['calls'][identity]={'model_id':'codex','status':'completed','historical':True,
                'response_id':'synthetic-response-'+str(i),'requested_model':'gpt-5.3-codex'}
            write_private(self.root/'api-audit/http-requests'/(identity+'.json'),{'status':'completed','request_payload':body,
                'response_id':'synthetic-response-'+str(i)})
        settle_due(self.book,self.nodes[-1]+timedelta(minutes=1))
        for model in MODEL_IDS:
            self.runtime['feedback_applied'][model]=[r['release_at'] for r in self.book['accounts'][model]['history']]
            write_private(self.root/'models'/model/'unchanged-native.json',{'model':model,'immutable':True})
        write_private(self.root/'book.json',self.book);write_private(self.root/'runtime.json',self.runtime)
        write_private(self.root/'api-budget.json',self.budget);write_private(self.root/'config.json',{'no_key':True})
        self.write_audits()
        write_private(self.root/'company-context.json',{'as_of':'2026-11-01T09:30:00-05:00','generated_at':'2026-11-02T08:40:00-05:00',
            'companies':{ticker:{'recent_operations_summary':'Cached historical evidence.','source_ids':[]} for ticker in TICKERS}})
    def write_audits(self):
        path=self.root/'api-audit/model-calls.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in self.audits))
    def authorize(self,**kwargs):return migration.authorization(self.root,'Explicit synthetic user authorization.',**kwargs)
    def prepare(self,**kwargs):
        auth=self.authorize(**kwargs);return auth,migration.prepare(self.root,auth)

    def test_authorization_is_read_only_and_exact_wire_changes_only_model(self):
        before={p:file_hash(p) for p in self.root.rglob('*') if p.is_file()}
        auth=self.authorize()
        self.assertEqual(before,{p:file_hash(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(len(auth['plans']),18);self.assertEqual(auth['settled_prefix'],16)
        for i,plan in enumerate(auth['plans']):
            self.assertEqual(plan['basis'],'exact_accepted_request_model_only_change')
            old=read_json(self.root/'api-audit/http-requests'/('source-'+str(i)+'.json'))['request_payload']
            new=migration._wire(plan['request'],migration.MODEL)
            self.assertEqual({**new,'model':old['model']},old)

    def test_missing_first_two_request_holds_unless_explicit_fallback_is_authorized(self):
        self.audits=self.audits[2:];self.write_audits()
        with self.assertRaisesRegex(ValueError,'step 1'):self.authorize()
        auth=self.authorize(allow_fallback_first_two=True)
        self.assertTrue(all(p['basis']=='explicit_frozen_observation_compact_projection_change' for p in auth['plans'][:2]))
        self.assertTrue(all(p['basis']=='exact_accepted_request_model_only_change' for p in auth['plans'][2:]))
        for plan in auth['plans'][:2]:
            self.assertNotIn('round_id',plan['request']['payload']['current_observation'])
            self.assertEqual(set(plan['request']['payload']['current_observation']['prices']),set(TICKERS))

    def test_missing_later_request_never_falls_back_and_future_context_is_rejected(self):
        self.audits.pop(4);self.write_audits()
        with self.assertRaisesRegex(ValueError,'step 5'):self.authorize(allow_fallback_first_two=True)
        self.audits=self.audits[2:];self.write_audits()
        context=read_json(self.root/'company-context.json');context['generated_at']='2026-11-02T09:31:00-05:00'
        write_private(self.root/'company-context.json',context)
        with self.assertRaisesRegex(ValueError,'predate'):self.authorize(allow_fallback_first_two=True)

    def test_only_explicit_step_three_reconstruction_requires_original_source_and_completed_ledger(self):
        (self.root/'api-audit/http-requests/source-2.json').unlink()
        with self.assertRaisesRegex(ValueError,'step 3'):self.authorize()
        source=self.root/'original-runner.py'
        source.write_text("def original(gateway,payload):\n return gateway.json(model='gpt-5.3-codex',payload=payload,instructions='frozen',schema=ACTION_SCHEMA,name='codex_pair_action',max_output_tokens=2048)\n")
        options={'allow_gateway_reconstruction_steps':(3,),'gateway_source_path':source}
        auth=self.authorize(**options);plan=auth['plans'][2]
        self.assertEqual(plan['basis'],'reconstructed_gateway_request_no_original_wire')
        self.assertEqual(plan['source_implementation_sha256'],file_hash(source))
        self.assertEqual(plan['request']['payload'],self.audits[2]['payload']);self.assertEqual(plan['request']['instructions'],self.audits[2]['instructions'])
        with self.assertRaises(ValueError):self.authorize(allow_gateway_reconstruction_steps=(4,),gateway_source_path=source)
        ledger=read_json(self.root/'api-budget.json');ledger['calls']['source-2']['status']='failed';write_private(self.root/'api-budget.json',ledger)
        with self.assertRaisesRegex(ValueError,'completed provider'):self.authorize(**options)

    def test_changed_cached_response_cannot_be_claimed_as_a_real_provider_action(self):
        _,operation=self.prepare();gateway=Gateway(self.root,failure=3)
        with self.assertRaises(RuntimeError):migration.run(self.root,operation,gateway)
        record=read_json(operation/'calls/01.json');record['action']['action']='F';write_private(operation/'calls/01.json',record)
        resumed=Gateway(self.root)
        with self.assertRaisesRegex(RuntimeError,'actual accepted provider'):migration.run(self.root,operation,resumed)
        self.assertEqual(resumed.calls,[])

    def test_full_replay_installs_only_codex_preserving_prices_native_books_and_costs(self):
        auth,operation=self.prepare();before=copy.deepcopy(self.book)
        original_files={p:file_hash(p) for folder in ('models','inputs','releases') for p in (self.root/folder).rglob('*') if p.is_file()}
        gateway=Gateway(self.root);result=migration.run(self.root,operation,gateway)
        self.assertEqual(result['status'],'installed');self.assertEqual(len(gateway.calls),18)
        self.assertTrue(all(call['model']=='gpt-6.1-sol' for call in gateway.calls))
        self.assertTrue(all(scope[:2]==('codex','replay') for scope in gateway.scopes))
        book=read_json(self.root/'book.json');validate_book(book)
        self.assertEqual(book['valuations'],before['valuations'])
        for model in MODEL_IDS:
            if model!='codex':
                self.assertEqual(book['accounts'][model],before['accounts'][model]);self.assertEqual(book['decisions'][model],before['decisions'][model])
        self.assertEqual(len(book['accounts']['codex']['history']),16);self.assertEqual(len(book['decisions']['codex']),18)
        equity=1.
        for stamp,row in sorted(book['decisions']['codex'].items()):
            self.assertEqual(row['mode'],'retrospective_point_in_time');self.assertEqual(row['generated_at'],aware(NOW).isoformat())
            if row['step']<=16:
                ret=pair_return(row['action'],book['valuations'][row['entry_at']]['prices'],book['valuations'][row['exit_at']]['prices'])
                equity*=1+ret;self.assertAlmostEqual(equity,book['accounts']['codex']['history'][row['step']-1]['equity'])
        self.assertEqual(read_json(self.root/'api-budget.json'),self.budget)
        self.assertEqual(original_files,{p:file_hash(p) for p in original_files})
        self.assertEqual((operation/'archive/book.json').read_bytes(),json.dumps(before,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n')
        after=read_json(self.root/'runtime.json')
        for model in MODEL_IDS:
            if model!='codex':self.assertEqual(after['feedback_applied'][model],self.runtime['feedback_applied'][model])
        self.assertEqual(migration.run(self.root,operation,gateway),result);self.assertEqual(len(gateway.calls),18)

    def test_completed_paid_cache_survives_pre_book_failure_without_repeating_calls(self):
        _,operation=self.prepare();gateway=Gateway(self.root);original=migration.record_decision;counter=[0]
        def fail_once(*args,**kwargs):
            counter[0]+=1
            if counter[0]==6:raise RuntimeError('Synthetic crash after durable successful sixth response')
            return original(*args,**kwargs)
        with patch.object(migration,'record_decision',side_effect=fail_once):
            with self.assertRaises(RuntimeError):migration.run(self.root,operation,gateway)
        self.assertEqual(len(gateway.calls),6);self.assertEqual(read_json(self.root/'book.json'),self.book)
        resumed=Gateway(self.root);migration.run(self.root,operation,resumed)
        self.assertEqual(len(resumed.calls),12)

    def test_ambiguous_provider_attempt_is_never_automatically_reissued(self):
        _,operation=self.prepare();gateway=Gateway(self.root,failure=3)
        with self.assertRaises(RuntimeError):migration.run(self.root,operation,gateway)
        self.assertEqual(len(gateway.calls),3);self.assertEqual(read_json(self.root/'book.json'),self.book)
        resumed=Gateway(self.root)
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):migration.run(self.root,operation,resumed)
        self.assertEqual(resumed.calls,[])

    def test_install_exception_rolls_back_both_files_then_reuses_all_cached_responses(self):
        _,operation=self.prepare();gateway=Gateway(self.root);original=migration._bytes;failed=[False]
        def fail_once(path,value):
            if Path(path)==self.root/'runtime.json' and not failed[0]:
                failed[0]=True;raise OSError('Synthetic install failure')
            return original(path,value)
        with patch.object(migration,'_bytes',side_effect=fail_once):
            with self.assertRaises(OSError):migration.run(self.root,operation,gateway)
        self.assertEqual(read_json(self.root/'book.json'),self.book);self.assertEqual(read_json(self.root/'runtime.json'),self.runtime)
        resumed=Gateway(self.root);self.assertEqual(migration.run(self.root,operation,resumed)['status'],'installed');self.assertEqual(resumed.calls,[])

    def test_process_kill_mid_install_blocks_dispatch_and_recovers_before_pin_validation(self):
        _,operation=self.prepare();gateway=Gateway(self.root);original=migration._bytes
        def kill(path,value):
            if Path(path)==self.root/'runtime.json':raise SystemExit('Synthetic process kill')
            return original(path,value)
        with patch.object(migration,'_bytes',side_effect=kill):
            with self.assertRaises(SystemExit):migration.run(self.root,operation,gateway)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):migration.assert_installation_consistent(self.root)
        resumed=Gateway(self.root);self.assertEqual(migration.run(self.root,operation,resumed)['status'],'installed');self.assertEqual(resumed.calls,[])
        migration.assert_installation_consistent(self.root)

    def test_changed_sources_other_model_or_old_costs_block_before_paid_calls(self):
        for kind in ('source','other_model','cost'):
            with self.subTest(kind=kind):
                auth=self.authorize();operation=migration.prepare(self.root,auth)
                if kind=='source':
                    p=self.root/'releases'/(release_id(self.nodes[0])+'.json');old=p.read_bytes();p.write_bytes(old+b' ')
                elif kind=='other_model':
                    p=self.root/'models/finmem/unchanged-native.json';old=p.read_bytes();p.write_bytes(old+b' ')
                else:
                    p=self.root/'api-budget.json';old=p.read_bytes();value=read_json(p);value['calls']['source-0']['historical']=False;write_private(p,value)
                gateway=Gateway(self.root)
                with self.assertRaises(RuntimeError):migration.run(self.root,operation,gateway)
                self.assertEqual(gateway.calls,[]);p.write_bytes(old)

    def test_wrong_gateway_root_or_unauthorized_shape_never_executes(self):
        auth,operation=self.prepare()
        with self.assertRaises(ValueError):migration.run(self.root,operation,Gateway(self.root/'another'))
        auth['target_model']='another-model'
        with self.assertRaises(ValueError):migration.prepare(self.root,auth)


if __name__=='__main__':unittest.main()
