import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from live_testing import initial_training,llm,moira_native
from live_testing.resume_initial_training import (ReuseClock,enum_selection_schema,enum_direction_schema,
    failed_selection,gateway_class,resume,tree_hashes)
from live_testing.storage import read_json,write_private


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'original/moira2-top1';self.episode=self.source/'episodes/moira2-top1/frozen-episode'
        self.at='2026-10-21T15:30:00-04:00'
        self.monitor={'task_type':None,'system_prompt':'Exact real old monitoring prompt',
            'payload':{'current_observation':{'timestamp':self.at},'statistical_deterioration':{}},
            'response':'{"news_break_level":"HEALTHY","is_relationship_break_signal":false}',
            'started_at':'2026-11-02T16:58:30+00:00','generated_at':'2026-11-02T16:58:31+00:00'}
        self.bad={'task_type':'pair_selection','system_prompt':'Exact real old pair prompt',
            'payload':{'current_observation':{'timestamp':self.at},'top5_candidates':[{'pair':['NVDA','GOOG']},{'pair':['META','JPM']}]},
            'response':'{"selected_pair":"AMZN/TSLA","pair_selection_reason":"Synthetic rejected output."}',
            'started_at':'2026-11-02T16:58:32+00:00','generated_at':'2026-11-02T16:58:33+00:00'}
        write_private(self.source/'initial-training-status.json',{'status':'pending','model_id':'moira2-top1',
            'error':'Native inference failed; no synthetic fallback accepted','started_at':'2026-11-02T16:42:00+00:00','max_cycles':8})
        self.state={'decisions':{'old':{'action':{'stock1':'NVDA','stock2':'GOOG','action':'LS'}}},
            'attempts':{'old':{'status':'complete'},'failed':{'status':'in_progress'}},'feedback':{'old':{'step_return':.01}}}
        write_private(self.episode/'live-state.json',self.state)
        (self.episode/'model-calls.jsonl').write_text('\n'.join(json.dumps(r) for r in [self.monitor,self.bad])+'\n')

    def test_only_invalid_unaccepted_native_enum_can_resume(self):
        result=failed_selection(self.source);self.assertEqual(result[3],'failed');self.assertEqual(len(result[4]),1)
        state=copy.deepcopy(self.state);state['decisions']['failed']={'accepted':True};write_private(self.episode/'live-state.json',state)
        with self.assertRaises(ValueError):failed_selection(self.source)

    def test_action_failure_accepted_but_valid_response_not_retried(self):
        row=copy.deepcopy(self.bad);row['task_type']='daily_direction';row['response']=json.dumps({'action':'INVALID','direction_reason':'Rejected.'})
        (self.episode/'model-calls.jsonl').write_text('\n'.join(json.dumps(r) for r in [self.monitor,row])+'\n')
        self.assertEqual(failed_selection(self.source)[5]['task_type'],'daily_direction')
        row['response']=json.dumps({'action':'FLAT','direction_reason':'Valid.'})
        (self.episode/'model-calls.jsonl').write_text(json.dumps(row)+'\n')
        with self.assertRaises(ValueError):failed_selection(self.source)

    def test_reused_real_response_requires_exact_system_payload_and_preserves_time(self):
        clock=ReuseClock([self.monitor]);self.assertEqual(clock.timestamp(),self.monitor['started_at'])
        with self.assertRaises(ValueError):clock.consume('different',self.monitor['payload'],None)
        row=clock.consume(self.monitor['system_prompt'],self.monitor['payload'],None)
        self.assertEqual(row,self.monitor);self.assertEqual(clock.timestamp(),self.monitor['generated_at'])
        self.assertGreater(datetime.fromisoformat(clock.timestamp()),datetime.fromisoformat(self.monitor['generated_at']))

    def test_enum_guard_is_real_gateway_schema_not_posthoc_action_substitution(self):
        payload=self.bad['payload'];schema=enum_selection_schema(payload)
        self.assertEqual(schema['properties']['selected_pair']['enum'],['NVDA/GOOG','META/JPM'])
        self.assertEqual(enum_direction_schema()['properties']['action']['enum'],['LONG_SHORT','SHORT_LONG','FLAT'])
        cls=gateway_class(ReuseClock([]),self.root/'guard-audit.jsonl');gateway=cls('synthetic-no-key',SimpleNamespace(),self.root)
        returned={'selected_pair':'META/JPM','pair_selection_reason':'Actual selected candidate.'}
        with patch.object(llm.Gateway,'json',return_value=returned) as actual:
            value=json.loads(gateway.native_text(system_prompt='native',payload=payload,task_type='pair_selection'))
        self.assertEqual(value,returned);self.assertEqual(actual.call_args.kwargs['schema'],schema)
        self.assertEqual(actual.call_args.kwargs['model'],'gpt-4.1-mini')

    def test_new_copy_reuses_old_cache_and_clears_only_failed_attempt_without_new_api(self):
        before=tree_hashes(self.source);out=self.root/'resumed'
        def fake_train(root,output,model,max_cycles):
            directory=Path(output)/model;state=read_json(directory/'episodes/moira2-top1/frozen-episode/live-state.json')
            self.assertEqual(state['decisions'],self.state['decisions']);self.assertEqual(state['feedback'],self.state['feedback'])
            self.assertEqual(state['attempts'],{'old':{'status':'complete'}})
            self.assertEqual(initial_training.datetime.now(timezone.utc).isoformat(),'2026-11-02T16:42:00+00:00')
            gateway=llm.Gateway('synthetic-no-key',SimpleNamespace(),directory)
            self.assertEqual(moira_native.now(),self.monitor['started_at'])
            response=gateway.native_text(system_prompt=self.monitor['system_prompt'],payload=self.monitor['payload'],task_type=None)
            self.assertEqual(response,self.monitor['response']);self.assertEqual(moira_native.now(),self.monitor['generated_at'])
            write_private(directory/'prepared.json',{'status':'ready','synthetic_fixture':True})
            return {'model_id':model,'status':'ready','completed_cycles':3}
        with patch.object(initial_training,'source_proof',return_value={}),patch.object(initial_training,'train_model',fake_train):
            result=resume(self.root,self.source,out)
        self.assertEqual(result['status'],'ready');self.assertEqual(tree_hashes(self.source),before)
        proof=read_json(out/'moira2-top1/recovery-proof.json');self.assertEqual(proof['cleared_unaccepted_attempts'],1)
        self.assertTrue(proof['all_prior_accepted_decisions_retained']);self.assertEqual(proof['status'],'ready')


if __name__=='__main__':unittest.main()
