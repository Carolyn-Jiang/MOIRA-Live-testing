"""Offline weekly v2 routing, immutable artifact pointers and rollback tests."""
import copy
from datetime import timedelta
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from live_testing import maintenance as maintenance,belief_review,ranking_v2_runtime
from live_testing.memory_v2 import PROTOCOL_VERSION
from live_testing.registry import TOP_K
from live_testing.storage import read_json,write_private,file_hash,digest
from live_testing.training import Observation,Execution
from live_testing.weekly_dataset import WeeklyDataset
from tests.test_maintenance import Runtime,initialize_models,publish,event,reviewed_checkpoint


class WeeklyV2RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.runtime=Runtime(self.root);initialize_models(self.root)
        self.event=event('2026-11-09',8,'training')
        for model in TOP_K:
            directory=self.root/'models'/model
            for name in ('prepared.json','live-state.json'):
                doc=read_json(directory/name);doc['memory_policy_protocol']=PROTOCOL_VERSION;doc['model_id']=model;write_private(directory/name,doc)
            result_path=self.root/'training/initial'/model/'training-result.json'
            write_private(result_path,self.valid_report(read_json(directory/'prepared.json')))
            write_private(directory/'memory-policy.json',{'protocol':PROTOCOL_VERSION,'status':'ready','model_id':model,
                'checkpoint_sha256':file_hash(directory/'prepared.json'),'training_result_sha256':file_hash(result_path),
                'training_result_path':str(result_path),
                'training_bank_sha256':'b'*64})
    def valid_report(self,checkpoint):
        return {'checkpoint':copy.deepcopy(checkpoint),'objective':'sample_sharpe_without_annualization',
            'completed_cycles':1,'max_cycles':8,'accepted_updates':1,
            'cycles':[{'accepted':True,'baseline_metrics':{'sharpe_ratio':.1},'candidate_metrics':{'sharpe_ratio':.2}}]}
    def expected(self):
        return {str(p):file_hash(p) for p in (self.root/'models').rglob('*.json')}
    def models_bytes(self):
        return {str(p):p.read_bytes() for p in (self.root/'models').rglob('*.json')}
    def results(self):
        result={}
        for model in maintenance.TRAINED_MODELS:
            learned=read_json(self.root/'models'/model/'prepared.json');text='Weekly learned '+model
            learned['prompt_memory']['active_prompt']=text;learned['frozen_prompt_sha256']=hashlib.sha256(text.encode()).hexdigest()
            learned['pair_memory']=[{'weekly':model}]
            directory=self.root/'training/2026-W46'/model
            report={**self.valid_report(learned),'training_at':self.event.at.isoformat(),
                'finished_at':(self.event.at+timedelta(minutes=20)).isoformat(),'artifact_directory':str(directory)}
            write_private(directory/'training-result.json',report);result[model]=report
        return result
    def test_install_updates_real_marker_paths_hashes_but_preserves_live_banks(self):
        before={m:read_json(self.root/'models'/m/'live-state.json')for m in maintenance.TRAINED_MODELS}
        results=self.results();source_hashes={m:file_hash(Path(r['artifact_directory'])/'training-result.json')for m,r in results.items()}
        maintenance.install_training_results(self.root,results,self.expected(),self.root/'backup')
        for model in maintenance.TRAINED_MODELS:
            after=read_json(self.root/'models'/model/'live-state.json')
            for key in set(before[model])-{'pair_memory','prepared_prompt_sha256'}:self.assertEqual(after[key],before[model][key])
            self.assertEqual(file_hash(Path(results[model]['artifact_directory'])/'training-result.json'),source_hashes[model])
            if model in TOP_K:
                marker=read_json(self.root/'models'/model/'memory-policy.json')
                self.assertEqual(marker['checkpoint_sha256'],file_hash(self.root/'models'/model/'prepared.json'))
                self.assertEqual(marker['training_result_sha256'],source_hashes[model])
                self.assertEqual(marker['training_result_path'],str(Path(results[model]['artifact_directory'])/'training-result.json'))
                self.assertTrue(marker['live_memory_bank_preserved']);self.assertEqual(marker['live_seed_training_bank_sha256'],'b'*64)
                self.assertEqual(len(marker['checkpoint_history']),1)
            else:self.assertFalse((self.root/'models'/model/'memory-policy.json').exists())
    def test_mid_marker_failure_restores_all_checkpoint_live_and_marker_bytes(self):
        results=self.results();before=self.models_bytes();original=maintenance.write_private
        def fail(path,value):
            if Path(path)==self.root/'models/moira2-top3/memory-policy.json':raise OSError('Synthetic marker installation failure')
            return original(path,value)
        with patch.object(maintenance,'write_private',side_effect=fail),self.assertRaises(OSError):
            maintenance.install_training_results(self.root,results,self.expected(),self.root/'backup')
        self.assertEqual(self.models_bytes(),before)
    def test_v2_requires_exact_persisted_artifact_and_prevents_protocol_downgrade(self):
        results=self.results();before=self.models_bytes();model='moira2-top1'
        results[model]['checkpoint'].pop('memory_policy_protocol')
        with self.assertRaises(ValueError):maintenance.install_training_results(self.root,results,self.expected(),self.root/'backup')
        self.assertEqual(self.models_bytes(),before)
        results=self.results();results[model]['artifact_directory']=str(self.root/'not-training')
        write_private(self.root/'not-training/training-result.json',results[model])
        with self.assertRaises(ValueError):maintenance.install_training_results(self.root,results,self.expected(),self.root/'backup2')
        self.assertEqual(self.models_bytes(),before)
        results=self.results();path=Path(results[model]['artifact_directory'])/'training-result.json';changed=read_json(path);changed['finished_at']='tampered';write_private(path,changed)
        with self.assertRaises(ValueError):maintenance.install_training_results(self.root,results,self.expected(),self.root/'backup3')
        self.assertEqual(self.models_bytes(),before)
    def test_full_weekly_pipeline_routes_original_legacy_and_three_variants_v2(self):
        for at in maintenance.DEFAULT_CALENDAR.release_nodes[:30]:publish(self.runtime,at)
        legacy=[];v2=[]
        def review(dataset,prepared,bundle,directory,gateway,model_id,**kwargs):
            return reviewed_checkpoint(directory,prepared)
        def train(destination):
            def run(dataset,prepared,bundle,directory,gateway,model,max_cycles):
                destination.append(model);self.assertEqual(max_cycles,8)
                self.assertEqual(prepared.get('memory_policy_protocol'),PROTOCOL_VERSION if model in TOP_K else None)
                return {**self.valid_report(prepared),'training_at':dataset.training_at.isoformat(),
                    'finished_at':(dataset.training_at+timedelta(minutes=20)).isoformat(),'artifact_directory':str(directory)}
            return run
        with (patch.object(maintenance,'now',return_value=self.event.at.isoformat()),
              patch('live_testing.belief_review.review_weekly_beliefs',side_effect=review),
              patch('live_testing.moira_native.train_week',side_effect=train(legacy)),
              patch('live_testing.moira_v2_native.train_week',side_effect=train(v2)),
              patch.object(maintenance,'_company_context',return_value={'generated_at':self.event.at.isoformat()})):
            result=maintenance.run_maintenance(self.runtime,self.event)
        self.assertEqual(result['status'],'completed');self.assertEqual(legacy,['moira']);self.assertEqual(set(v2),set(TOP_K))
    def test_missing_or_mismatched_active_v2_marker_never_falls_back_to_legacy(self):
        prepared=read_json(self.root/'models/moira2-top1/prepared.json')
        (self.root/'models/moira2-top1/memory-policy.json').unlink()
        with self.assertRaises((ValueError,RuntimeError)):maintenance._training_module(self.root,'moira2-top1',prepared)
        prepared.pop('memory_policy_protocol');write_private(self.root/'models/moira2-top1/prepared.json',prepared)
        with self.assertRaises(RuntimeError):maintenance._training_module(self.root,'moira2-top1',prepared)
    def test_bad_hash_pinned_report_blocks_all_review_and_native_calls(self):
        for at in maintenance.DEFAULT_CALENDAR.release_nodes[:30]:publish(self.runtime,at)
        model='moira2-top3';marker_path=self.root/'models'/model/'memory-policy.json'
        marker=read_json(marker_path);report_path=Path(marker['training_result_path'])
        report=read_json(report_path);report['objective']='cumulative_return';write_private(report_path,report)
        # Even a report whose file hash matches the marker must prove the right objective.
        marker['training_result_sha256']=file_hash(report_path);write_private(marker_path,marker)
        with (patch.object(maintenance,'now',return_value=self.event.at.isoformat()),
              patch('live_testing.ranking_v2_runtime.ready',wraps=ranking_v2_runtime.ready) as ready,
              patch('live_testing.belief_review.review_weekly_beliefs') as review,
              patch('live_testing.moira_native.train_week') as legacy,
              patch('live_testing.moira_v2_native.train_week') as v2):
            result=maintenance.run_maintenance(self.runtime,self.event)
        self.assertEqual(result['status'],'pending');self.assertEqual(result['reason'],'all_four_ready_checkpoints_required')
        self.assertEqual(result['failures'],{model:'RuntimeError'})
        self.assertEqual({call.args[1] for call in ready.call_args_list},set(TOP_K))
        review.assert_not_called();legacy.assert_not_called();v2.assert_not_called()
    def test_false_or_exception_strict_v2_gate_cannot_route_or_start_review(self):
        prepared=read_json(self.root/'models/moira2-top1/prepared.json')
        for kwargs in ({'return_value':False},{'side_effect':RuntimeError('Synthetic changed report')}):
            with patch('live_testing.ranking_v2_runtime.ready',**kwargs) as ready,self.assertRaises(RuntimeError):
                maintenance._training_module(self.root,'moira2-top1',prepared)
            ready.assert_called_once_with(self.root,'moira2-top1')
    def test_changed_actual_report_hash_is_rejected_before_routing(self):
        marker=read_json(self.root/'models/moira2-top1/memory-policy.json')
        report_path=Path(marker['training_result_path']);report=read_json(report_path)
        report['completed_cycles']=7;write_private(report_path,report)
        with self.assertRaises(RuntimeError):
            maintenance._training_module(self.root,'moira2-top1',read_json(self.root/'models/moira2-top1/prepared.json'))
    def test_original_and_genuine_old_variant_keep_legacy_route(self):
        from live_testing import moira_native
        with patch('live_testing.ranking_v2_runtime.ready') as ready:
            self.assertIs(maintenance._training_module(self.root,'moira',read_json(self.root/'models/moira/prepared.json')),moira_native)
        ready.assert_not_called()
        directory=self.root/'models/moira2-top1';(directory/'memory-policy.json').unlink()
        for name in ('prepared.json','live-state.json'):
            doc=read_json(directory/name);doc.pop('memory_policy_protocol');write_private(directory/name,doc)
        self.assertIs(maintenance._training_module(self.root,'moira2-top1',read_json(directory/'prepared.json')),moira_native)


class ReviewAdapterRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        nodes=maintenance.DEFAULT_CALENDAR.release_nodes[:4];prices=[{ticker:100.+i for ticker in maintenance.TICKERS}for i in range(4)]
        observations=[Observation(str(i),at,prices[i],{ticker:[]for ticker in prices[i]},at)for i,at in enumerate(nodes[:2])]
        labels=[Execution(str(i),nodes[i+1],nodes[i+2],nodes[i+2],prices[i+1],prices[i+2],{'source':'synthetic canonical'})for i in range(2)]
        self.dataset=WeeklyDataset.build(observations,labels,training_at=event('2026-11-09',8,'training').at)
        text='Frozen original policy';self.checkpoint={'model':'gpt-4.1-mini','status':'ready','pair_memory':[],
            'prompt_memory':{'active_prompt':text},'frozen_prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
        self.calls=[]
    def adapter(self,label):
        calls=self.calls
        class Adapter:
            def __init__(self,**kwargs):calls.append((label,kwargs['model_id']))
            def begin_episode(self,checkpoint,namespace):
                class Policy:
                    def decide(self,observation):return {'stock1':'NVDA','stock2':'GOOG','action':'LS'}
                    def on_feedback(self,feedback):pass
                    def checkpoint(self):return copy.deepcopy(checkpoint)
                return Policy()
        return Adapter
    def test_isolated_belief_episodes_route_by_checkpoint_protocol_and_cache_separately(self):
        from live_testing import moira_v2_native
        v2=copy.deepcopy(self.checkpoint);v2['memory_policy_protocol']=PROTOCOL_VERSION
        with (patch.object(belief_review,'NativeTrainingAdapter',self.adapter('legacy')),
              patch.object(moira_v2_native,'NativeTrainingAdapter',self.adapter('v2'))):
            legacy=belief_review._episode(self.dataset,self.checkpoint,{},self.root/'legacy',None,'moira','review','old')
            actual=belief_review._episode(self.dataset,v2,{},self.root/'v2',None,'moira2-top1','review','old')
            before=len(self.calls);cached=belief_review._episode(self.dataset,v2,{},self.root/'v2',None,'moira2-top1','review','old')
            self.assertEqual(actual,cached);self.assertEqual(len(self.calls),before)
            with self.assertRaises(RuntimeError):belief_review._episode(self.dataset,self.checkpoint,{},self.root/'v2',None,'moira2-top1','review','old')
        self.assertEqual(self.calls,[('legacy','moira'),('v2','moira2-top1')])
        self.assertEqual(legacy['memory_policy_protocol'],'legacy-native/1');self.assertEqual(actual['memory_policy_protocol'],PROTOCOL_VERSION)
        self.assertIn(PROTOCOL_VERSION,actual['native_namespace']['run_id'])
    def test_unknown_protocol_or_original_moira_v2_declaration_rejected_before_model(self):
        for model,protocol in [('moira',PROTOCOL_VERSION),('moira2-top1','unrecognized/3')]:
            checkpoint={**self.checkpoint,'memory_policy_protocol':protocol}
            with self.assertRaises(ValueError):belief_review._episode(self.dataset,checkpoint,{},self.root/'bad',None,model,'review','old')
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
