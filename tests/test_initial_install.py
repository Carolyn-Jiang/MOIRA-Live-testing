import copy
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from live_testing import initial_install as installer
from live_testing.accounting import new_book
from live_testing.calendar import NY,START_AT
from live_testing.memory import CounterfactualMemoryBank,MemoryUnit,Namespace
from live_testing.registry import TICKERS,MODEL_IDS,TOP_K
from live_testing.storage import write_private,read_json,file_hash
from live_testing.moira_native import _unit,session_age


def bundle_fixture():
    dates=[datetime(2026,10,d,h,30,tzinfo=NY) for d in range(19,24) for h in ((9,10,11,12,14) if d==23 else (9,10,11,12,14,15))]
    grid=dates+[datetime(2026,10,23,15,30,tzinfo=NY),datetime(2026,10,23,16,tzinfo=NY)]
    prices=[{t:100.+i for t in TICKERS} for i in range(31)]
    obs=[{'round_id':at.strftime('%Y-%m-%d_%H:%M'),'timestamp':at.isoformat(),'prices':prices[i],
          'news':{t:['Synthetic historical article.'] for t in TICKERS}} for i,at in enumerate(dates)]
    intervals={row['round_id']:{'entry_timestamp':grid[i+1].isoformat(),'exit_timestamp':grid[i+2].isoformat(),
        'entry_prices':prices[i+1],'exit_prices':prices[i+2]} for i,row in enumerate(obs)}
    return {'observations':obs,'intervals':intervals,'candidates':[{}]*5,'embedding_dimension':2}


def tree(directory):return {str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*') if p.is_file()}


class InitialInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);base=Path(self.temp.name).resolve()
        self.root=base/'runtime';self.stage=base/'trained';self.root.mkdir()
        book=new_book(MODEL_IDS,TICKERS);write_private(self.root/'book.json',book)
        write_private(self.root/'runtime.json',{'model_attempts':{},'existing_original_state':'must remain exact'})
        for model in ['codex','finmem','moira']:
            write_private(self.root/'models'/model/'original.json',{'preserve':model})
        for model in TOP_K:
            write_private(self.root/'models'/model/'prepared.json',{'seed_only':model})
            write_private(self.root/'models'/model/'live-state.json',{'active_pair':None,'decisions':{},'attempts':{},'seed_bank':29})
        bundle=bundle_fixture();daily=self.root/'seed/daily.bin';daily.parent.mkdir();daily.write_bytes(b'synthetic frozen reference')
        bundle.update(daily_path=str(daily),daily_sha256=file_hash(daily),source_files=[])
        for i,row in enumerate(bundle['observations']):
            path=self.root/'seed/history'/f'{i}.json';write_private(path,row)
            bundle['source_files'].append({'path':str(path),'sha256':file_hash(path)})
        write_private(self.root/'seed/moira-bundle.json',bundle);write_private(self.root/'seed/moira-prepared.json',{'parent':'already trained common checkpoint'})
        proof=installer.source_proof(self.root);self.assertEqual(len(proof),32);budget={'calls':{}}
        self.finished={};self.bundle=bundle
        for index,model in enumerate(TOP_K):
            directory=self.stage/model;started=datetime(2026,11,2,9,31,tzinfo=NY);finished=started+timedelta(minutes=9+index);self.finished[model]=finished
            bank=CounterfactualMemoryBank(Namespace(model,'training','retained-episode'),session_age=session_age,
                embedding_id='explicit-synthetic',dimension=2)
            for i,row in enumerate(bundle['observations']):
                inter=bundle['intervals'][row['round_id']];unit=MemoryUnit('unit-'+str(i),row['round_id'],TICKERS[0],TICKERS[1],'F',
                    datetime.fromisoformat(row['timestamp']),datetime.fromisoformat(inter['entry_timestamp']),datetime.fromisoformat(inter['exit_timestamp']),
                    datetime.fromisoformat(inter['exit_timestamp']),0.,{'LS':0.,'SL':0.,'F':0.},'One complete synthetic reflection '+str(i),
                    tuple(TICKERS[:2]),{'model_id':model,'source_kind':'synthetic immutable action outcome'})
                bank.add_finalized(unit,as_of=finished,embedding=[1.,0.])
            exits=[datetime.fromisoformat(bundle['intervals'][r['round_id']]['exit_timestamp']) for r in bundle['observations']]
            bank.retrieve('synthetic query',as_of=finished,source_cutoff=exits[0],top_k=TOP_K[model],retrieval_id='q',query_embedding=[1.,0.])
            bank.apply_outcome_feedback('q',actual_return=1.,exit_at=exits[1],available_at=exits[1],as_of=finished)
            snapshot=bank.to_dict();seeds=[]
            for item in snapshot['units']:
                seed=copy.deepcopy(item['unit']);seed['available_at']=finished.isoformat();seed['provenance']['initial_training_model']=model;seeds.append(seed)
            text='Genuine synthetic variant policy '+model;checkpoint={'model':'gpt-4.1-mini','status':'ready',
                'prompt_memory':{'active_prompt':text},'pair_memory':[{'trained':model}],
                'frozen_prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
            report={'checkpoint':checkpoint,'completed_cycles':1,'max_cycles':8,'cycles':[{'cycle':1}],
                'finished_at':finished.isoformat(),'source_hashes_before':proof,'source_hashes_after':proof}
            prepared={**copy.deepcopy(checkpoint),'schema':'moira.initial-rri-checkpoint/1','model_id':model,
                'parent_checkpoint_sha256':file_hash(self.root/'seed/moira-prepared.json'),'finished_at':finished.isoformat(),
                'initial_rri_training':{'completed_cycles':1,'max_cycles':8,'memory_top_k':TOP_K[model],'tau_sessions':252,'feedback_step':18}}
            for name,value in [('prepared.json',prepared),('training-result.json',report),('seed-records.json',seeds),
                               ('retained-training-bank.json',snapshot),('frozen-training-inputs.json',bundle)]:write_private(directory/name,value)
            adapters={str(Path(installer.__file__).with_name(name)):file_hash(Path(installer.__file__).with_name(name)) for name in ['initial_training.py','moira_native.py','memory.py','training.py']}
            status={'status':'ready','model_id':model,'backbone':'gpt-4.1-mini','started_at':started.isoformat(),'finished_at':finished.isoformat(),
                'max_cycles':8,'completed_cycles':1,'checkpoint_sha256':file_hash(directory/'prepared.json'),
                'source_hashes_before':proof,'source_hashes_after':proof,'adapter_source_hashes':adapters,
                'parent_checkpoint_sha256':file_hash(self.root/'seed/moira-prepared.json'),'actual_observations':29,'actual_matured_labels':29,
                'source_window_start':bundle['observations'][0]['timestamp'],'source_window_end':'2026-10-23T16:00:00-04:00',
                'seed_unit_count':29,'seed_records_sha256':file_hash(directory/'seed-records.json'),
                'retained_training_bank_sha256':file_hash(directory/'retained-training-bank.json'),
                'actual_api_calls':1,'actual_completed_calls':1,'actual_failed_calls':0}
            write_private(directory/'initial-training-status.json',status)
            budget['calls'][model]={'model_id':model,'stage':'training','event_id':'initial-historical-rri',
                'status':'completed','requested_model':'gpt-4.1-mini'}
        write_private(self.root/'api-budget.json',budget)
        self.init_calls=[]

    def initialize(self,prepared,bundle,directory,seeds,gateway,*,model_id,scope,run_id):
        self.assertEqual((scope,run_id),('live','experiment'));self.assertEqual(len(seeds),29)
        for seed in seeds:self.assertEqual(len(gateway.embed(json.dumps(_unit(seed).to_prompt(),sort_keys=True))),2)
        self.init_calls.append(model_id)
        write_private(directory/'prepared.json',prepared);write_private(directory/'training-inputs.json',bundle)
        live={'active_pair':None,'decisions':{},'attempts':{},'direction_memory':[{'actual_pnl':s['actual_return']} for s in seeds],
              'prepared_prompt_sha256':prepared['frozen_prompt_sha256'],'memory_bank':{'initial':'to be replaced'}}
        write_private(directory/'live-state.json',live);return live

    def install(self):
        with patch('live_testing.moira_native.initialize',side_effect=self.initialize):
            return installer.install_initial_training(self.root,self.stage)

    def test_validate_all_real_variant_lineage_and_32_source_hashes(self):
        bundle,results,proof=installer.validate_staged_results(self.root,self.stage)
        self.assertEqual(len(proof),32);self.assertEqual(set(results),set(TOP_K));self.assertEqual(bundle,self.bundle)

    def test_explicit_mixed_recovery_directories_keep_all_sources_exact(self):
        resumed=self.stage.with_name('resumed')
        resumed.mkdir()
        mapping={model:self.stage/model for model in TOP_K}
        for model in ['moira2-top1','moira2-top3']:
            (self.stage/model).rename(resumed/model);mapping[model]=resumed/model
        before={model:tree(directory) for model,directory in mapping.items()}
        with patch('live_testing.moira_native.initialize',side_effect=self.initialize):
            result=installer.install_initial_training(self.root,mapping)
        self.assertEqual(result['source_directories'],{model:str(directory) for model,directory in mapping.items()})
        self.assertEqual({model:tree(directory) for model,directory in mapping.items()},before)
        self.assertTrue(result['all_original_state_preserved'])

    def test_source_mapping_rejects_partial_duplicate_or_cross_variant_directories(self):
        mapping={model:self.stage/model for model in TOP_K}
        with self.assertRaises(ValueError):installer.validate_staged_results(self.root,{'moira2-top1':mapping['moira2-top1']})
        duplicate={model:mapping['moira2-top1'] for model in TOP_K}
        with self.assertRaises(ValueError):installer.validate_staged_results(self.root,duplicate)
        mapping['moira2-top1'],mapping['moira2-top3']=mapping['moira2-top3'],mapping['moira2-top1']
        with self.assertRaises(ValueError):installer.validate_staged_results(self.root,mapping)

    def test_three_model_atomic_install_preserves_originals_and_trained_weights(self):
        protected=installer._preserved(self.root);old={model:tree(self.root/'models'/model) for model in TOP_K};staged_before=tree(self.stage)
        result=self.install();self.assertEqual(result['status'],'ready');self.assertEqual(installer._preserved(self.root),protected)
        self.assertEqual(tree(self.stage),staged_before);self.assertEqual(set(self.init_calls),set(TOP_K))
        for model in TOP_K:
            directory=self.root/'models'/model;ready=read_json(directory/'initial-training-ready.json')
            self.assertEqual(ready['checkpoint_sha256'],file_hash(directory/'initial-training-checkpoint.json'))
            self.assertEqual((directory/'initial-training-checkpoint.json').read_bytes(),(self.stage/model/'prepared.json').read_bytes())
            live=read_json(directory/'live-state.json');self.assertIsNone(live['active_pair']);self.assertEqual(live['decisions'],{})
            bank=CounterfactualMemoryBank.from_dict(live['memory_bank'],namespace=Namespace(model,'live','experiment'),session_age=session_age)
            self.assertEqual(bank.importance_at('unit-0',max(self.finished.values()),source_cutoff=START_AT),68.)
            self.assertEqual(tree(Path(result['transaction'])/'before'/model),old[model])
        permission=read_json(self.root/'preparation-replay-authorization.json')
        self.assertEqual(datetime.fromisoformat(permission['training_completion_cutoff']),max(self.finished.values()))

    def test_complete_install_is_idempotent_without_model_helpers(self):
        result=self.install();before=tree(self.root/'models')
        with patch('live_testing.moira_native.initialize',side_effect=AssertionError('No reinitialization')):
            self.assertEqual(installer.install_initial_training(self.root,self.stage),result)
        self.assertEqual(tree(self.root/'models'),before)

    def test_any_unready_variant_prevents_any_directory_replacement(self):
        path=self.stage/'moira2-top3/initial-training-status.json';s=read_json(path);s['status']='training';write_private(path,s)
        before=tree(self.root/'models')
        with self.assertRaises(RuntimeError):self.install()
        self.assertEqual(tree(self.root/'models'),before);self.assertFalse((self.root/'initial-training-install.json').exists())

    def test_existing_variant_attempt_is_never_overwritten(self):
        runtime=read_json(self.root/'runtime.json');runtime['model_attempts']['moira2-top1:initial']={'status':'ambiguous'};write_private(self.root/'runtime.json',runtime)
        before=tree(self.root/'models')
        with self.assertRaises(ValueError):self.install()
        self.assertEqual(tree(self.root/'models'),before)

    def test_corrupt_staged_bank_or_source_hash_rejected(self):
        source=Path(self.bundle['source_files'][0]['path']);source.write_text('corrupted immutable source')
        with self.assertRaises(ValueError):installer.validate_staged_results(self.root,self.stage)

    def test_mid_promotion_rename_failure_restores_all_old_directories_and_state(self):
        before=tree(self.root/'models');protected=installer._preserved(self.root);original=Path.rename
        def fail(path,target):
            if path.parent.name=='staging' and path.name=='moira2-top3':raise OSError('Synthetic promotion failure')
            return original(path,target)
        with patch.object(Path,'rename',fail):
            with self.assertRaises(OSError):self.install()
        self.assertEqual(tree(self.root/'models'),before);self.assertEqual(installer._preserved(self.root),protected)
        self.assertEqual(read_json(self.root/'initial-training-install.json')['status'],'rolled_back')
        self.assertFalse((self.root/'preparation-replay-authorization.json').exists())

    def test_cached_vector_alias_never_calls_provider_and_preserves_causal_knowledge(self):
        model='moira2-top1';seeds=read_json(self.stage/model/'seed-records.json');bank=read_json(self.stage/model/'retained-training-bank.json')
        aliases=installer.CachedTrainingVectors(model,seeds,bank)
        with self.assertRaises(RuntimeError):aliases.embed('a new unembedded text')
        mutated=copy.deepcopy(seeds);mutated[0]['reflection']='Changed conceptual evidence'
        with self.assertRaises(ValueError):installer.CachedTrainingVectors(model,mutated,bank)
        self.install();live=read_json(self.root/'models'/model/'live-state.json')
        restored=CounterfactualMemoryBank.from_dict(live['memory_bank'],namespace=Namespace(model,'live','experiment'),session_age=session_age)
        before=restored.retrieve('q',as_of=START_AT,source_cutoff=START_AT,top_k=1,retrieval_id='before',query_embedding=[1.,0.])
        self.assertEqual(before.hits,())
        after=restored.retrieve('q',as_of=max(self.finished.values()),source_cutoff=START_AT,top_k=1,retrieval_id='after',query_embedding=[1.,0.])
        self.assertEqual(len(after.hits),1)


if __name__=='__main__':unittest.main()
