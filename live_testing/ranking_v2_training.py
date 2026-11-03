"""Explicit isolated training of the three quarterly attributed Moira variants.

The 29 frozen October 19–23 observations retain their next-two-node labels.
Artifacts remain private and staged. No live checkpoint, action, or website is
installed by this module. Every provider call uses the existing shared budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .initial_training import phase_two_dataset, source_proof, _retained_episode
from .memory_v2 import PROTOCOL_VERSION, SCHEMA, TAU_MARKET_HOURS, EMA
from .registry import TOP_K
from .storage import digest, file_hash, locked, now, read_json, write_private


def _sources(root, model):
    proof=source_proof(root)
    parent=root/'models'/model/'prepared.json'
    proof[str(parent)]=file_hash(parent)
    return proof


def _code_sources():
    folder=Path(__file__).parent
    return {str(folder/name):file_hash(folder/name) for name in (
        'ranking_v2_training.py','memory_v2.py','moira_v2_native.py',
        'training.py','initial_training.py','native_source.py','llm.py','budget.py',
        'moira_native.py','memory.py','calendar.py','registry.py','pair_lifecycle.py')}


def _checked_files(proof):
    if any(file_hash(path)!=value for path,value in proof.items()):
        raise RuntimeError('A frozen training source or implementation changed')


def train_model(root, output, model, *, authorization, max_cycles=8, resume=False):
    from .budget import Budget
    from .llm import Gateway, read_settings
    from .moira_v2_native import NativeTrainingAdapter, validate_state
    from .moira_native import native_types
    from .training import train_native, objective

    if model not in TOP_K or not isinstance(authorization,str) or not authorization.strip():
        raise ValueError('An explicitly authorized registered Moira 2.0 variant is required')
    if type(max_cycles)is not int or not 1<=max_cycles<=8:
        raise ValueError('Each variant has an independent maximum of eight cycles')
    root,output=Path(root).resolve(),Path(output).resolve()
    if root==output or root not in output.parents:
        raise ValueError('Training artifacts must be staged under the private runtime root')
    directory=output/model;directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    with locked(directory/'training.lock',nonblocking=True):
        status=read_json(directory/'status.json')
        if status is not None and status.get('status')=='ready':
            if (status.get('model_id')!=model or status.get('protocol')!=PROTOCOL_VERSION
                    or status.get('checkpoint_sha256')!=file_hash(directory/'prepared.json')
                    or status.get('retained_bank_sha256')!=file_hash(directory/'retained-training-bank.json')):
                raise RuntimeError('A completed version-specific training artifact changed')
            return status
        if status is not None and not resume:
            raise RuntimeError('Incomplete paid training requires explicit audited resumption')
        if status is None:
            at=datetime.now(timezone.utc);sources=_sources(root,model);code=_code_sources()
            bundle=read_json(root/'seed/moira-bundle.json')
            parent=read_json(root/'models'/model/'prepared.json')
            if parent.get('model')!='gpt-4.1-mini' or parent.get('status')!='ready':
                raise ValueError('The retained genuine native parent checkpoint is required')
            checkpoint=copy.deepcopy(parent)
            checkpoint['memory_policy_protocol']=PROTOCOL_VERSION
            run_id='ranking-v2-initial:'+model+':'+at.isoformat()
            status={'schema':'moira.ranking-v2-training/1','status':'training','model_id':model,
                'backbone':'gpt-4.1-mini','protocol':PROTOCOL_VERSION,'memory_schema':SCHEMA,
                'tau_market_hours':TAU_MARKET_HOURS,'importance_initial':.5,
                'ema':{name:list(value) for name,value in EMA.items()},
                'started_at':at.isoformat(),'training_at':at.isoformat(),'run_id':run_id,
                'event_id':run_id,'max_cycles':max_cycles,'authorization':authorization,
                'objective':'sample_sharpe_without_annualization','source_hashes_before':sources,
                'adapter_source_hashes':code,'parent_checkpoint_sha256':file_hash(root/'models'/model/'prepared.json'),
                'empty_training_bank_per_episode':True,'no_current_live_outcomes_or_company_context':True,
                'in_sample':True,'training_returns_enter_live_cr':False,'live_install_performed':False}
            write_private(directory/'status.json',status)
            write_private(directory/'parent-checkpoint.json',parent)
            write_private(directory/'starting-checkpoint.json',checkpoint)
            write_private(directory/'frozen-training-inputs.json',bundle)
            write_private(directory/'source-proof.json',{'files':sources,'code':code})
        else:
            if (status.get('protocol')!=PROTOCOL_VERSION or status.get('model_id')!=model
                    or status.get('max_cycles')!=max_cycles or status.get('authorization')!=authorization):
                raise ValueError('A pending training cannot change its authorization, policy, or limits')
            at=datetime.fromisoformat(status['training_at']);sources=status['source_hashes_before']
            code=status['adapter_source_hashes'];run_id=status['run_id']
            bundle=read_json(directory/'frozen-training-inputs.json')
            checkpoint=read_json(directory/'starting-checkpoint.json')
            _checked_files(sources);_checked_files(code)
            status.update(status='training',resumed_at=now())
            write_private(directory/'status.json',status)
        config=read_json(root/'config.json')
        os.environ['PAIR_TRADING_FIX_ROOT']=config['framework_root']
        os.environ['MOIRA_LIVE_STATE_DIR']=str(root)
        os.environ.setdefault('TIKTOKEN_CACHE_DIR',str(root.parent/'trading-local/tokenizer-cache'))
        native_types()
        dataset=phase_two_dataset(bundle,training_at=at)
        gateway=Gateway(read_settings(config['key_file'])['OPENAI_API_KEY'],
            Budget(root).scoped(model,'training',status['event_id']),root/'api-audit')
        try:
            adapter=NativeTrainingAdapter(dataset=dataset,bundle=bundle,directory=directory,
                gateway=gateway,model_id=model)
            report=train_native(adapter,dataset,checkpoint,version=model,run_id=run_id,max_cycles=max_cycles)
            episode,state,kind=_retained_episode(directory,model,run_id,report)
            if (len(state['decisions'])!=29 or len(state['feedback'])!=29
                    or state['memory_bank'].get('schema')!=SCHEMA):
                raise RuntimeError('The retained native v2 episode is incomplete or uses a legacy bank')
            for path in (directory/'episodes'/model).iterdir():
                native=read_json(path/'live-state.json')
                if native is not None:validate_state(native,model_id=model,directory=path,bundle=bundle)
            metrics=objective(state['feedback'][row.action_id]['step_return'] for row in dataset.observations)
            expected=report['cycles'][-1]['candidate_metrics' if kind=='candidate' else 'baseline_metrics']
            if metrics!=expected:raise RuntimeError('Retained metrics differ from the actual delayed outcomes')
            for cycle in report['cycles']:
                candidate=cycle.get('candidate_metrics')
                if candidate is not None:
                    before=cycle['baseline_metrics']['sharpe_ratio'];after=candidate['sharpe_ratio']
                    if bool(cycle['accepted'])!=(before is not None and after is not None and after>before):
                        raise RuntimeError('A training update did not follow strict sample-Sharpe improvement')
            _checked_files(sources);_checked_files(code)
            finished=now();trained=copy.deepcopy(report['checkpoint'])
            trained.update(schema='moira.quarter-ranking-checkpoint/2',model_id=model,
                model='gpt-4.1-mini',status='ready',memory_policy_protocol=PROTOCOL_VERSION,
                finished_at=finished,ranking_v2_training={
                    'protocol':PROTOCOL_VERSION,'memory_schema':SCHEMA,'tau_market_hours':TAU_MARKET_HOURS,
                    'objective':report['objective'],'completed_cycles':report['completed_cycles'],
                    'max_cycles':max_cycles,'accepted_updates':report['accepted_updates'],
                    'source_window_start':dataset.window_start.isoformat(),
                    'source_window_end':dataset.window_end.isoformat(),'actual_observations':29,
                    'actual_matured_labels':29,'retained_metrics':metrics,'retained_episode':str(episode),
                    'parent_checkpoint_sha256':status['parent_checkpoint_sha256'],'in_sample':True})
            report.update(finished_at=finished,retained_metrics=metrics,retained_episode=str(episode),
                retained_evaluation_kind=kind,artifact_directory=str(directory),memory_policy_protocol=PROTOCOL_VERSION,
                in_sample=True,source_hashes_before=sources,adapter_source_hashes=code)
            write_private(directory/'prepared.json',trained)
            write_private(directory/'retained-training-bank.json',state['memory_bank'])
            write_private(directory/'training-result.json',report)
            calls=[row for row in read_json(root/'api-budget.json')['calls'].values()
                if row.get('model_id')==model and row.get('event_id')==status['event_id']]
            status.update(status='ready',finished_at=finished,completed_cycles=report['completed_cycles'],
                accepted_updates=report['accepted_updates'],retained_metrics=metrics,
                last_candidate_metrics=report['cycles'][-1].get('candidate_metrics'),
                source_window_start=dataset.window_start.isoformat(),source_window_end=dataset.window_end.isoformat(),
                actual_observations=29,actual_matured_labels=29,
                checkpoint_sha256=file_hash(directory/'prepared.json'),
                retained_bank_sha256=file_hash(directory/'retained-training-bank.json'),
                actual_api_calls=len(calls),actual_completed_calls=sum(c['status']=='completed' for c in calls),
                actual_failed_calls=sum(c['status']=='failed' for c in calls),
                actual_cost_usd=sum(c.get('actual_cost_usd',0) for c in calls))
            write_private(directory/'status.json',status)
            return status
        except Exception as error:
            status.update(status='pending',failed_at=now(),error_type=type(error).__name__)
            write_private(directory/'status.json',status)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--authorization',required=True);parser.add_argument('--model',choices=['all',*TOP_K],default='all')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);models=list(TOP_K) if args.model=='all' else [args.model]
    failures=[]
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        jobs={pool.submit(train_model,args.root,args.output,model,authorization=args.authorization,
            max_cycles=8,resume=args.resume):model for model in models}
        for job in as_completed(jobs):
            model=jobs[job]
            try:
                value=job.result()
                print(json.dumps({key:value.get(key) for key in ('model_id','status','protocol','completed_cycles',
                    'accepted_updates','retained_metrics','actual_api_calls','actual_cost_usd','checkpoint_sha256')}),flush=True)
            except Exception as error:
                failures.append(model);print(json.dumps({'model_id':model,'status':'pending','error_type':type(error).__name__}),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
