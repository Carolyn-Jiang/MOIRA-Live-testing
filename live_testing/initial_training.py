"""Explicit historical Moira 2.0 initialization, separate from future Mondays.

The user-selected October 19–23 archive has its own frozen execution grid.
It is never admitted through, or confused with, the new live calendar. Calls
retain their real generation times; all metrics are in-sample training metrics.
This module produces staged artifacts only and never installs a live model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass
from datetime import datetime, time, timezone
import json
import math
from pathlib import Path

from .calendar import NY, aware
from .memory import Namespace
from .registry import TICKERS, TOP_K
from .storage import digest, file_hash, locked, now, read_json, write_private
from .training import Observation, Execution, objective, train_native


@dataclass(frozen=True)
class HistoricalDataset:
    training_at: datetime
    observations: tuple[Observation, ...]
    executions: dict[str, Execution]
    pending_action_ids: tuple[str, ...]
    release_nodes: tuple[datetime, ...]
    valuation_nodes: tuple[datetime, ...]
    window_start: datetime
    window_end: datetime

    @classmethod
    def build(cls, observations, executions, *, training_at, release_nodes,
              valuation_nodes, window_start, window_end):
        cutoff, start, end = aware(training_at), aware(window_start), aware(window_end)
        releases = tuple(aware(value) for value in release_nodes)
        valuations = tuple(aware(value) for value in valuation_nodes)
        rows = tuple(copy.deepcopy(row) for row in observations)
        if not releases or start > releases[0] or releases[-1] > end or end > cutoff:
            raise ValueError('Explicit completed historical source window required')
        if tuple(sorted(set(releases))) != releases or tuple(row.timestamp for row in rows) != releases:
            raise ValueError('Observations must match the complete explicit historical release grid')
        if len({row.action_id for row in rows}) != len(rows): raise ValueError('Duplicate historical action id')
        grid = tuple(sorted(set(releases+valuations)))
        if len(grid) != len(releases)+len(valuations) or any(at <= releases[-1] or at > end for at in valuations):
            raise ValueError('Only actual distinct terminal historical valuations are permitted')
        ids = {row.action_id: row for row in rows}; prices_at = {}
        for row in rows:
            if (not row.action_id or aware(row.timestamp) > aware(row.available_at) or aware(row.available_at) > cutoff
                    or set(row.prices) != set(TICKERS) or set(row.news) != set(TICKERS)
                    or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in row.prices.values())
                    or any(not isinstance(v,list) or not all(isinstance(s,str) for s in v) for v in row.news.values())):
                raise ValueError('Complete causal historical ten-ticker evidence required')
            prices_at[aware(row.timestamp)] = row.prices
        known = {}
        for label in executions:
            if label.action_id not in ids: raise ValueError('Unknown historical outcome id')
            if label.action_id in known: raise ValueError('Duplicate historical outcome')
            if aware(label.exit_at) > cutoff or aware(label.available_at) > cutoff: continue
            i = grid.index(aware(ids[label.action_id].timestamp))
            if (i+2 >= len(grid) or aware(label.entry_at) != grid[i+1] or aware(label.exit_at) != grid[i+2]
                    or not aware(label.entry_at) < aware(label.exit_at) <= aware(label.available_at) <= cutoff
                    or not label.provenance):
                raise ValueError('Outcome must follow the immutable next-two-boundary historical execution grid')
            for at, values in ((aware(label.entry_at),label.entry_prices),(aware(label.exit_at),label.exit_prices)):
                if set(values) != set(TICKERS) or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in values.values()):
                    raise ValueError('Complete positive immutable historical valuations required')
                if at in prices_at and prices_at[at] != values: raise ValueError('Historical shared valuation changed')
                prices_at[at] = copy.deepcopy(values)
            known[label.action_id] = copy.deepcopy(label)
        return cls(cutoff,rows,known,tuple(row.action_id for row in rows if row.action_id not in known),
                   releases,valuations,start,end)


def phase_two_dataset(bundle, *, training_at):
    """The specifically authorized 29 real releases and two real final marks."""
    releases = tuple(datetime(2026,10,day,hour,30,tzinfo=NY)
        for day in range(19,24) for hour in ((9,10,11,12,14) if day==23 else (9,10,11,12,14,15)))
    valuations = (datetime(2026,10,23,15,30,tzinfo=NY),datetime(2026,10,23,16,tzinfo=NY))
    rows = tuple(Observation.from_dict(row) for row in bundle['observations'])
    labels = []
    for row in rows:
        value = bundle['intervals'][row.action_id]
        labels.append(Execution(row.action_id,aware(value['entry_timestamp']),aware(value['exit_timestamp']),
            aware(value['exit_timestamp']),copy.deepcopy(value['entry_prices']),copy.deepcopy(value['exit_prices']),
            {'source':'frozen organizer historical shared valuation', 'bundle_sha256':digest(bundle),
             'source_action_id':row.action_id}))
    dataset = HistoricalDataset.build(rows,labels,training_at=training_at,release_nodes=releases,
        valuation_nodes=valuations,window_start=releases[0],window_end=valuations[-1])
    if len(dataset.observations)!=29 or len(dataset.executions)!=29 or dataset.pending_action_ids:
        raise ValueError('All 29 actual historical releases and matured labels required')
    return dataset


def source_proof(root):
    root=Path(root);bundle=read_json(root/'seed/moira-bundle.json')
    paths=[root/'seed/moira-bundle.json',root/'seed/moira-prepared.json',Path(bundle['daily_path'])]
    if file_hash(bundle['daily_path']) != bundle['daily_sha256']: raise ValueError('Frozen daily history hash changed')
    for item in bundle['source_files']:
        path=Path(item['path'])
        if file_hash(path)!=item['sha256']: raise ValueError('An immutable historical release hash changed')
        paths.append(path)
    return {str(path):file_hash(path) for path in paths}


def _retained_episode(directory, model, run_id, report):
    last=report['cycles'][-1];kind='candidate' if last.get('accepted') else 'baseline'
    namespace=Namespace(model,'training',f'{run_id}:{last["cycle"]}:{kind}')
    path=Path(directory)/'episodes'/model/digest(vars(namespace))[:24]
    state=read_json(path/'live-state.json')
    if state is None: raise RuntimeError('Actual retained native evaluation episode absent')
    return path,state,kind


def train_model(root, output, model, *, max_cycles=8):
    from .budget import Budget
    from .llm import Gateway,read_settings
    from .moira_native import NativeTrainingAdapter, native_types
    root,output=Path(root),Path(output);directory=output/model
    if model not in TOP_K: raise ValueError('Only explicitly requested Moira 2.0 variants allowed')
    with locked(directory/'training.lock',nonblocking=True):
        existing=read_json(directory/'initial-training-status.json')
        if existing is not None:
            if existing.get('status')=='ready' and existing.get('checkpoint_sha256')==file_hash(directory/'prepared.json'):
                return existing
            raise RuntimeError('An incomplete/ambiguous paid training requires audit rather than automatic replay')
        before=source_proof(root);bundle=read_json(root/'seed/moira-bundle.json');prepared=read_json(root/'seed/moira-prepared.json')
        at=datetime.now(timezone.utc);dataset=phase_two_dataset(bundle,training_at=at)
        config=read_json(root/'config.json')
        import os
        os.environ['PAIR_TRADING_FIX_ROOT']=config['framework_root']
        native_types()
        gateway=Gateway(read_settings(config['key_file'])['OPENAI_API_KEY'],
            Budget(root).scoped(model,'training','initial-historical-rri'),root/'api-audit')
        run_id='initial-historical-rri:'+at.isoformat();status={'schema':'moira.initial-rri-training/1','status':'training',
            'model_id':model,'backbone':'gpt-4.1-mini','started_at':at.isoformat(),'source_window_start':dataset.window_start.isoformat(),
            'source_window_end':dataset.window_end.isoformat(),'actual_observations':29,'actual_matured_labels':29,
            'max_cycles':max_cycles,'parent_checkpoint_sha256':before[str(root/'seed/moira-prepared.json')],
            'source_hashes_before':before,'empty_training_bank_at_each_episode_start':True,
            'common_parent_already_trained_on_same_window':True,'evaluation_scope':'IN_SAMPLE_TRAINING',
            'training_returns_enter_live_cr':False,'initialization_is_not_a_scheduled_Monday_job':True}
        status['adapter_source_hashes']={str(Path(__file__).with_name(name)):file_hash(Path(__file__).with_name(name))
            for name in ('initial_training.py','moira_native.py','memory.py','training.py')}
        write_private(directory/'initial-training-status.json',status);write_private(directory/'frozen-training-inputs.json',bundle)
        write_private(directory/'parent-checkpoint.json',prepared)
        try:
            adapter=NativeTrainingAdapter(dataset=dataset,bundle=bundle,directory=directory,gateway=gateway,model_id=model)
            # No parent/live 29-unit seed enters an earlier historical action.
            report=train_native(adapter,dataset,prepared,version=model,run_id=run_id,max_cycles=max_cycles)
            episode_path,state,kind=_retained_episode(directory,model,run_id,report)
            if len(state['decisions'])!=29 or len(state['feedback'])!=29: raise RuntimeError('Incomplete native retained episode')
            returns=[state['feedback'][row.action_id]['step_return'] for row in dataset.observations]
            metrics=objective(returns)
            expected=report['cycles'][-1]['candidate_metrics' if kind=='candidate' else 'baseline_metrics']
            if metrics!=expected: raise RuntimeError('Retained native episode metrics disagree with immutable outcomes')
            after=source_proof(root)
            if before!=after: raise RuntimeError('Immutable initial historical training source changed')
            completed_at=now();seeds=[]
            for item in state['memory_bank']['units']:
                unit=copy.deepcopy(item['unit']);original_available=unit['available_at'];unit['available_at']=completed_at
                unit['provenance'].update(initial_training_model=model,initial_training_episode=str(episode_path),
                    source_outcome_available_at=original_available,trained_memory_available_at=completed_at,
                    training_scope='explicit initial Oct19–23 in-sample native RRI training')
                seeds.append(unit)
            if len(seeds)!=29: raise RuntimeError('All finalized retained action CF units required')
            report.update(finished_at=completed_at,retained_episode=str(episode_path),retained_evaluation_kind=kind,
                retained_metrics=metrics,last_candidate_metrics=report['cycles'][-1].get('candidate_metrics'),
                in_sample=True,source_hashes_before=before,source_hashes_after=after)
            checkpoint={key:copy.deepcopy(report['checkpoint'][key]) for key in ('model','status','prompt_memory','pair_memory','frozen_prompt_sha256')}
            checkpoint.update(schema='moira.initial-rri-checkpoint/1',framework='native Moira with RRI action-CF memory',
                finished_at=completed_at,model_id=model,parent_checkpoint_sha256=status['parent_checkpoint_sha256'],
                parent_training_cycles=prepared.get('completed_cycles'),initial_rri_training={
                    'source_window_start':dataset.window_start.isoformat(),'source_window_end':dataset.window_end.isoformat(),
                    'actual_observations':29,'actual_matured_labels':29,'completed_cycles':report['completed_cycles'],
                    'max_cycles':max_cycles,'accepted_updates':report['accepted_updates'],'objective':report['objective'],
                    'retained_episode':str(episode_path),'retained_metrics':metrics,
                    'last_candidate_metrics':report['cycles'][-1].get('candidate_metrics'),'in_sample':True,
                    'memory_top_k':TOP_K[model],'tau_sessions':252,'feedback_step':18})
            write_private(directory/'seed-records.json',seeds)
            write_private(directory/'retained-training-bank.json',state['memory_bank'])
            write_private(directory/'prepared.json',checkpoint);write_private(directory/'training-result.json',report)
            calls=[row for row in read_json(root/'api-budget.json')['calls'].values()
                if row.get('model_id')==model and row.get('stage')=='training' and row.get('event_id')=='initial-historical-rri']
            status.update(status='ready',finished_at=completed_at,completed_cycles=report['completed_cycles'],
                accepted_updates=report['accepted_updates'],checkpoint_sha256=file_hash(directory/'prepared.json'),
                source_hashes_after=after,retained_metrics=metrics,last_candidate_metrics=report['cycles'][-1].get('candidate_metrics'),
                actual_api_calls=len(calls),actual_completed_calls=sum(c['status']=='completed' for c in calls),
                actual_failed_calls=sum(c['status']=='failed' for c in calls),
                actual_cost_usd=sum(c.get('actual_cost_usd',0) for c in calls),
                actual_prompt_tokens=sum(c.get('input_tokens',0) for c in calls),
                actual_completion_tokens=sum(c.get('output_tokens',0) for c in calls),
                seed_unit_count=len(seeds),seed_records_sha256=file_hash(directory/'seed-records.json'),
                retained_training_bank_sha256=file_hash(directory/'retained-training-bank.json'))
            write_private(directory/'initial-training-status.json',status);return status
        except Exception as error:
            status.update(status='pending',failed_at=now(),error_type=type(error).__name__,error=str(error))
            write_private(directory/'initial-training-status.json',status);raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=['all',*TOP_K],default='all');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    proof=source_proof(args.root);bundle=read_json(args.root/'seed/moira-bundle.json')
    data=phase_two_dataset(bundle,training_at=datetime.now(timezone.utc))
    if not args.execute:
        print(json.dumps({'status':'checked_only','observations':len(data.observations),'matured_labels':len(data.executions),'source_files':len(proof)}));return
    models=list(TOP_K) if args.model=='all' else [args.model]
    # Separate native agents, checkpoint books and RRI banks; shared locked budget.
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        jobs={pool.submit(train_model,args.root,args.output,model):model for model in models}
        failed=[]
        for job in as_completed(jobs):
            model=jobs[job]
            try:
                result=job.result();print(json.dumps({key:result.get(key) for key in (
                    'model_id','status','completed_cycles','accepted_updates','retained_metrics','last_candidate_metrics',
                    'actual_api_calls','actual_cost_usd','checkpoint_sha256')}),flush=True)
            except Exception as error:
                failed.append(model);print(json.dumps({'model_id':model,'status':'pending','error_type':type(error).__name__}),flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
