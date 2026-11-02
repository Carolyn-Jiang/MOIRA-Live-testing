"""Explicit staged retraining of the legacy CR-trained original Moira.

This repair uses the frozen October 19–23 observations and delayed labels.
It never installs a checkpoint, changes a live action, fetches market data, or
publishes a website. All provider calls keep their actual generation times.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .storage import digest, file_hash, locked, now, read_json, write_private


def fresh_checkpoint(legacy):
    """Retain the default prompt, with no CR-selected learned beliefs/memory."""
    if legacy.get('model') != 'gpt-4.1-mini' or legacy.get('status') != 'ready':
        raise ValueError('The retained original Moira checkpoint is required')
    base = legacy['prompt_memory']['base_prompt']
    if not isinstance(base, str) or not base.strip():
        raise ValueError('An explicit original default prompt is required')
    return {'model': 'gpt-4.1-mini', 'status': 'ready', 'pair_memory': [],
        'prompt_memory': {'version': 1, 'base_prompt': base, 'active_prompt': base,
            'accepted_lessons': [], 'rejected_lessons': []},
        'frozen_prompt_sha256': hashlib.sha256(base.encode()).hexdigest()}


def train(root, output, *, authorization, max_cycles=8):
    from .initial_training import phase_two_dataset, source_proof, _retained_episode
    from .moira_native import NativeTrainingAdapter, native_types
    from .native_output_guard import create_guarded_gateway
    from .budget import Budget
    from .llm import read_settings
    from .training import objective, train_native
    from .pair_lifecycle import validate_pair_lifecycle

    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError('Explicit current user retraining authorization is required')
    if type(max_cycles) is not int or not 1 <= max_cycles <= 8:
        raise ValueError('At most eight cycles per invocation')
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root not in output.parents:
        raise ValueError('Stage training artifacts inside the private runtime directory')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with locked(output/'training.lock', nonblocking=True):
        previous = read_json(output/'status.json')
        if previous is not None:
            if (previous.get('status') == 'ready'
                    and previous.get('checkpoint_sha256') == file_hash(output/'prepared.json')):
                return previous
            raise RuntimeError('An incomplete paid training needs audit before resumption')
        proof = source_proof(root)
        parent = root/'models/moira/prepared.json'
        proof[str(parent)] = file_hash(parent)
        bundle = read_json(root/'seed/moira-bundle.json')
        legacy = read_json(parent)
        checkpoint = fresh_checkpoint(legacy)
        config = read_json(root/'config.json')
        os.environ['PAIR_TRADING_FIX_ROOT'] = config['framework_root']
        os.environ['MOIRA_LIVE_STATE_DIR'] = str(root)
        os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(root.parent/'trading-local/tokenizer-cache'))
        native_types()
        at = datetime.now(timezone.utc)
        dataset = phase_two_dataset(bundle, training_at=at)
        event_id = 'original-sharpe-retraining:' + at.isoformat()
        gateway = create_guarded_gateway(read_settings(config['key_file'])['OPENAI_API_KEY'],
            Budget(root).scoped('moira', 'training', event_id), root/'api-audit')
        status = {'schema': 'moira.original-sharpe-retraining/1', 'status': 'training',
            'model_id': 'moira', 'backbone': 'gpt-4.1-mini', 'authorization': authorization,
            'started_at': at.isoformat(), 'event_id': event_id, 'max_cycles': max_cycles,
            'objective': 'sample_sharpe_without_annualization', 'source_hashes_before': proof,
            'source_window_start': dataset.window_start.isoformat(),
            'source_window_end': dataset.window_end.isoformat(),
            'actual_observations': 29, 'actual_matured_labels': 29,
            'fresh_default_prompt': True, 'fresh_learned_beliefs': True, 'fresh_pair_memory': True,
            'pair_reselection_policy': 'BREAK pending until real historical NY15:30 release',
            'evaluation_scope': 'IN_SAMPLE_TRAINING', 'training_returns_enter_live_cr': False,
            'live_install_performed': False,
            'adapter_source_hashes': {str(Path(__file__).with_name(name)):file_hash(Path(__file__).with_name(name))
                for name in ('retrain_original_sharpe.py', 'initial_training.py', 'moira_native.py', 'training.py', 'pair_lifecycle.py')}}
        write_private(output/'status.json', status)
        write_private(output/'legacy-checkpoint.json', legacy)
        write_private(output/'fresh-checkpoint.json', checkpoint)
        write_private(output/'frozen-training-inputs.json', bundle)
        try:
            adapter = NativeTrainingAdapter(dataset=dataset, bundle=bundle,
                directory=output, gateway=gateway, model_id='moira')
            run_id = 'original-sharpe-retraining:' + at.isoformat()
            report = train_native(adapter, dataset, checkpoint, version='moira',
                run_id=run_id, max_cycles=max_cycles)
            episode, state, kind = _retained_episode(output, 'moira', run_id, report)
            if len(state['decisions']) != 29 or len(state['feedback']) != 29:
                raise RuntimeError('The retained historical episode is incomplete')
            # Historical release nodes differ from the long-test calendar.
            class HistoricalCalendar:
                release_nodes = dataset.release_nodes
            for path in (output/'episodes/moira').iterdir():
                native = read_json(path/'live-state.json')
                if native is not None:
                    validate_pair_lifecycle('moira', native['decisions'], calendar=HistoricalCalendar())
            metrics = objective(state['feedback'][row.action_id]['step_return'] for row in dataset.observations)
            expected = report['cycles'][-1]['candidate_metrics' if kind == 'candidate' else 'baseline_metrics']
            if metrics != expected:
                raise RuntimeError('Retained metrics differ from immutable delayed outcomes')
            if any(file_hash(path) != value for path, value in proof.items()):
                raise RuntimeError('Protected historical source or live checkpoint changed')
            finished = now()
            trained = copy.deepcopy(report['checkpoint'])
            trained.update(schema='moira.original-sharpe-checkpoint/1', model_id='moira',
                framework='native Moira', finished_at=finished,
                original_sharpe_retraining={'objective': report['objective'],
                    'max_cycles': max_cycles, 'completed_cycles': report['completed_cycles'],
                    'accepted_updates': report['accepted_updates'], 'retained_metrics': metrics,
                    'retained_episode': str(episode), 'in_sample': True,
                    'source_window_start': dataset.window_start.isoformat(),
                    'source_window_end': dataset.window_end.isoformat(),
                    'source_legacy_checkpoint_sha256': proof[str(parent)],
                    'default_prompt_sha256': checkpoint['frozen_prompt_sha256'],
                    'fresh_learned_beliefs': True, 'fresh_pair_memory': True})
            report.update(finished_at=finished, retained_metrics=metrics,
                retained_episode=str(episode), retained_evaluation_kind=kind, in_sample=True)
            write_private(output/'training-result.json', report)
            write_private(output/'prepared.json', trained)
            calls = [c for c in read_json(root/'api-budget.json')['calls'].values()
                if c.get('model_id') == 'moira' and c.get('event_id') == event_id]
            status.update(status='ready', finished_at=finished,
                completed_cycles=report['completed_cycles'], accepted_updates=report['accepted_updates'],
                retained_metrics=metrics, last_candidate_metrics=report['cycles'][-1].get('candidate_metrics'),
                checkpoint_sha256=file_hash(output/'prepared.json'),
                actual_api_calls=len(calls), actual_completed_calls=sum(c['status']=='completed' for c in calls),
                actual_cost_usd=sum(c.get('actual_cost_usd',0) for c in calls),
                actual_prompt_tokens=sum(c.get('input_tokens',0) for c in calls),
                actual_completion_tokens=sum(c.get('output_tokens',0) for c in calls))
            write_private(output/'status.json', status)
            return status
        except Exception as error:
            status.update(status='pending', failed_at=now(), error_type=type(error).__name__)
            write_private(output/'status.json', status)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorization', required=True)
    args=parser.parse_args(argv)
    result=train(args.root,args.output,authorization=args.authorization)
    print(json.dumps({k:result.get(k) for k in ('status','model_id','objective','completed_cycles',
        'accepted_updates','retained_metrics','actual_api_calls','actual_cost_usd','checkpoint_sha256')}),flush=True)


if __name__=='__main__':main()
