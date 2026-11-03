"""Explicit three-variant v2 replay/install transaction; ordinary dispatch never replays.

All market inputs and old cost records are read-only.  Paid work is staged in
separate native directories and installed only after every variant verifies.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil

from .accounting import new_book, record_decision, settle_due, matured_feedback, validate_book
from .calendar import START_AT, DEFAULT_CALENDAR, aware
from .data import observation, release_id, round_id
from .memory import Namespace
from .memory_v2 import PROTOCOL_VERSION as PROTOCOL
from .pair_lifecycle import validate_pair_lifecycle
from .registry import TOP_K, MODEL_IDS
from .repair_moira_lifecycle import _bytes, _tree
from .storage import digest, file_hash, locked, now, read_json, write_private

MODELS = tuple(TOP_K)
SCHEMA = 'moira.ranking-v2-install/1'
ARTIFACTS = ('status.json', 'prepared.json', 'retained-training-bank.json',
             'training-result.json', 'source-proof.json', 'parent-checkpoint.json')


def _bank(snapshot, model):
    from .memory_v2 import CounterfactualMemoryBankV2
    namespace = Namespace(**snapshot['namespace'])
    if namespace.version != model or namespace.scope != 'training':
        raise ValueError('A separate, genuinely trained schema2 bank for this variant is required')
    result = CounterfactualMemoryBankV2.from_dict(snapshot, namespace=namespace)
    if len(snapshot['units']) != 29:
        raise ValueError('Exactly 29 genuine retained historical action units are required')
    return result


def _training(directory, model):
    directory = Path(directory).resolve()
    documents = {name: read_json(directory/name) for name in ARTIFACTS}
    if any(value is None for value in documents.values()):
        raise ValueError('Every completed training artifact and its source proof is required')
    status, prepared = documents['status.json'], documents['prepared.json']
    if (status.get('status') != 'ready' or status.get('model_id') != model
            or status.get('checkpoint_sha256') != file_hash(directory/'prepared.json')
            or status.get('retained_bank_sha256') != file_hash(directory/'retained-training-bank.json')
            or prepared.get('status') != 'ready' or prepared.get('model') != 'gpt-4.1-mini'
            or prepared.get('memory_policy_protocol') != PROTOCOL
            or hashlib.sha256(prepared['prompt_memory']['active_prompt'].encode()).hexdigest()
               != prepared.get('frozen_prompt_sha256')
            or not status.get('finished_at')):
        raise ValueError('A real ready hash-pinned v2 checkpoint and bank are required')
    report = documents['training-result.json']
    if (report.get('objective') != 'sample_sharpe_without_annualization'
            or not 1 <= report.get('completed_cycles', 0) <= report.get('max_cycles', 0) <= 8):
        raise ValueError('The actual maximum-eight-cycle sample-Sharpe training report is required')
    if report.get('checkpoint', {}).get('frozen_prompt_sha256') != prepared['frozen_prompt_sha256']:
        raise ValueError('The installed prompt must be the genuinely evaluated retained checkpoint')
    for cycle in report['cycles']:
        before = cycle['baseline_metrics']['sharpe_ratio']
        after = cycle.get('candidate_metrics', {}).get('sharpe_ratio')
        if cycle['accepted'] != (before is not None and after is not None and after > before):
            raise ValueError('Training acceptance differs from strict sample Sharpe')
    _bank(documents['retained-training-bank.json'], model)
    return {'directory': str(directory), 'finished_at': status['finished_at'],
            'files': {name: file_hash(directory/name) for name in ARTIFACTS}}


def _times():
    result = tuple(at for at in DEFAULT_CALENDAR.release_nodes if at.date() == START_AT.date())
    if len(result) != 6: raise ValueError('Only the six explicitly authorized first-day releases may be rebuilt')
    return result


def _shape(book):
    validate_book(book)
    expected = {at.isoformat() for at in _times()}
    if set(book['models']) != set(MODEL_IDS): raise ValueError('The six original independent accounts must remain')
    for model in MODELS:
        if set(book['decisions'][model]) != expected or len(book['accounts'][model]['history']) != 4:
            raise ValueError('This correction cannot drop a future decision or an additional genuine exit')


def authorization(root, training, *, statement):
    """Read-only pin construction; the CLI saves the explicit authorization."""
    root = Path(root).resolve()
    if not isinstance(statement, str) or not statement.strip() or set(training) != set(MODELS):
        raise ValueError('Explicit user authorization and three independent training directories are required')
    book, runtime = read_json(root/'book.json'), read_json(root/'runtime.json')
    _shape(book)
    paths = [root/name for name in ('book.json', 'runtime.json', 'config.json',
                                    'seed/moira-bundle.json', 'company-context.json')]
    for at in _times():
        stamp = at.isoformat(); published = runtime['published'].get(stamp)
        if not published: raise ValueError('Only the already-published six immutable observations may be replayed')
        data_path = root/'releases'/(release_id(at)+'.json')
        data = read_json(data_path)
        if (data['timestamp'] != stamp or data['prices'] != book['valuations'][stamp]['prices']):
            raise ValueError('Published observation and canonical valuation disagree')
        paths += [data_path, root/'inputs'/(release_id(at)+'.json')]
        for filename, expected in published['file_sha256'].items():
            if Path(filename).name != filename: raise ValueError('Unsafe published file basename')
            path = root/'news/hf-staging'/published['folder']/filename
            if file_hash(path) != expected: raise ValueError('Published file bytes changed')
            paths.append(path)
    descriptors = {model: _training(training[model], model) for model in MODELS}
    return {'schema': SCHEMA, 'status': 'authorized', 'root': str(root),
            'authorization': statement, 'authorized_at': now(), 'models': list(MODELS),
            'release_times': [at.isoformat() for at in _times()], 'training': descriptors,
            'files': {str(path): file_hash(path) for path in paths},
            'model_trees': {model: _tree(root/'models'/model) for model in MODEL_IDS},
            'old_cost_calls_sha256': digest(read_json(root/'api-budget.json')['calls'])}


def _pins(root, auth):
    if (auth.get('schema') != SCHEMA or auth.get('status') != 'authorized'
            or auth.get('root') != str(root) or auth.get('models') != list(MODELS)
            or auth.get('release_times') != [at.isoformat() for at in _times()]):
        raise ValueError('The exact private three-variant correction authorization is required')
    for path, expected in auth['files'].items():
        if file_hash(path) != expected: raise ValueError('An authorized source advanced; do not replay automatically')
    for model, expected in auth['model_trees'].items():
        if _tree(root/'models'/model) != expected: raise ValueError('An accepted native model advanced')
    for model, source in auth['training'].items():
        if _training(source['directory'], model) != source: raise ValueError('A completed training artifact changed')


def _mature(book, directory, bundle, gateway, at, native):
    # Never settle the two cross-session actions merely because repair runs later.
    reached = copy.deepcopy(book)
    reached['valuations'] = {key: value for key, value in book['valuations'].items() if aware(key) <= at}
    settle_due(reached, aware(now()))
    book['accounts'] = reached['accounts']
    write_private(directory/'account.json', book)
    for feedback in matured_feedback(reached, book['models'][0], aware(now()), observation_asof=at):
        native.apply_feedback(feedback, directory/'native', bundle, gateway,
                              book['models'][0], source_cutoff=at)


def _transitions(state, stamps):
    pending = False; previous = None
    for stamp in stamps:
        at = aware(stamp); row = state['decisions'][round_id(at)]
        pair = tuple(row['action'][key] for key in ('stock1', 'stock2'))
        pending = pending or bool((row.get('monitor') or {}).get('pending_break'))
        switch = previous is not None and pending and (at.hour, at.minute) == (15, 30)
        if bool(row.get('pair_switched')) != switch or previous is not None and (pair != previous) != switch:
            raise ValueError('A v2 pair transition must match its actual pending BREAK at 15:30')
        if switch: pending = False
        previous = pair
    if bool(state.get('pending_break')) != pending:
        raise ValueError('The v2 pending-BREAK state disagrees with its accepted transitions')


def _variant(root, operation, auth, model, gateway, native):
    directory = operation/'variants'/model; shadow = directory/'native'
    complete = read_json(directory/'complete.json')
    if complete is not None:
        if (_tree(shadow) != complete['native_tree']
                or file_hash(directory/'account.json') != complete['account_sha256']):
            raise ValueError('A completed shadow artifact changed')
        native.validate_state(read_json(shadow/'live-state.json'), model_id=model,
                              directory=shadow, bundle=read_json(root/'seed/moira-bundle.json'))
        return complete
    bundle = read_json(root/'seed/moira-bundle.json')
    descriptor = auth['training'][model]; trained = Path(descriptor['directory'])
    if not (directory/'account.json').exists():
        if shadow.exists(): raise RuntimeError('Ambiguous partially initialized shadow needs explicit review')
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        native.initialize(read_json(trained/'prepared.json'), bundle, shadow,
            training_bank=read_json(trained/'retained-training-bank.json'), gateway=gateway,
            model_id=model, scope='live', run_id='ranking-v2-install:'+digest(auth),
            training_completed_at=descriptor['finished_at'],
            checkpoint_sha256=descriptor['files']['prepared.json'])
        if file_hash(shadow/'prepared.json') != descriptor['files']['prepared.json']:
            raise ValueError('Native initialization changed the genuinely trained frozen checkpoint')
        single = new_book([model], read_json(root/'book.json')['symbols'])
        single['valuations'] = copy.deepcopy(read_json(root/'book.json')['valuations'])
        write_private(directory/'account.json', single)
    single = read_json(directory/'account.json')
    for stamp in auth['release_times']:
        if stamp in single['decisions'][model]: continue
        at = aware(stamp); _mature(single, directory, bundle, gateway, at, native)
        obs = observation(read_json(root/'releases'/(release_id(at)+'.json')))
        accepted = read_json(shadow/'live-state.json')['decisions'].get(round_id(at))
        if accepted is not None:
            if accepted.get('input_sha256') != digest(obs): raise ValueError('Accepted shadow input changed')
            action = accepted['action']
        else:
            action = native.decide(obs, model, shadow, bundle, gateway,
                                   asof=aware(now()), replay=True)
        record_decision(single, model, at, {key: action[key] for key in ('stock1', 'stock2', 'action')},
            generated_at=action['generated_at'], input_asof=at, mode='retrospective_point_in_time',
            audit={'authorization': auth['authorization'], 'input_sha256': digest(obs),
                   'ranking_v2_install_authorization_sha256': digest(auth),
                   'data_manifest_sha256': digest(read_json(root/'runtime.json')['published'][stamp])})
        write_private(directory/'account.json', single)
    _mature(single, directory, bundle, gateway, aware(auth['release_times'][-1]), native)
    validate_book(single)
    state = read_json(shadow/'live-state.json')
    if len(single['decisions'][model]) != 6 or len(single['accounts'][model]['history']) != 4:
        raise ValueError('Exactly six genuine actions and four exited returns are required')
    validate_pair_lifecycle(model, single['decisions'][model])
    validate_pair_lifecycle(model, state['decisions'])
    _transitions(state, auth['release_times'])
    native.validate_state(state, model_id=model, directory=shadow, bundle=bundle)
    if file_hash(shadow/'prepared.json') != descriptor['files']['prepared.json']:
        raise ValueError('The prepared checkpoint changed during genuine replay')
    if len(state['decisions']) != 6 or len(state['feedback']) != 4:
        raise ValueError('Native citations and all four attributed feedback batches must be complete')
    marker = {'protocol': PROTOCOL, 'status': 'ready', 'model_id': model,
              'checkpoint_sha256': descriptor['files']['prepared.json'],
              'training_result_sha256': descriptor['files']['training-result.json'],
              'training_result_path': str(trained/'training-result.json'),
              'training_bank_sha256': descriptor['files']['retained-training-bank.json'],
              'installation_authorization_sha256': digest(auth), 'installed_from_shadow_at': now()}
    write_private(shadow/'memory-policy.json', marker)
    result = {'status': 'complete', 'model_id': model, 'completed_at': now(),
              'native_tree': _tree(shadow), 'account_sha256': file_hash(directory/'account.json')}
    write_private(directory/'complete.json', result)
    return result


def assert_installation_consistent(root):
    pointer = read_json(Path(root)/'repairs/ranking-v2/current.json')
    if pointer:
        journal = read_json(Path(pointer['operation'])/'status.json')
        if journal is None or journal.get('status') == 'committing':
            raise RuntimeError('The interrupted three-model installation must recover before dispatch')


def _rollback(root, operation, auth, journal):
    # Idempotent even if the process died during a previous rollback itself.
    for name in ('book.json', 'runtime.json'):
        if file_hash(operation/'archive'/name) != auth['files'][str(root/name)]:
            raise ValueError('The original transaction rollback archive changed')
    for model in MODELS:
        old = operation/'live-before'/model; live = root/'models'/model
        shadow = operation/'variants'/model/'native'
        if old.exists():
            if _tree(old) != auth['model_trees'][model]: raise ValueError('The old native model changed')
            if live.exists():
                if shadow.exists(): raise RuntimeError('Ambiguous interrupted native installation')
                live.rename(shadow)
            old.rename(live)
        elif _tree(live) != auth['model_trees'][model]:
            raise ValueError('The unchanged/restored native model is unavailable')
    for name in ('book.json', 'runtime.json'):
        _bytes(root/name, (operation/'archive'/name).read_bytes())
    journal.update(status='shadow_pending', interrupted_install_restored_at=now())
    write_private(operation/'status.json', journal)


def _installed(root, operation, journal):
    book = read_json(root/'book.json'); validate_book(book)
    for model in MODELS:
        for key, expected in journal['installed_decisions'][model].items():
            if digest(book['decisions'][model][key]) != expected: raise ValueError('An installed action changed')
        if digest(book['accounts'][model]['history'][:4]) != journal['installed_histories'][model]:
            raise ValueError('An installed historical return changed')
    return journal


def install(root, authorization_path, gateway, *, native=None):
    """Explicit real replay only; importing/authorizing never makes a provider call."""
    root = Path(root).resolve(); auth = read_json(authorization_path)
    if native is None:
        from . import moira_v2_native as native
    operation = root/'repairs/ranking-v2'/digest(auth)[:24]
    with locked(root/'run.lock', nonblocking=True):
        journal = read_json(operation/'status.json')
        if journal and journal.get('status') == 'installed': return _installed(root, operation, journal)
        if journal and journal.get('status') == 'committing': _rollback(root, operation, auth, journal)
        _pins(root, auth)
        pointer = root/'repairs/ranking-v2/current.json'; active = read_json(pointer)
        if active and active.get('authorization_sha256') != digest(auth):
            raise ValueError('Another three-model correction already needs review')
        if journal is None:
            operation.mkdir(parents=True, exist_ok=True, mode=0o700)
            for name in ('book.json', 'runtime.json', 'api-budget.json'):
                _bytes(operation/'archive'/name, (root/name).read_bytes())
            if digest(read_json(operation/'archive/api-budget.json')['calls']) != auth['old_cost_calls_sha256']:
                raise ValueError('The original cost ledger advanced before the explicit replay began')
            for model in MODELS:
                # No paid work starts before this journal.  A killed byte-copy
                # may safely resume from the same hash-pinned original files.
                target = operation/'archive/models'/model
                shutil.copytree(root/'models'/model, target, dirs_exist_ok=True)
                if _tree(target) != auth['model_trees'][model]:
                    raise ValueError('The original native archive contains changed or unexpected files')
            journal = {'schema': SCHEMA, 'status': 'shadow_running', 'started_at': now(),
                       'authorization_sha256': digest(auth), 'operation': str(operation)}
            write_private(operation/'status.json', journal)
            write_private(pointer, {'operation': str(operation), 'authorization_sha256': digest(auth)})
        failures = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = {pool.submit(_variant, root, operation, auth, model,
                gateway.scoped(model, 'replay', 'ranking-v2-install:'+digest(auth)), native): model for model in MODELS}
            for task in as_completed(tasks):
                try: task.result()
                except Exception as error: failures[tasks[task]] = type(error).__name__
        if failures:
            journal.update(status='shadow_pending', failures=failures, updated_at=now())
            write_private(operation/'status.json', journal)
            return {'status': 'pending', 'failures': failures, 'operation': str(operation)}
        _pins(root, auth)
        original = read_json(operation/'archive/book.json'); book = copy.deepcopy(original)
        runtime = read_json(operation/'archive/runtime.json')
        for model in MODELS:
            staged = read_json(operation/'variants'/model/'account.json')
            book['accounts'][model] = staged['accounts'][model]
            book['decisions'][model] = staged['decisions'][model]
            runtime['feedback_applied'][model] = [row['release_at'] for row in staged['accounts'][model]['history']]
            state = read_json(operation/'variants'/model/'native/live-state.json')
            for stamp in auth['release_times']:
                row = state['decisions'][round_id(aware(stamp))]
                runtime['model_attempts'][model+':'+stamp] = {'status': 'complete',
                    'started_at': state['attempts'][round_id(aware(stamp))]['started_at'],
                    'generated_at': book['decisions'][model][stamp]['generated_at'],
                    'input_sha256': row['input_sha256'], 'ranking_v2_authorization_sha256': digest(auth)}
        _shape(book)
        previous_costs = read_json(operation/'archive/api-budget.json')['calls']
        costs = read_json(root/'api-budget.json')['calls']
        if any(costs.get(key) != value for key, value in previous_costs.items()):
            raise ValueError('Historical API costs cannot be removed, refunded, or rewritten')
        for model in MODEL_IDS:
            if model not in MODELS and (book['decisions'][model] != original['decisions'][model]
                    or book['accounts'][model] != original['accounts'][model]):
                raise ValueError('An original model account changed')
        journal.update(status='committing'); write_private(operation/'status.json', journal)
        (operation/'live-before').mkdir(exist_ok=True, mode=0o700)
        try:
            for model in MODELS:
                (root/'models'/model).rename(operation/'live-before'/model)
                (operation/'variants'/model/'native').rename(root/'models'/model)
            for model in MODELS:
                directory = root/'models'/model
                native.validate_state(read_json(directory/'live-state.json'), model_id=model,
                    directory=directory, bundle=read_json(root/'seed/moira-bundle.json'))
            write_private(root/'book.json', book); write_private(root/'runtime.json', runtime)
            for model in MODEL_IDS:
                if model not in MODELS and _tree(root/'models'/model) != auth['model_trees'][model]:
                    raise ValueError('An original native model changed during installation')
            journal.update(status='installed', installed_at=now(), models=list(MODELS),
                replayed_decisions=18, settled_prefix=4, old_cost_records_preserved=True,
                installed_decisions={model: {key:digest(value) for key,value in book['decisions'][model].items()} for model in MODELS},
                installed_histories={model:digest(book['accounts'][model]['history']) for model in MODELS},
                website_update_required=True)
            write_private(operation/'status.json', journal)
        except Exception:
            _rollback(root, operation, auth, journal)
            raise
        return journal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--authorize', action='store_true'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--statement'); parser.add_argument('--training', action='append', default=[])
    args = parser.parse_args(argv)
    if args.authorize == args.run: raise ValueError('Choose exactly one explicit authorization or execution mode')
    if args.authorize:
        if args.authorization.exists(): raise ValueError('Do not replace an explicit saved authorization')
        training = dict(item.split('=', 1) for item in args.training)
        record = authorization(args.root, training, statement=args.statement)
        import json
        write_private(args.authorization, record); print(json.dumps({'status': 'authorized', 'models': list(MODELS)})); return
    from .budget import Budget
    from .llm import read_settings
    from .native_output_guard import create_guarded_gateway
    import json, os
    config = read_json(args.root/'config.json')
    os.environ['PAIR_TRADING_FIX_ROOT'] = config['framework_root']
    os.environ['MOIRA_LIVE_STATE_DIR'] = str(args.root.resolve())
    gateway = create_guarded_gateway(read_settings(config['key_file'])['OPENAI_API_KEY'], Budget(args.root), args.root/'api-audit')
    result = install(args.root, args.authorization, gateway)
    print(json.dumps({key:result.get(key) for key in ('status', 'replayed_decisions', 'settled_prefix', 'failures')}))


if __name__ == '__main__': main()
