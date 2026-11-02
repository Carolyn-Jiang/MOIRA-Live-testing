"""Explicit, resumable, single-model lifecycle correction in a private shadow.

Importing performs no work. Real execution requires a pinned authorization and
an explicit gateway; historical market timestamps never replace API wall time.
Old Task 2, released inputs, valuations, other five models and past costs are
read-only. A failed/ambiguous shadow request is never automatically repeated.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from .accounting import (new_book, record_decision, settle_due, matured_feedback,
                         validate_book)
from .calendar import DEFAULT_CALENDAR, START_AT, aware
from .data import observation, release_id, round_id
from .registry import MODEL_IDS
from .pair_lifecycle import validate_pair_lifecycle
from .storage import read_json, write_private, file_hash, digest, locked, now

MODEL = 'moira'
SCHEMA = 'moira.live-testing.lifecycle-correction/1'
FIRST = START_AT.isoformat()


def _tree(directory):
    result = {}
    for path in Path(directory).rglob('*'):
        if path.is_symlink(): raise ValueError('Correction sources cannot contain symlinks')
        if path.is_file(): result[str(path.relative_to(directory))] = file_hash(path)
    return result


def _bytes(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with os.fdopen(os.open(temporary, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, 0o600), 'wb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); path.chmod(0o600)


def _prepared(document):
    if (document.get('status') != 'ready' or document.get('model') != 'gpt-4.1-mini'
            or hashlib.sha256(document['prompt_memory']['active_prompt'].encode()).hexdigest()
            != document.get('frozen_prompt_sha256') or not isinstance(document.get('pair_memory'), list)):
        raise ValueError('A genuine, valid frozen native checkpoint is required')


def _other_models(root):
    return {model:_tree(root/'models'/model) for model in MODEL_IDS if model != MODEL}


def authorization(root, *, original_native_copy, statement, preserve_first=True,
                  replacement_prepared=None):
    """Read-only construction; the human/root caller saves this privately."""
    root = Path(root).resolve(); source = Path(original_native_copy).resolve()
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError('Explicit correction authorization text is required')
    if type(preserve_first) is not bool: raise ValueError('Explicit first-action policy required')
    book, runtime, bootstrap = (read_json(root/name) for name in ('book.json','runtime.json','bootstrap.json'))
    validate_book(book)
    copied = [item for item in bootstrap['source_files'].values()
              if Path(item['copy']).resolve() == root/'models/moira/live-state.json']
    if len(copied) != 1 or file_hash(source) != copied[0]['copy_sha256']:
        raise ValueError('The immutable bootstrap native copy must match its original recorded hash')
    original = read_json(source); first = original['decisions'][round_id(START_AT)]
    retained = book['decisions'][MODEL][FIRST]
    if ({key:first['action'][key] for key in ('stock1','stock2','action')} != retained['action']
            or aware(first['action']['generated_at']) != aware(retained['generated_at'])):
        raise ValueError('First original native action does not match the accepted accounting decision')
    current_prepared = root/'models/moira/prepared.json'
    prepared_path = Path(replacement_prepared).resolve() if replacement_prepared else current_prepared
    _prepared(read_json(prepared_path))
    if preserve_first and file_hash(prepared_path) != file_hash(current_prepared):
        raise ValueError('A changed checkpoint requires a complete new retrospective replay')
    releases = [at for at in DEFAULT_CALENDAR.release_nodes if at.date()==START_AT.date()]
    if len(releases) != 6 or any(at.isoformat() not in runtime['published'] for at in releases):
        raise ValueError('All six already-published frozen first-day observations are required')
    if (set(book['decisions'][MODEL]) != {at.isoformat() for at in releases}
            or len(book['accounts'][MODEL]['history']) != 4):
        raise ValueError('This correction is limited to six first-day actions and four exited outcomes')
    paths = [root/name for name in ('book.json','runtime.json','config.json','bootstrap.json',
             'seed/moira-bundle.json','company-context.json')]
    paths += [root/'models/moira'/name for name in ('live-state.json','prepared.json','training-inputs.json')]
    paths += [root/folder/(release_id(at)+'.json') for at in releases for folder in ('releases','inputs')]
    for at in releases:
        published = runtime['published'][at.isoformat()]
        for filename, expected in published['file_sha256'].items():
            path = root/'news/hf-staging'/published['folder']/filename
            if file_hash(path) != expected: raise ValueError('Published private data bytes changed')
            paths.append(path)
    paths += [source, prepared_path]
    return {'schema':SCHEMA, 'status':'authorized', 'authorized_at':now(), 'authorization':statement,
        'root':str(root), 'model_id':MODEL, 'preserve_first':preserve_first,
        'original_native_copy':str(source), 'prepared_source':str(prepared_path),
        'release_times':[at.isoformat() for at in releases],
        'file_sha256':{str(path):file_hash(path) for path in paths},
        'model_tree_sha256':_tree(root/'models/moira'), 'other_model_trees':_other_models(root),
        'published_sha256':digest(runtime['published']), 'valuations_sha256':digest(book['valuations'])}


def restored_prefix(original, prepared, accepted, *, preserve_first=True):
    """Reconstruct the unresolved post-09:30 state without outcome contamination."""
    _prepared(prepared); state = copy.deepcopy(original)
    state['history'] = {key:copy.deepcopy(value) for key,value in original['history'].items()
                        if aware(key) < START_AT or (preserve_first and aware(key) == START_AT)}
    state.update(model_id=MODEL, scope='live', run_id='experiment',
        prepared_prompt_sha256=prepared['frozen_prompt_sha256'], pair_memory=copy.deepcopy(prepared['pair_memory']),
        decisions={}, attempts={}, feedback={}, direction_memory=[], active_pair=None,
        pending_break=False, pending_since=None)
    state.pop('closed', None)
    state.pop('prepared_sha256', None)  # A legacy byte hash cannot label a replacement checkpoint.
    if preserve_first:
        key = round_id(START_AT); first = copy.deepcopy(original['decisions'][key])
        if ({name:first['action'][name] for name in ('stock1','stock2','action')} != accepted['action']
                or aware(first['action']['generated_at']) != aware(accepted['generated_at'])):
            raise ValueError('Retained first decision changed')
        source = original['direction_memory'][first['memory_index']]
        # Exact fields emitted by native DirectionMemory.add_decision, before
        # set_counterfactual can attach prices, deltas, labels or realized PnL.
        clean = {name:copy.deepcopy(source.get(name)) for name in (
            'selected_pair','action','reason','decision_index','decision_date',
            'lifecycle_state','original_action','final_action')}
        clean.update(realized_pnl=None, decision_index=0)
        first['memory_index'] = 0
        state.update(active_pair=[first['action']['stock1'],first['action']['stock2']],
                     decisions={key:first}, direction_memory=[clean])
    return state


def _check_pins(root, auth):
    if (auth.get('schema') != SCHEMA or auth.get('status') != 'authorized'
            or auth.get('root') != str(root) or auth.get('model_id') != MODEL):
        raise ValueError('A private, explicit pinned correction authorization is required')
    for path, expected in auth['file_sha256'].items():
        if file_hash(path) != expected: raise ValueError('An authorized source advanced; review rather than replay')
    if _tree(root/'models/moira') != auth['model_tree_sha256'] or _other_models(root) != auth['other_model_trees']:
        raise ValueError('A live model changed after correction authorization')


def _mature(book, shadow, bundle, gateway, cutoff, directory, native):
    """Only this model and quotes reached by this observation can mature."""
    single = new_book([MODEL], book['symbols'])
    single['accounts'][MODEL] = copy.deepcopy(book['accounts'][MODEL])
    single['decisions'][MODEL] = copy.deepcopy(book['decisions'][MODEL])
    single['valuations'] = {key:copy.deepcopy(value) for key,value in book['valuations'].items()
                            if aware(key) <= cutoff}
    settle_due(single, aware(now()))
    book['accounts'][MODEL] = copy.deepcopy(single['accounts'][MODEL])
    write_private(directory/'book.json', book)  # Fixed settled_at survives retry.
    for feedback in matured_feedback(single, MODEL, aware(now()), observation_asof=cutoff):
        if aware(feedback['exit_at']) > cutoff: raise ValueError('Future-source feedback is forbidden')
        native.apply_feedback(feedback, shadow, bundle,
            gateway.scoped(MODEL,'replay','lifecycle-feedback:'+feedback['release_at']),MODEL)


def _preserved(before_book, book, before_runtime, runtime):
    first, second = copy.deepcopy(before_book), copy.deepcopy(book)
    for document in (first,second):
        document['accounts'].pop(MODEL); document['decisions'].pop(MODEL)
    if first != second: raise ValueError('Another model, usage record or immutable valuation changed')
    first, second = copy.deepcopy(before_runtime), copy.deepcopy(runtime)
    for document in (first,second):
        document['feedback_applied'].pop(MODEL)
        document['model_attempts'] = {key:value for key,value in document['model_attempts'].items()
                                    if not key.startswith(MODEL+':')}
    if first != second: raise ValueError('Another dispatcher account changed')


def _installed(root, journal):
    current = read_json(root/'book.json')
    for key, expected in journal['installed_decision_sha256'].items():
        if digest(current['decisions'][MODEL][key]) != expected:
            raise ValueError('An installed correction decision changed')
    if digest(current['accounts'][MODEL]['history'][:journal['settled_prefix']]) != journal['installed_history_sha256']:
        raise ValueError('An installed correction outcome changed')
    return journal


def assert_installation_consistent(root):
    """Dispatcher preflight: an interrupted installation must be recovered first."""
    root=Path(root); pointer=read_json(root/'repairs/moira-lifecycle/current.json')
    if pointer:
        journal=read_json(Path(pointer['operation'])/'status.json')
        if journal and journal.get('status')=='committing':
            raise RuntimeError('An interrupted single-model installation requires correction recovery')


def _recover_commit(root, operation, journal, auth):
    """A process kill cannot leave a mixed live model/book/runtime on retry."""
    previous=operation/'live-before'; shadow=operation/'shadow/moira'; live=root/'models/moira'
    for name in ('book.json','runtime.json'):
        if file_hash(operation/'archive'/name)!=auth['file_sha256'][str(root/name)]:
            raise ValueError('The exact interrupted-install rollback archive changed')
    if _tree(previous if previous.exists() else live)!=auth['model_tree_sha256']:
        raise ValueError('The original model rollback archive changed')
    if previous.exists():
        if live.exists():
            if shadow.exists(): raise RuntimeError('Ambiguous interrupted model installation requires review')
            live.rename(shadow)
        previous.rename(live)
    for name in ('book.json','runtime.json'):
        _bytes(root/name,(operation/'archive'/name).read_bytes())
    journal.update(status='shadow_pending',interrupted_commit_restored_at=now())
    write_private(operation/'status.json',journal)


@contextmanager
def _repair_locks(root,operation,auth):
    with locked(root/'run.lock',nonblocking=True):
        journal=read_json(operation/'status.json')
        if journal and journal.get('status')=='committing':_recover_commit(root,operation,journal,auth)
        with locked(root/'models/moira/runtime.lock',nonblocking=True):yield


def repair(root, authorization_path, gateway, *, native=None):
    """Paid execution is root-only; all failure states remain private and resumable."""
    root = Path(root).resolve(); auth = read_json(authorization_path)
    if native is None:
        from . import moira_native as native
    operation = root/'repairs/moira-lifecycle'/digest(auth)[:24]
    with _repair_locks(root,operation,auth):
        journal_path = operation/'status.json'; journal = read_json(journal_path)
        if journal and journal.get('status') == 'installed': return _installed(root,journal)
        _check_pins(root,auth)
        pointer = root/'repairs/moira-lifecycle/current.json'; active = read_json(pointer)
        if active and active.get('authorization_sha256') != digest(auth):
            raise ValueError('A different correction operation already requires review')
        before_book, before_runtime = read_json(root/'book.json'), read_json(root/'runtime.json')
        shadow = operation/'shadow/moira'
        if journal is None:
            operation.mkdir(parents=True,exist_ok=True,mode=0o700)
            for name in ('book.json','runtime.json','api-budget.json'):
                _bytes(operation/'archive'/name,(root/name).read_bytes())
            shutil.copytree(root/'models/moira',operation/'archive/moira')
            shutil.copytree(root/'models/moira',shadow)
            if not auth['preserve_first']:
                # The old checkpoint's complete call log remains in archive.
                # A newly trained lineage must expose only its actual new
                # calls to future weekly belief-attribution reviews.
                (shadow/'model-calls.jsonl').unlink(missing_ok=True)
            prepared = read_json(auth['prepared_source']); original = read_json(auth['original_native_copy'])
            initial = restored_prefix(original,prepared,before_book['decisions'][MODEL][FIRST],
                                      preserve_first=auth['preserve_first'])
            initial['prepared_sha256'] = file_hash(auth['prepared_source'])
            initial['checkpoint_correction_authorization_sha256'] = digest(auth)
            write_private(shadow/'live-state.json',initial)
            _bytes(shadow/'prepared.json',Path(auth['prepared_source']).read_bytes())
            working = copy.deepcopy(before_book); working['accounts'][MODEL] = {'equity':1.,'history':[]}
            working['decisions'][MODEL] = ({FIRST:copy.deepcopy(before_book['decisions'][MODEL][FIRST])}
                                          if auth['preserve_first'] else {})
            write_private(operation/'book.json',working)
            journal = {'schema':SCHEMA,'status':'shadow_running','started_at':now(),
                'authorization_sha256':digest(auth),'operation':str(operation),'preserve_first':auth['preserve_first']}
            write_private(journal_path,journal); write_private(pointer,{'authorization_sha256':digest(auth),'operation':str(operation)})
        working = read_json(operation/'book.json'); bundle = read_json(root/'seed/moira-bundle.json')
        try:
            for stamp in auth['release_times']:
                at = aware(stamp)
                if stamp in working['decisions'][MODEL]: continue
                _mature(working,shadow,bundle,gateway,at,operation,native)
                data = read_json(root/'releases'/(release_id(at)+'.json')); obs = observation(data)
                accepted = read_json(shadow/'live-state.json')['decisions'].get(round_id(at))
                if accepted is not None:
                    if accepted.get('input_sha256') != digest(obs): raise ValueError('Accepted shadow input changed')
                    action = accepted['action']  # Crash after acceptance never repeats a paid call.
                else:
                    action = native.decide(obs,MODEL,shadow,bundle,
                        gateway.scoped(MODEL,'replay','lifecycle-correction:'+round_id(at)),
                        asof=aware(now()),replay=True)
                record_decision(working,MODEL,at,{key:action[key] for key in ('stock1','stock2','action')},
                    generated_at=action['generated_at'],input_asof=at,mode='retrospective_point_in_time',
                    audit={'authorization':auth['authorization'],'input_sha256':digest(obs),
                           'data_manifest_sha256':digest(before_runtime['published'][stamp]),
                           'lifecycle_correction_authorization_sha256':digest(auth)})
                write_private(operation/'book.json',working)
            _mature(working,shadow,bundle,gateway,aware(auth['release_times'][-1]),operation,native)
            validate_book(working)
            if len(working['decisions'][MODEL]) != 6 or len(working['accounts'][MODEL]['history']) != 4:
                raise ValueError('Exactly six first-day decisions and four genuinely exited outcomes required')
            live = read_json(shadow/'live-state.json'); pending = False; previous = None
            validate_pair_lifecycle(MODEL,live['decisions'])
            validate_pair_lifecycle(MODEL,working['decisions'][MODEL])
            for stamp in auth['release_times']:
                at=aware(stamp); row=live['decisions'][round_id(at)]; pair=tuple(row['action'][k] for k in ('stock1','stock2'))
                pending = pending or bool((row.get('monitor') or {}).get('pending_break'))
                if previous is not None:
                    expected_switch = pending and (at.hour,at.minute)==(15,30)
                    if bool(row.get('pair_switched')) != expected_switch or (pair != previous) != expected_switch:
                        raise ValueError('A corrected pair failed the exact pending-BREAK 15:30 transition')
                    if expected_switch: pending=False
                previous=pair
            if bool(live.get('pending_break')) != pending:
                raise ValueError('The corrected pending-BREAK state differs from its accepted transitions')
            runtime = copy.deepcopy(before_runtime)
            runtime['feedback_applied'][MODEL]=[row['release_at'] for row in working['accounts'][MODEL]['history']]
            for stamp in auth['release_times'][1 if auth['preserve_first'] else 0:]:
                rid=round_id(aware(stamp));attempt=live['attempts'][rid]
                runtime['model_attempts'][MODEL+':'+stamp]={'status':'complete','started_at':attempt['started_at'],
                    'generated_at':working['decisions'][MODEL][stamp]['generated_at'],
                    'input_sha256':attempt['input_sha256'],'correction_authorization_sha256':digest(auth)}
            _preserved(before_book,working,before_runtime,runtime); _check_pins(root,auth)
            current_calls=read_json(root/'api-budget.json')['calls']
            archived_calls=read_json(operation/'archive/api-budget.json')['calls']
            if any(current_calls.get(key)!=value for key,value in archived_calls.items()):
                raise ValueError('Historical API expenses or usage changed')
            journal.update(status='committing'); write_private(journal_path,journal)
            moved=False
            try:
                (root/'models/moira').rename(operation/'live-before'); moved=True
                shadow.rename(root/'models/moira')
                write_private(root/'book.json',working); write_private(root/'runtime.json',runtime)
                if _other_models(root)!=auth['other_model_trees']: raise ValueError('Another model changed during installation')
                journal.update(status='installed',installed_at=now(),settled_prefix=4,
                    installed_decision_sha256={key:digest(value) for key,value in working['decisions'][MODEL].items()},
                    installed_history_sha256=digest(working['accounts'][MODEL]['history']),
                    other_models_unchanged=True,published_and_valuations_unchanged=True,
                    replayed_decisions=5 if auth['preserve_first'] else 6,old_cost_records_preserved=True,
                    website_update_required=True)
                write_private(journal_path,journal)
            except Exception:
                if (root/'models/moira').exists() and moved:
                    (root/'models/moira').rename(operation/'failed-install')
                if moved:(operation/'live-before').rename(root/'models/moira')
                for name in ('book.json','runtime.json'):_bytes(root/name,(operation/'archive'/name).read_bytes())
                journal.update(status='rolled_back',failed_at=now());write_private(journal_path,journal);raise
            return journal
        except Exception as error:
            if journal.get('status') not in ('installed','rolled_back'):
                journal.update(status='shadow_pending',failed_at=now(),error_type=type(error).__name__)
                write_private(journal_path,journal)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--authorize',action='store_true')
    parser.add_argument('--original-native-copy',type=Path);parser.add_argument('--statement')
    parser.add_argument('--replacement-prepared',type=Path);parser.add_argument('--replay-all',action='store_true')
    parser.add_argument('--run',action='store_true');args=parser.parse_args(argv)
    if args.authorize:
        if args.authorization.exists():raise ValueError('An explicit authorization cannot be overwritten')
        record=authorization(args.root,original_native_copy=args.original_native_copy,statement=args.statement,
            preserve_first=not args.replay_all,replacement_prepared=args.replacement_prepared)
        write_private(args.authorization,record);print(json.dumps({'status':'authorized','retained_first':record['preserve_first']}));return
    if not args.run:raise ValueError('Use explicit --authorize or --run')
    from .budget import Budget
    from .llm import read_settings
    from .native_output_guard import create_guarded_gateway
    config=read_json(args.root/'config.json');os.environ['PAIR_TRADING_FIX_ROOT']=config['framework_root']
    os.environ['MOIRA_LIVE_STATE_DIR']=str(args.root.resolve())
    gateway=create_guarded_gateway(read_settings(config['key_file'])['OPENAI_API_KEY'],Budget(args.root),args.root/'api-audit')
    result=repair(args.root,args.authorization,gateway)
    print(json.dumps({key:result[key] for key in ('status','replayed_decisions','settled_prefix','other_models_unchanged')}))


if __name__=='__main__':main()
