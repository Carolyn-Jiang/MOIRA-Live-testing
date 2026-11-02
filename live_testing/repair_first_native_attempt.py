"""Filesystem-only preparation for one audited first-step network retry.

No SDK, key, model, embedding, market request or accepted decision is changed.
The outer runtime lock must be free. Exact original wire/reservation metadata
remain in place, and all three modified records have byte-exact rollback copies.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import json
import os
import re
from pathlib import Path

from .data import observation
from .storage import digest,file_hash,locked,now,read_json,write_private

MODEL='moira2-top1'
SOURCE_AT='2026-11-02T09:30:00-05:00'
NATIVE_ID='2026-11-02_09:30'


def _bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_suffix(path.suffix+'.repair-tmp')
    with os.fdopen(os.open(temporary,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600),'wb') as handle:
        handle.write(data);handle.flush();os.fsync(handle.fileno())
    temporary.replace(path);path.chmod(0o600)


def prepare(root,*,reservation_id):
    if not isinstance(reservation_id,str) or not re.fullmatch(r'[a-f0-9]{32}',reservation_id):
        raise ValueError('An exact private reservation identity is required')
    root=Path(root);folder=root/'models'/MODEL;attempt_key=MODEL+':'+SOURCE_AT
    with locked(root/'run.lock',nonblocking=True),locked(folder/'runtime.lock',nonblocking=True):
        runtime=read_json(root/'runtime.json');book=read_json(root/'book.json');native=read_json(folder/'live-state.json')
        marker=read_json(folder/'initial-training-ready.json');saved=read_json(folder/'initial-training-ready-before-replay-review.json')
        promotion=read_json(folder/'initial-training-bank-promotion.json');budget=read_json(root/'api-budget.json')
        wire_path=root/'api-audit/http-requests'/f'{reservation_id}.json';wire=read_json(wire_path)
        failed=budget['calls'].get(reservation_id)
        source=read_json(root/'releases/2026-11-02_09-30_NY.json');fingerprint=digest(observation(source))
        if (book['decisions'][MODEL] or book['accounts'][MODEL]['equity']!=1.0 or book['accounts'][MODEL]['history']
                or native['decisions'] or native['feedback'] or native['active_pair'] is not None
                or any(key.startswith('2026-11-02') for key in native['history'])
                or len(native['direction_memory'])!=29 or len(native['memory_bank']['units'])!=29
                or digest(native['memory_bank'])!=promotion['promoted_snapshot_sha256']):
            raise ValueError('First-step repair cannot replace accepted decisions or learned/native business state')
        if (marker.get('status')!='operational_hold' or saved.get('status')!='ready'
                or saved.get('model_id')!=MODEL or saved.get('backbone')!='gpt-4.1-mini'
                or saved.get('checkpoint_sha256')!=file_hash(folder/'initial-training-checkpoint.json')
                or file_hash(folder/'prepared.json')!=saved['checkpoint_sha256']):
            raise ValueError('Explicit held marker and unchanged trained checkpoint proof required')
        native_attempt=native['attempts'].get(NATIVE_ID);runtime_attempt=runtime['model_attempts'].get(attempt_key)
        if (len(native['attempts'])!=1 or not native_attempt or native_attempt.get('status')!='in_progress'
                or native_attempt.get('input_sha256')!=fingerprint or not runtime_attempt
                or runtime_attempt.get('status')!='ambiguous' or runtime_attempt.get('input_sha256')!=fingerprint):
            raise ValueError('Exactly the first unaccepted immutable-input attempt may be cleared')
        if (not failed or failed.get('model_id')!=MODEL or failed.get('stage')!='replay'
                or failed.get('event_id')!=NATIVE_ID or failed.get('status')!='failed'
                or failed.get('error_type')!='RemoteProtocolError' or failed.get('status_code') is not None
                or failed.get('response_id') is not None or failed.get('guard_cost_usd',0)<=0
                or wire.get('status')!='failed' or wire.get('error_type')!='RemoteProtocolError'
                or wire.get('status_code') is not None or wire.get('phase')!='native_pair_selection_exact_candidates'
                or failed.get('request_sha256')!=wire.get('request_sha256')
                or wire.get('request_sha256')!=digest(wire['request_payload'])):
            raise ValueError('Only the specifically audited unanswered network failure may retry')
        targets=[root/'runtime.json',folder/'live-state.json',folder/'initial-training-ready.json']
        original={str(path):path.read_bytes() for path in targets}
        protected={str(root/'book.json'):file_hash(root/'book.json'),str(root/'api-budget.json'):file_hash(root/'api-budget.json'),
            str(wire_path):file_hash(wire_path),str(folder/'prepared.json'):file_hash(folder/'prepared.json'),
            str(folder/'initial-training-checkpoint.json'):file_hash(folder/'initial-training-checkpoint.json')}
        for model in ('moira2-top3','moira2-top5','moira'):
            path=root/'models'/model/'live-state.json'
            if path.exists():protected[str(path)]=file_hash(path)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        archive=root/'native-request-repairs'/('first-k1-'+stamp);archive.mkdir(parents=True,mode=0o700)
        for i,path in enumerate(targets):_bytes(archive/f'original-{i}-{path.name}',original[str(path)])
        write_private(archive/'failed-reservation.json',failed);write_private(archive/'failed-wire-reference.json',{
            'path':str(wire_path),'sha256':file_hash(wire_path),'reservation_id':reservation_id})
        proof={'schema':'moira.first-unaccepted-network-repair/1','status':'preparing','prepared_at':now(),
            'model_id':MODEL,'source_timestamp':SOURCE_AT,'source_input_sha256':fingerprint,
            'reservation_id':reservation_id,'failed_reservation_retained':True,'failed_guard_cost_retained':failed['guard_cost_usd'],
            'helper_sha256':file_hash(__file__),'original_file_sha256':{p:__import__('hashlib').sha256(v).hexdigest() for p,v in original.items()},
            'protected_hashes_before':protected,'accepted_decisions_changed':0,'trained_checkpoints_reset':False,'new_http_calls':0,
            'retry_count_authorized':1,'caller_must_reissue_first_step_before_later_observations':True}
        write_private(archive/'proof.json',proof)
        try:
            revised_runtime=copy.deepcopy(runtime);del revised_runtime['model_attempts'][attempt_key]
            revised_native=copy.deepcopy(native);del revised_native['attempts'][NATIVE_ID]
            write_private(root/'runtime.json',revised_runtime);write_private(folder/'live-state.json',revised_native)
            _bytes(folder/'initial-training-ready.json',(folder/'initial-training-ready-before-replay-review.json').read_bytes())
            for path,expected in protected.items():
                if file_hash(path)!=expected:raise RuntimeError('Protected model/book/budget/wire changed during repair')
            check_native=read_json(folder/'live-state.json');check_runtime=read_json(root/'runtime.json')
            check_native['attempts'][NATIVE_ID]=native_attempt;check_runtime['model_attempts'][attempt_key]=runtime_attempt
            if check_native!=native or check_runtime!=runtime:raise RuntimeError('Repair modified state beyond the two unaccepted attempt markers')
            proof.update(status='prepared',finished_at=now(),protected_hashes_after={p:file_hash(p) for p in protected},
                modified_file_sha256={str(p):file_hash(p) for p in targets})
            write_private(archive/'proof.json',proof);return {'status':'prepared','archive':str(archive),
                'model_id':MODEL,'source_timestamp':SOURCE_AT,'new_http_calls':0,'failed_reservation_retained':True}
        except Exception:
            for path,data in original.items():_bytes(path,data)
            proof.update(status='rolled_back',finished_at=now());write_private(archive/'proof.json',proof);raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reservation-id',required=True);args=parser.parse_args(argv)
    print(json.dumps(prepare(args.root,reservation_id=args.reservation_id)))


if __name__=='__main__':main()
