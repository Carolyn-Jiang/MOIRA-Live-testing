"""Resume an audited, unaccepted invalid native pair or action enum response.

Original paid artifacts are immutable. A new staging copy reuses completed
native decisions/reflections through their existing caches. Successful monitor
responses from the failed step are reused only on exact prompt/payload equality.
The invalid selection is reissued through a real, budgeted enum-constrained API;
no selection or action is coerced, invented or accepted from a fallback.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from . import initial_training, llm, moira_native
from .storage import append_private,digest,file_hash,now,read_json,write_private


def tree_hashes(directory):
    directory=Path(directory)
    return {str(p.relative_to(directory)):file_hash(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def failed_selection(source):
    source=Path(source);status=read_json(source/'initial-training-status.json')
    if (not status or status.get('status')!='pending' or status.get('error')!='Native inference failed; no synthetic fallback accepted'):
        raise ValueError('Only a specifically audited native validation rejection may resume')
    failed=[]
    for path in (source/'episodes'/status['model_id']).glob('*/live-state.json'):
        state=read_json(path)
        for key,attempt in state.get('attempts',{}).items():
            if attempt['status']!='complete':failed.append((path,state,key))
    if len(failed)!=1:raise ValueError('Exactly one unaccepted failed attempt is required')
    path,state,key=failed[0]
    if key in state['decisions']:raise ValueError('Accepted decisions cannot be retried')
    calls=[json.loads(line) for line in (path.parent/'model-calls.jsonl').read_text().splitlines()]
    last=calls[-1]
    if last.get('task_type') not in ('pair_selection','daily_direction') or last.get('error_type') or not last.get('response'):
        raise ValueError('The paid failure must be a completed pair or action response')
    parsed=json.loads(last['response'])
    if last['task_type']=='pair_selection':
        value=parsed.get('selected_pair');pool=last['payload']['top5_candidates']
        selected=[p.strip().upper() for p in value.split('/')] if isinstance(value,str) else value
        allowed=[list(item['pair']) for item in pool]
        if isinstance(selected,list) and (selected in allowed or selected[::-1] in allowed):
            raise ValueError('A valid selection is not an audited rejection')
    elif parsed.get('action') in ('LONG_SHORT','SHORT_LONG','FLAT'):
        raise ValueError('A valid accepted action cannot be retried')
    timestamp=last['payload']['current_observation']['timestamp']
    records=[row for row in calls if row['payload']['current_observation']['timestamp']==timestamp]
    prefix=records[:-1]
    if any(row.get('error_type') or not row.get('response') or row.get('task_type')=='daily_direction' for row in prefix):
        raise ValueError('Only completed successful prior native responses may be reused')
    return status,path,state,key,prefix,last


class ReuseClock:
    def __init__(self,prefix):self.queue=copy.deepcopy(prefix);self.current=None;self.generated_returned=False
    def timestamp(self):
        if self.current is not None:
            if not self.generated_returned:
                self.generated_returned=True;return self.current['generated_at']
            self.current=None
        if self.queue:return self.queue[0]['started_at']
        return now()
    def consume(self,system,payload,task):
        if not self.queue:return None
        row=self.queue[0]
        if row['system_prompt']!=system or row['payload']!=payload or row.get('task_type')!=task:
            raise ValueError('Successful paid monitor response cannot replay against a different prompt/payload')
        self.queue.pop(0);self.current=row;self.generated_returned=False;return row


def enum_selection_schema(payload):
    pairs=['/'.join(row['pair']) for row in payload['top5_candidates']]
    if not pairs or len(set(pairs))!=len(pairs):raise ValueError('Exact nonempty native candidate pool required')
    return {'type':'object','properties':{'selected_pair':{'type':'string','enum':pairs},
        'pair_selection_reason':{'type':'string'}},'required':['selected_pair','pair_selection_reason'],'additionalProperties':False}


def enum_direction_schema():
    return {'type':'object','properties':{'action':{'type':'string','enum':['LONG_SHORT','SHORT_LONG','FLAT']},
        'direction_reason':{'type':'string'}},'required':['action','direction_reason'],'additionalProperties':False}


def gateway_class(clock,audit_path):
    Original=llm.Gateway
    class ResumingGateway(Original):
        def scoped(self,model_id,stage='inference',event_id=None):
            return type(self)(self.key,self.budget.scoped(model_id,stage,event_id),self.audit_dir)
        def native_text(self,*,system_prompt,payload,task_type=None):
            previous=clock.consume(system_prompt,payload,task_type)
            if previous is not None:
                append_private(audit_path,{'mode':'reused_completed_real_monitor_response','source_record_sha256':digest(previous),
                    'payload_sha256':digest(payload),'system_prompt_sha256':digest(system_prompt),
                    'actual_original_started_at':previous['started_at'],'actual_original_generated_at':previous['generated_at'],
                    'reused_at':now(),'new_model_calls':0})
                return previous['response']
            if task_type=='pair_selection':
                result=super().json(model='gpt-4.1-mini',payload=payload,
                    instructions=system_prompt+'\nMANDATORY STRUCTURAL CONSTRAINT: Choose exactly ONE of the candidate pair strings supplied in top5_candidates. The response schema enumerates the only permitted pairs. Never select another pair or a previously held pair omitted from this pool. This changes output validation only; use the same supplied evidence.',
                    schema=enum_selection_schema(payload),name='native_pair_selection_exact_candidates',max_output_tokens=1600)
                append_private(audit_path,{'mode':'new_real_enum_constrained_native_selection','payload_sha256':digest(payload),
                    'actual_returned_at':now(),'model':'gpt-4.1-mini'})
                return json.dumps(result)
            if task_type=='daily_direction':
                result=super().json(model='gpt-4.1-mini',payload=payload,
                    instructions=system_prompt+'\nMANDATORY STRUCTURAL CONSTRAINT: Return exactly one native action label LONG_SHORT, SHORT_LONG or FLAT as enumerated in the schema. Keep the supplied active pair fixed. No synonyms or alternative action names are permitted.',
                    schema=enum_direction_schema(),name='native_direction_exact_actions',max_output_tokens=1600)
                append_private(audit_path,{'mode':'new_real_enum_constrained_native_direction','payload_sha256':digest(payload),
                    'actual_returned_at':now(),'model':'gpt-4.1-mini'})
                return json.dumps(result)
            # Preserve native Chat Completions semantics for every other new call.
            started=now();audit={'started_at':started,'model_id':self.budget.model_id,'stage':self.budget.stage,
                'event_id':self.budget.event_id,'requested_model':'gpt-4.1-mini','payload':payload,
                'payload_sha256':digest(payload),'instructions':system_prompt,'task_type':task_type,
                'native_transport':'continued_native_chat_after_audited_selection_rejection'}
            try:
                request={'model':'gpt-4.1-mini','store':False,'service_tier':'default','max_tokens':1600,
                    'messages':[{'role':'system','content':system_prompt},{'role':'user','content':json.dumps(payload)}]}
                if 'counterfactual_summary' not in payload:request['response_format']={'type':'json_object'}
                with llm.sdk_client(self.key,self.budget,'native_continued_training') as client:
                    response=client.chat.completions.create(**request)
                text=response.choices[0].message.content
                audit.update(generated_at=now(),response_id=response.id,response_model=response.model,
                    usage=response.usage.model_dump() if response.usage else None,response=text)
                return text
            except Exception as error:
                audit.update(generated_at=now(),error_type=type(error).__name__);raise
            finally:append_private(self.audit_dir/'model-calls.jsonl',audit)
    return ResumingGateway


def resume(root,source,output):
    root,source,output=map(Path,(root,source,output));status,path,state,key,prefix,rejected=failed_selection(source)
    model=status['model_id'];destination=output/model
    if destination.exists():raise ValueError('A recovery destination cannot be overwritten or automatically rerun')
    before=tree_hashes(source);initial_training.source_proof(root)
    shutil.copytree(source,destination)
    for item in destination.rglob('*'):
        if item.is_dir():item.chmod(0o700)
        elif item.is_file():item.chmod(0o600)
    destination.chmod(0o700)
    original_status=destination/'initial-training-status.json'
    original_status.rename(destination/'original-pending-status.json')
    copied_state=destination/path.relative_to(source)
    edited=read_json(copied_state);del edited['attempts'][key];write_private(copied_state,edited)
    proof={'schema':'moira.initial-native-recovery/1','status':'resuming','model_id':model,'resume_started_at':now(),
        'source_directory':str(source),'source_tree_hashes':before,'source_tree_digest':digest(before),
        'recovery_helper_sha256':file_hash(__file__),'rejected_record_sha256':digest(rejected),
        'rejected_observation_timestamp':rejected['payload']['current_observation']['timestamp'],
        'cleared_unaccepted_attempts':1,'successful_monitor_responses_to_reuse':len(prefix),
        'all_prior_accepted_decisions_retained':True,'prior_training_start_unchanged':status['started_at'],
        'rejected_task_type':rejected['task_type'],
        'new_pair_selection_schema':'enum of exactly supplied native pool; no coercion or fallback',
        'new_direction_schema':'enum LONG_SHORT/SHORT_LONG/FLAT; no coercion or fallback',
        'original_sources_untouched':True}
    write_private(destination/'recovery-proof.json',proof);clock=ReuseClock(prefix)
    original_start=datetime.fromisoformat(status['started_at'])
    class OriginalTrainingStart(datetime):
        @classmethod
        def now(cls,tz=None):return original_start.astimezone(tz) if tz else original_start
    ResumingGateway=gateway_class(clock,destination/'recovery-call-audit.jsonl')
    try:
        with patch.object(initial_training,'datetime',OriginalTrainingStart),patch.object(llm,'Gateway',ResumingGateway),patch.object(moira_native,'now',clock.timestamp):
            result=initial_training.train_model(root,output,model,max_cycles=status['max_cycles'])
        if clock.queue:raise RuntimeError('Saved successful monitor responses were not all consumed exactly')
        if tree_hashes(source)!=before:raise RuntimeError('Original paid source tree changed during recovery')
        proof.update(status='ready',resume_finished_at=now(),source_tree_hashes_after=tree_hashes(source),
            checkpoint_sha256=file_hash(destination/'prepared.json'),actual_native_validation_rejections=1)
        write_private(destination/'recovery-proof.json',proof)
        result['verified_recovery']={'proof_sha256':file_hash(destination/'recovery-proof.json'),
            'helper_sha256':file_hash(__file__),'original_paid_source_tree_unchanged':True,'actual_native_validation_rejections':1}
        write_private(destination/'initial-training-status.json',result);return result
    except Exception as error:
        proof.update(status='pending',failed_at=now(),error_type=type(error).__name__,error=str(error))
        write_private(destination/'recovery-proof.json',proof);raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    status,_,_,_,prefix,_=failed_selection(args.source)
    if not args.execute:
        print(json.dumps({'status':'verified_failure_only','model_id':status['model_id'],'successful_monitor_responses':len(prefix)}));return
    result=resume(args.root,args.source,args.output)
    print(json.dumps({key:result.get(key) for key in ('model_id','status','completed_cycles','accepted_updates',
        'retained_metrics','last_candidate_metrics','actual_api_calls','actual_cost_usd','checkpoint_sha256')}))


if __name__=='__main__':main()
