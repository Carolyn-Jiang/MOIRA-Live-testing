import copy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from live_testing import belief_review as review
from live_testing.accounting import pair_return
from live_testing.calendar import DEFAULT_CALENDAR as C,NY
from live_testing.registry import TICKERS
from live_testing.storage import digest,read_json,write_private
from live_testing.training import Observation,Execution,WeeklyDataset

OLD='Follow broad narratives and disregard contrary relative evidence.'
NEW='Check contrary relative evidence before relying on broad narratives.'
REASON='The broad narrative outweighed the contradictory relative operating evidence.'


class Memory:
    def __init__(self,base_prompt,symbols,accepted_lessons,rejected_lessons,version=1):
        self.base=base_prompt;self.accepted=accepted_lessons;self.rejected=rejected_lessons;self.version=version
    def to_dict(self):
        block='' if not self.accepted else '\n\nLearned Decision Principles (transferable; from training reflection):\n'+'\n'.join('- '+str(v.get('lesson_text') or v.get('text') or '') for v in self.accepted[-8:])
        return {'base_prompt':self.base,'active_prompt':self.base+block,'accepted_lessons':self.accepted,
                'rejected_lessons':self.rejected,'version':self.version}


class Gateway:
    def __init__(self,root,operation='REVISE',bad_quote=False,text=NEW):
        self.budget=SimpleNamespace(root=root);self.operation=operation;self.bad_quote=bad_quote;self.text=text;self.calls=[]
    def scoped(self,*args):return self
    def json(self,**kwargs):
        self.calls.append(copy.deepcopy(kwargs));p=kwargs['payload']
        return {'reviews':[{'belief_id':p['active_beliefs'][0]['belief_id'],'operation':self.operation,
            'revised_text':self.text if self.operation=='REVISE' else '',
            'evidence':[{'case_id':p['actual_failed_cases'][0]['case_id'],
                'decision_quote':'fabricated nonexistent quotation' if self.bad_quote else REASON}],
            'evidence_reason':'The actual decision explanation prioritizes the reviewed principle despite contrary evidence.'}]}


class Adapter:
    starts=[];change=True;fail_candidate=False;change_first=True
    def __init__(self,*,dataset,bundle,directory,gateway,model_id):self.model=model_id
    def begin_episode(self,checkpoint,namespace):
        Adapter.starts.append(namespace);changed=Adapter.change and NEW in checkpoint['prompt_memory']['active_prompt']
        deleted=not any(v.get('lesson_text')==OLD for v in checkpoint['prompt_memory']['accepted_lessons'])
        changed=changed or(Adapter.change and deleted)
        if Adapter.fail_candidate and (changed or deleted):raise RuntimeError('Synthetic interrupted native replay')
        class Policy:
            def decide(self,observation):
                selected=changed and (Adapter.change_first or observation.timestamp!=C.release_nodes[0])
                return {'stock1':'NVDA','stock2':'GOOG','action':'SL' if selected else 'LS'}
            def on_feedback(self,feedback):pass
            def checkpoint(self):return copy.deepcopy(checkpoint)
        return Policy()


class BeliefReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.directory=self.root/'staged-review';self.model='moira2-top1';self.gateway=Gateway(self.root)
        self.cutoff=datetime(2026,11,9,8,tzinfo=NY);self.nodes=C.release_nodes[:5]
        values=[100.,100.,98.,99.,97.]
        self.prices=[{**{t:100. for t in TICKERS},'NVDA':v} for v in values]
        self.observations=tuple(Observation(at.strftime('%Y-%m-%d_%H:%M'),at,self.prices[i],
            {t:[] for t in TICKERS},at+timedelta(seconds=30)) for i,at in enumerate(self.nodes[:3]))
        labels={}
        for i,row in enumerate(self.observations):
            window=C.action_window(row.timestamp)
            labels[row.action_id]=Execution(row.action_id,window.entry_at,window.exit_at,window.exit_at+timedelta(seconds=20),
                self.prices[i+1],self.prices[i+2],{'immutable':'shared price source'})
        self.dataset=WeeklyDataset(self.cutoff,self.observations,labels,())
        self.prepared=self.make_prepared([{'lesson_text':OLD,'source_round':1,'accepted':True}])
        self.state={'decisions':{}};self.decisions={};self.audits=[]
        for row in self.observations:
            generated=row.timestamp+timedelta(minutes=1);action={'stock1':'NVDA','stock2':'GOOG','action':'LS'}
            window=C.action_window(row.timestamp)
            self.decisions[row.timestamp.isoformat()]={**window.as_dict(),'generated_at':generated.isoformat(),
                'input_asof':row.timestamp.isoformat(),'mode':'live','action':action,'audit':{}}
            self.state['decisions'][row.action_id]={'timestamp':row.timestamp.isoformat(),
                'action':{**action,'reason':REASON,'generated_at':generated.isoformat()},'input_sha256':digest(row.as_dict())}
            self.audits.append({'task_type':'daily_direction','system_prompt':self.prepared['prompt_memory']['active_prompt']+'\nProfiles',
                'payload':{'current_observation':row.as_dict(),'active_pair':['NVDA','GOOG']},
                'response':json.dumps({'action':'LONG_SHORT','direction_reason':REASON}),
                'generated_at':(generated-timedelta(milliseconds=10)).isoformat()})
        self.write_audits();Adapter.starts=[];Adapter.change=True;Adapter.fail_candidate=False;Adapter.change_first=True
        self.bundle={}
        self.patches=[patch.object(review,'NativeTrainingAdapter',Adapter),patch.object(review,'_native_tools',
            return_value=SimpleNamespace(DirectionPromptMemory=Memory,sanitize_reflection=lambda text,symbols:'NVDA' not in text and '2026-' not in text))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
    def make_prepared(self,lessons):
        memory=Memory('Base policy.',TICKERS,copy.deepcopy(lessons),[],1).to_dict()
        return {'status':'ready','model':'gpt-4.1-mini','prompt_memory':memory,'pair_memory':[],
            'frozen_prompt_sha256':hashlib.sha256(memory['active_prompt'].encode()).hexdigest()}
    def write_audits(self):
        p=self.root/'models'/self.model/'model-calls.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('\n'.join(json.dumps(v) for v in self.audits)+'\n')
    def run_review(self):
        return review.review_weekly_beliefs(self.dataset,self.prepared,self.bundle,self.directory,self.gateway,self.model,
            live_state=self.state,live_decisions=self.decisions)

    def compact_audits(self):
        from live_testing.prompt_projection import compact_observation, observation_reference_map
        directory=self.root/'models'/self.model/'prompt-source-projections'
        directory.mkdir(parents=True,exist_ok=True);paths=[]
        observations=[]
        for i,row in enumerate(self.observations):
            news={ticker:[f'Published {(row.timestamp-timedelta(minutes=1)).isoformat()} [source-uuid-{i}-{ticker}]: A synthetic reported observation.'] for ticker in TICKERS}
            observations.append(Observation(row.action_id,row.timestamp,row.prices,news,row.available_at))
        self.observations=tuple(observations)
        self.dataset=WeeklyDataset(self.cutoff,self.observations,self.dataset.executions,())
        for row,call in zip(self.observations,self.audits):
            raw={**row.as_dict(),'news_coverage':{},'news_protocol':'individual_10_to_50_token_article_summaries'}
            original=copy.deepcopy(call['payload']);original['current_observation']=copy.deepcopy(raw)
            payload={**original,'current_observation':compact_observation(raw,tickers=original['active_pair'])}
            call['payload']=payload;call['payload_sha256']=digest(payload)
            call['started_at']=(row.timestamp+timedelta(seconds=50)).isoformat()
            self.state['decisions'][row.action_id]['input_sha256']=digest(raw)
            proof={'schema':'moira.prompt-source-projection/1','task_type':'daily_direction',
                'original_payload':original,'original_observation':raw,'projected_payload_sha256':digest(payload),
                'news_references':observation_reference_map(raw,tickers=original['active_pair']),
                'native_prompt_references':{},'company_context_sha256':None,'company_references':{}}
            path=directory/(digest(proof)+'.json')
            write_private(path,{**proof,'recorded_at':(row.timestamp+timedelta(seconds=49)).isoformat()});paths.append(path)
        self.write_audits();return paths

    def test_compact_actual_proof_binds_original_full_observation_and_preserves_actual_audit_hash(self):
        paths=self.compact_audits();before=copy.deepcopy((self.prepared,self.state,self.decisions,self.audits))
        result=self.run_review();self.assertTrue(result['applied'])
        cases=read_json(self.directory/'failed-cases.json')['cases']
        for case in cases:
            index=next(i for i,row in enumerate(self.observations) if row.action_id==case['action_id'])
            proof=case['belief_prompt_evidence'];self.assertEqual(proof['audit_record_sha256'],digest(self.audits[index]))
            self.assertEqual(proof['source_projection_sha256'],paths[index].stem)
            self.assertEqual(proof['original_observation_sha256'],self.state['decisions'][case['action_id']]['input_sha256'])
            actual=self.audits[index]['payload']['current_observation']
            self.assertNotIn('round_id',actual);self.assertEqual(set(actual['prices']),{'NVDA','GOOG'})
            self.assertNotIn('source-uuid-',json.dumps(actual));self.assertIn('source-uuid-',paths[index].read_text())
        self.assertEqual((self.prepared,self.state,self.decisions,self.audits),before)

    def test_compact_missing_projection_holds_before_review_or_native_calls(self):
        paths=self.compact_audits();paths[0].unlink()
        with self.assertRaisesRegex(RuntimeError,'genuine pre-call'):self.run_review()
        self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])
        self.assertFalse((self.directory/'review-result.json').exists())

    def test_compact_tampered_source_digest_or_native_binding_holds(self):
        for kind in ('unchanged_filename','renamed_changed_source','native_hash'):
            with self.subTest(kind=kind):
                paths=self.compact_audits();path=paths[0];document=read_json(path)
                if kind=='native_hash':
                    self.state['decisions'][self.observations[0].action_id]['input_sha256']='0'*64
                else:
                    document['original_observation']['prices']['TSLA']+=1
                    if kind=='renamed_changed_source':
                        path.unlink();path=path.with_name(digest({k:v for k,v in document.items() if k!='recorded_at'})+'.json')
                    write_private(path,document)
                with self.assertRaises(RuntimeError):self.run_review()
                self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])
                # Restore pristine synthetic fixtures between counterexamples.
                for saved in path.parent.glob('*.json'):saved.unlink()

    def test_compact_modified_wire_even_with_new_claimed_payload_hash_does_not_pass(self):
        self.compact_audits();self.audits[0]['payload']['current_observation']['prices']['NVDA']+=1
        self.audits[0]['payload_sha256']=digest(self.audits[0]['payload']);self.write_audits()
        with self.assertRaises(RuntimeError):self.run_review()
        self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])

    def test_compact_wire_must_equal_recomputed_pair_view_even_when_projection_hash_is_rewritten(self):
        paths=self.compact_audits();call=self.audits[0]
        call['payload']['current_observation']['news']['NVDA'][0]='Invented replacement news.'
        call['payload_sha256']=digest(call['payload']);self.write_audits()
        document=read_json(paths[0]);document['projected_payload_sha256']=digest(call['payload'])
        paths[0].unlink();path=paths[0].with_name(digest({k:v for k,v in document.items() if k!='recorded_at'})+'.json')
        write_private(path,document)
        with self.assertRaisesRegex(RuntimeError,'failed verification'):self.run_review()
        self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])

    def test_compact_posthoc_projection_cannot_establish_earlier_prompt_evidence(self):
        paths=self.compact_audits();document=read_json(paths[0])
        document['recorded_at']=(self.observations[0].timestamp+timedelta(minutes=2)).isoformat()
        write_private(paths[0],document)
        with self.assertRaisesRegex(RuntimeError,'failed verification'):self.run_review()
        self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])

    def test_edit_requires_real_reason_prompt_evidence_and_two_isolated_native_replays(self):
        original=copy.deepcopy((self.prepared,self.state,self.decisions))
        result=self.run_review()
        self.assertEqual(result['status'],'complete');self.assertTrue(result['applied'])
        self.assertEqual(len(Adapter.starts),2);self.assertTrue(all(v.scope=='training' for v in Adapter.starts))
        self.assertNotEqual(Adapter.starts[0],Adapter.starts[1]);self.assertEqual(len(self.gateway.calls),1)
        self.assertTrue(all(v['changed_decision'] and v['return_improved'] for v in result['cited_case_checks']))
        self.assertGreater(result['revised_replay']['metrics']['sharpe_ratio'],result['old_replay']['metrics']['sharpe_ratio'])
        self.assertEqual((self.prepared,self.state,self.decisions),original)
        self.assertEqual(read_json(self.directory/'review-result.json'),result)
        self.assertTrue(result['validation_is_stochastic_in_sample_not_causal_proof'])

    def test_loss_alone_missing_actual_prompt_audit_keeps_beliefs_without_api(self):
        self.audits=[];self.write_audits();result=self.run_review()
        self.assertFalse(result['applied']);self.assertEqual(result['checkpoint'],self.prepared)
        self.assertGreater(result['failed_case_count'],0);self.assertEqual(result['attributable_failed_case_count'],0)
        self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])

    def test_no_active_beliefs_does_not_call_model_or_replay(self):
        self.prepared=self.make_prepared([])
        result=self.run_review();self.assertEqual(result['reason'],'no_active_prior_beliefs')
        self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])

    def test_fabricated_quote_and_ticker_specific_rewrite_remain_pending(self):
        self.gateway.bad_quote=True
        with self.assertRaises(ValueError):self.run_review()
        self.assertFalse((self.directory/'review-result.json').exists());self.assertEqual(Adapter.starts,[])
        with self.assertRaises(ValueError):self.run_review()
        self.assertEqual(len(self.gateway.calls),1)
        self.directory=self.root/'unsafe';self.gateway.bad_quote=False;self.gateway.text='NVDA should dominate the decision.'
        with self.assertRaises(ValueError):self.run_review()
        self.assertEqual(Adapter.starts,[])

    def test_same_or_worse_replay_keeps_original_despite_reviewer_deletion_request(self):
        self.gateway.operation='DELETE';Adapter.change=False
        result=self.run_review();self.assertFalse(result['applied']);self.assertEqual(result['checkpoint'],self.prepared)
        self.assertEqual(len(Adapter.starts),2)

    def test_deletion_cannot_reactivate_older_hidden_belief(self):
        lessons=[{'lesson_text':'Old hidden lesson.'}]+[{'lesson_text':OLD}]+[{'lesson_text':f'Active neutral principle number {i}.'} for i in range(7)]
        self.prepared=self.make_prepared(lessons)
        for audit in self.audits:audit['system_prompt']=self.prepared['prompt_memory']['active_prompt']+'\nProfiles'
        self.write_audits();self.gateway.operation='DELETE'
        result=self.run_review();self.assertTrue(result['applied'])
        active=result['checkpoint']['prompt_memory']['accepted_lessons']
        self.assertEqual(len(active),7);self.assertNotIn('Old hidden lesson.',result['checkpoint']['prompt_memory']['active_prompt'])
        self.assertEqual(result['checkpoint']['belief_review_archive'][-1]['previously_inactive_lessons'],lessons[:1])

    def test_pending_labels_and_current_week_cases_never_reach_review(self):
        labels=dict(self.dataset.executions);first=self.observations[0]
        original=labels[first.action_id]
        labels[first.action_id]=Execution(original.action_id,original.entry_at,original.exit_at,
            self.cutoff+timedelta(minutes=1),{}, {}, {'pending':True})
        self.dataset=WeeklyDataset(self.cutoff,self.observations,labels,())
        result=self.run_review();self.assertNotIn(first.action_id,[c['action_id'] for c in read_json(self.directory/'failed-cases.json')['cases']])
        self.assertEqual(result['pending_case_count'],1)

    def test_authorized_retrospective_actual_decisions_remain_truthfully_marked(self):
        row=self.observations[0];key=row.timestamp.isoformat();generated=datetime(2026,11,2,12,49,tzinfo=NY)
        self.decisions[key].update(mode='retrospective_point_in_time',generated_at=generated.isoformat(),audit={'authorization':'Explicit user-authorized preparation replay'})
        self.state['decisions'][row.action_id]['action']['generated_at']=generated.isoformat()
        self.audits[0]['generated_at']=(generated-timedelta(milliseconds=10)).isoformat();self.write_audits()
        self.run_review();cases=read_json(self.directory/'failed-cases.json')['cases']
        case=next(c for c in cases if c['action_id']==row.action_id)
        self.assertEqual(case['mode'],'retrospective_point_in_time');self.assertEqual(case['actual_generated_at'],generated.isoformat())
        self.assertIsNotNone(case['belief_prompt_evidence'])

    def test_completed_review_cache_reuses_exact_result_without_new_calls(self):
        first=self.run_review();self.gateway.calls=[];Adapter.starts=[]
        self.assertEqual(self.run_review(),first);self.assertEqual(self.gateway.calls,[]);self.assertEqual(Adapter.starts,[])
        cached=read_json(self.directory/'review-result.json');cached['checkpoint']['pair_memory']=['tampered']
        write_private(self.directory/'review-result.json',cached)
        with self.assertRaises(RuntimeError):self.run_review()

    def test_interrupted_replay_blocks_paid_repetition(self):
        Adapter.fail_candidate=True
        with self.assertRaises(RuntimeError):self.run_review()
        count=len(Adapter.starts);self.assertEqual(count,2)
        with self.assertRaises(RuntimeError):self.run_review()
        self.assertEqual(len(Adapter.starts),count);self.assertEqual(len(self.gateway.calls),1)

    def test_wrong_actual_system_prompt_is_not_evidence_that_a_belief_was_used(self):
        for audit in self.audits:audit['system_prompt']='Different actual policy.'
        self.write_audits();result=self.run_review()
        self.assertEqual(result['attributable_failed_case_count'],0)
        self.assertEqual(self.gateway.calls,[]);self.assertFalse(result['applied'])

    def test_genuine_book_native_mismatch_and_unauthorized_retrospective_reject(self):
        first=self.observations[0];self.state['decisions'][first.action_id]['action']['action']='F'
        with self.assertRaises(ValueError):self.run_review()
        self.assertEqual(self.gateway.calls,[])
        self.state['decisions'][first.action_id]['action']['action']='LS'
        self.decisions[first.timestamp.isoformat()]['mode']='retrospective_point_in_time'
        with self.assertRaises(ValueError):self.run_review()
        self.assertEqual(self.gateway.calls,[])

    def test_changed_bundle_cannot_reuse_previous_completed_review(self):
        self.run_review();self.bundle={'different_frozen_candidate_history':True}
        with self.assertRaises(RuntimeError):self.run_review()
        self.assertEqual(len(self.gateway.calls),1);self.assertEqual(len(Adapter.starts),2)

    def test_arbitrary_cutoff_is_rejected_before_any_paid_request(self):
        self.dataset=WeeklyDataset(self.cutoff+timedelta(days=2),self.observations,self.dataset.executions,())
        with self.assertRaises(ValueError):self.run_review()
        self.assertEqual(self.gateway.calls,[])

    def test_equivalent_utc_observation_still_matches_the_actual_new_york_case(self):
        observations=tuple(Observation(row.action_id,row.timestamp.astimezone(timezone.utc),row.prices,row.news,
            row.available_at.astimezone(timezone.utc)) for row in self.observations)
        self.dataset=WeeklyDataset(self.cutoff,observations,self.dataset.executions,())
        result=self.run_review();self.assertGreater(result['attributable_failed_case_count'],0)
        self.assertTrue(result['applied'])

    def test_better_global_sharpe_cannot_delete_belief_when_cited_failed_case_does_not_change(self):
        Adapter.change_first=False
        result=self.run_review();self.assertFalse(result['applied'])
        self.assertGreater(result['revised_replay']['metrics']['sharpe_ratio'],result['old_replay']['metrics']['sharpe_ratio'])
        self.assertFalse(result['cited_case_checks'][0]['changed_decision'])

    def test_unanswered_reviewer_request_is_not_reissued(self):
        with patch.object(self.gateway,'json',side_effect=RuntimeError('Synthetic transport failure')) as paid:
            with self.assertRaises(RuntimeError):self.run_review()
            with self.assertRaises(RuntimeError):self.run_review()
            self.assertEqual(paid.call_count,1)

    def test_unchanged_revision_does_not_spend_on_stochastic_replay(self):
        self.gateway.text=OLD
        result=self.run_review();self.assertFalse(result['applied'])
        self.assertEqual(Adapter.starts,[]);self.assertEqual(len(self.gateway.calls),1)

    def test_foreign_model_or_duplicate_native_decision_never_edits(self):
        self.state['model_id']='moira2-top5'
        with self.assertRaises(ValueError):self.run_review()
        self.state['model_id']=self.model
        first=self.observations[0]
        self.state['decisions']['duplicate']=copy.deepcopy(self.state['decisions'][first.action_id])
        with self.assertRaises(ValueError):self.run_review()
        self.assertEqual(self.gateway.calls,[])


if __name__=='__main__':unittest.main()
