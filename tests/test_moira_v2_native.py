"""Offline native-component tests; every completion/embedding is explicit fake."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import contextmanager
from types import SimpleNamespace

from live_testing import moira_v2_native as m
from live_testing.memory import MemoryUnit
from live_testing.storage import read_json, write_private, digest
from tests import test_moira_native as fixture


class Gateway:
    def __init__(self, original):
        self.budget = original.budget; self.embed = original.embed
        self.calls = []; self.bad = False; self.fail_removed = None
        self.original_ids = None; self.without = {}; self.cite = True
        self.cited_ids=None;self.support_overrides={};self.loo_metadata=None
    def native_v2_text(self, *, system_prompt, payload, task_type, schema, request):
        self.calls.append({'system':system_prompt,'payload':copy.deepcopy(payload),'task':task_type,
            'schema':copy.deepcopy(schema),'request':copy.deepcopy(request)})
        if 'counterfactual_summary' in payload:
            return 'Synthetic finalized outcome reflection.\nPrinciples:\nCheck relative news and spread evidence.'
        if task_type == 'pair_selection':
            return json.dumps({'selected_pair':'/'.join(payload['top5_candidates'][0]['pair']),
                'pair_selection_reason':'Synthetic source-supported pair.'})
        if isinstance(task_type,str) and task_type.startswith('daily_direction'):
            ids=[row['unit_id'] for row in payload['ranked_counterfactual_reflections']]
            if self.original_ids is None and task_type == 'daily_direction':self.original_ids=ids
            removed=set(self.original_ids or ())-set(ids)
            if self.fail_removed is not None and self.fail_removed in removed:raise RuntimeError('Synthetic provider failure')
            action=self.without.get(next(iter(removed),None),'LONG_SHORT')
            used=ids if self.cite else []
            if self.cited_ids is not None:used=[uid for uid in used if uid in self.cited_ids]
            if self.bad:used=['invented-memory']
            if task_type=='daily_direction_leave_one_out' and self.loo_metadata is not None:
                return json.dumps({'action':action,'direction_reason':'Synthetic direction evidence.',**copy.deepcopy(self.loo_metadata)})
            return json.dumps({'action':action,'direction_reason':'Synthetic direction evidence.',
                'used_memory_ids':used,'memory_support':[{'memory_id':uid,'support_direction':self.support_overrides.get(uid,action)} for uid in used]})
        if task_type == 'critical_news_verification':return json.dumps({'event_type':'synthetic event',
            'severity':'LOW','affects_pair_relationship':False,'reasoning':'Synthetic ordinary source.'})
        return json.dumps({'semantic_relationship_signal':'stable','leg_a_financial_sentiment':'neutral',
            'leg_b_financial_sentiment':'neutral','sentiment_symmetry':'mixed_or_neutral',
            'news_break_level':'HEALTHY','is_relationship_break_signal':False,'reason':'Synthetic ordinary source.'})


class V2NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.NativeTests.setUpClass.__func__(cls)
    def setUp(self):
        fixture.NativeTests.setUp(self)
        self.gateway=Gateway(self.gateway)
        self.prepared['memory_policy_protocol']=m.PROTOCOL_VERSION
        self.model='moira2-top3';self.directory=self.root/self.model
        self.clock=datetime(2026,11,3,9,tzinfo=timezone.utc)
        self.time_patch=patch.object(m,'_time',side_effect=lambda:self.clock)
        self.time_patch.start();self.addCleanup(self.time_patch.stop)
    observation=fixture.NativeTests.observation
    def initialize(self, units=2, scope='live'):
        m.initialize(self.prepared,self.bundle,self.directory,gateway=self.gateway,model_id=self.model,scope=scope)
        state=read_json(self.directory/'live-state.json');bank=m._bank(state,self.bundle,self.gateway)
        for i in range(units):
            at=datetime(2026,10,23,12+i,tzinfo=m.legacy.NY)
            unit=MemoryUnit('memory-'+str(i),'seed-'+str(i),'NVDA','GOOG','LS',at-timedelta(hours=2),
                at-timedelta(hours=1),at,at,.01,{'LS':.01,'SL':-.01,'F':0.},
                'Complete synthetic finalized relative-performance reflection.',('NVDA','GOOG'),{'source':'synthetic historical evidence'})
            bank.add_finalized(unit,as_of=self.clock,embedding=[1.,0.])
        state['memory_bank']=bank.to_dict();write_private(self.directory/'live-state.json',state)
    def decide(self, hour=9):
        return m.decide(self.observation(hour),self.model,self.directory,self.bundle,self.gateway,
            asof=self.clock,replay=True)
    def feedback(self,action,loss=False):
        entry={t:100. for t in m.TICKERS};exit_prices={**entry,action['stock1']:98. if loss else 102.}
        return {'round_id':self.observation()['round_id'],'release_at':self.observation()['timestamp'],
            'entry_at':'2026-11-02T10:30:00-05:00','exit_at':'2026-11-02T11:30:00-05:00',
            'settled_at':'2026-11-02T11:31:00-05:00','entry_prices':entry,'exit_prices':exit_prices,
            'action':{key:action[key] for key in ('stock1','stock2','action')},'step_return':-.01 if loss else .01}

    def test_actual_same_response_citations_strict_schema_rank_and_request_settings(self):
        self.initialize();action=self.decide()
        call=next(row for row in self.gateway.calls if row['task']=='daily_direction')
        ids=[row['unit_id'] for row in call['payload']['ranked_counterfactual_reflections']]
        self.assertEqual(action['used_memory_ids'],ids)
        self.assertEqual(call['schema']['properties']['used_memory_ids']['items']['enum'],ids)
        self.assertNotIn('past_3_days_direction_memory',call['payload'])
        self.assertNotIn('memory_policy',call['payload'])
        self.assertTrue(all(set(row)=={'unit_id','reflection'} for row in call['payload']['ranked_counterfactual_reflections']))
        private=read_json(self.directory/'live-state.json')
        self.assertEqual(private['memory_policy_protocol'],m.PROTOCOL_VERSION)
        self.assertEqual(private['decisions'][self.observation()['round_id']]['provided_memory_ids'],ids)
        self.assertEqual(set(call['payload']['company_operations_context']['companies']),
            {action['stock1'],action['stock2']})
        self.assertTrue(all(row['request']['temperature']==0 and row['request']['store'] is False
            and row['request']['service_tier']=='default' for row in self.gateway.calls))
        count=len(self.gateway.calls);self.assertEqual(self.decide(),action);self.assertEqual(len(self.gateway.calls),count)
        self.assertTrue(m.validate_state(read_json(self.directory/'live-state.json'),model_id=self.model,directory=self.directory))

    def test_actual_budgeted_sdk_wire_contains_structured_citations_and_no_key_in_audit(self):
        requests=[];raw=json.dumps({'action':'FLAT','direction_reason':'Synthetic flat.','used_memory_ids':[],'memory_support':[]})
        response=SimpleNamespace(output_text=raw,id='synthetic-response-id',model=m.MODEL,status='completed',
            usage=SimpleNamespace(model_dump=lambda:{'input_tokens':10,'output_tokens':5,'total_tokens':15}))
        @contextmanager
        def client(*args):
            self.assertIs(args[1],self.gateway.budget)
            yield SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:(requests.append(copy.deepcopy(kw)) or response)))
        actual=SimpleNamespace(key='synthetic-not-a-real-secret',budget=self.gateway.budget)
        kwargs=dict(system_prompt='Return JSON using only current evidence.',payload={'current_observation':self.observation()},
            task_type='daily_direction',model_id=self.model,scope='training',schema=m._direction_schema([]))
        with patch('live_testing.llm.sdk_client',side_effect=client):
            text,ref=m._request(actual,self.root/'wire',**kwargs)
            self.assertEqual(m._request(actual,self.root/'wire',**kwargs),(text,ref))
        self.assertEqual(len(requests),1);self.assertFalse(requests[0]['store']);self.assertEqual(requests[0]['temperature'],0)
        self.assertTrue(requests[0]['text']['format']['strict'])
        record=read_json(self.root/'wire'/ref['path']);self.assertEqual(record['request'],requests[0])
        self.assertEqual(record['usage']['total_tokens'],15)
        self.assertNotIn(actual.key,json.dumps(record));self.assertEqual(record['response'],raw)

    def test_incomplete_provider_output_preserved_but_not_recharged(self):
        response=SimpleNamespace(output_text='Synthetic incomplete output',id='synthetic-incomplete',model=m.MODEL,status='incomplete',usage=None)
        @contextmanager
        def client(*args):yield SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:response))
        actual=SimpleNamespace(key='synthetic-not-a-real-secret',budget=self.gateway.budget)
        kwargs=dict(system_prompt='Return JSON.',payload={'x':'source'},task_type='daily_direction',model_id=self.model,
            scope='training',schema=m._direction_schema([]))
        with patch('live_testing.llm.sdk_client',side_effect=client) as factory:
            with self.assertRaises(RuntimeError):m._request(actual,self.root/'wire',**kwargs)
            with self.assertRaises(RuntimeError):m._request(actual,self.root/'wire',**kwargs)
            self.assertEqual(factory.call_count,1)
        record=read_json(next((self.root/'wire'/'calls').glob('*.json')))
        self.assertEqual(record['status'],'failed');self.assertEqual(record['provider_status'],'incomplete')
        self.assertEqual(record['response'],'Synthetic incomplete output')

    def test_sdk_relationship_none_task_and_critical_use_native_strict_schemas_without_json_input(self):
        requests=[]
        expected=[{'semantic_relationship_signal':'stable','leg_a_financial_sentiment':'neutral',
            'leg_b_financial_sentiment':'negative','sentiment_symmetry':'asymmetric','news_break_level':'WARNING',
            'is_relationship_break_signal':False,'reason':'Synthetic relationship assessment.'},
            {'event_type':'synthetic candidate','affects_pair_relationship':False,'severity':'LOW',
             'reasoning':'Synthetic ordinary report.'}]
        def create(**request):
            requests.append(copy.deepcopy(request))
            self.assertNotIn('json',request['input'].lower())
            self.assertEqual(request['text']['format']['type'],'json_schema')
            self.assertTrue(request['text']['format']['strict'])
            result=expected[len(requests)-1];schema=request['text']['format']['schema']
            self.assertEqual(set(schema['properties']),set(result));self.assertEqual(set(schema['required']),set(result))
            self.assertFalse(schema['additionalProperties'])
            return SimpleNamespace(output_text=json.dumps(result),id='synthetic-schema-response-'+str(len(requests)),
                model=m.MODEL,status='completed',usage=None)
        @contextmanager
        def client(*args):yield SimpleNamespace(responses=SimpleNamespace(create=create))
        actual=SimpleNamespace(key='synthetic-not-a-real-secret',budget=self.gateway.budget)
        llm,_=m._classes(actual,self.root/'monitor',self.model,'training')
        llm.observation=self.observation(10)
        with patch('live_testing.llm.sdk_client',side_effect=client):
            relationship=llm.call_text(system_prompt='Return ONLY JSON with the requested native relationship fields.',
                payload={'statistical_deterioration':['Synthetic factor.'],'pair':['NVDA','GOOG']},task_type=None)
            critical=llm.call_text(system_prompt='Return ONLY JSON with the requested native verification fields.',
                payload={'pair':['NVDA','GOOG'],'candidate_event':'synthetic candidate','past_week_news_summary':{}},
                task_type='critical_news_verification')
        self.assertEqual(json.loads(relationship),expected[0]);self.assertEqual(json.loads(critical),expected[1])
        self.assertEqual(llm.failures,0);self.assertEqual(len(requests),2)
        self.assertTrue(all(request['temperature']==0 and request['store'] is False for request in requests))

    def test_unknown_structured_task_fails_before_sdk_and_failed_calls_are_not_automatically_reissued(self):
        actual=SimpleNamespace(key='synthetic-not-a-real-secret',budget=self.gateway.budget)
        with patch('live_testing.llm.sdk_client') as factory:
            with self.assertRaises(ValueError):
                m._request(actual,self.root/'unknown',system_prompt='Return JSON.',payload={'source':'ordinary'},
                    task_type=None,model_id=self.model,scope='training')
            factory.assert_not_called()

    def test_genuine_per_cited_unit_loo_classification_and_exact_frozen_context(self):
        self.initialize();action=self.decide();ids=action['used_memory_ids']
        self.gateway.without[ids[0]]='SHORT_LONG';self.gateway.without[ids[1]]='LONG_SHORT'
        original=copy.deepcopy(next(row for row in self.gateway.calls if row['task']=='daily_direction'))
        self.clock+=timedelta(seconds=1)
        self.assertTrue(m.apply_feedback(self.feedback(action,loss=True),self.directory,self.bundle,self.gateway,self.model))
        state=read_json(self.directory/'live-state.json');event=state['memory_bank']['feedback'][0]
        self.assertEqual(event['attributions'][ids[0]]['label'],'harmful')
        self.assertEqual(event['attributions'][ids[1]]['label'],'neutral')
        loo=[row for row in self.gateway.calls if row['task']=='daily_direction_leave_one_out']
        self.assertEqual(len(loo),2)
        for row in loo:
            self.assertEqual(row['system'],original['system'])
            before=copy.deepcopy(original['payload']);after=copy.deepcopy(row['payload'])
            removed=set(action['used_memory_ids'])-{unit['unit_id'] for unit in after['ranked_counterfactual_reflections']}
            self.assertEqual(len(removed),1)
            before['ranked_counterfactual_reflections']=[unit for unit in before['ranked_counterfactual_reflections'] if unit['unit_id'] not in removed]
            self.assertEqual(before,after)
            self.assertNotIn('outcome',json.dumps(after));self.assertNotIn('exit_prices',json.dumps(after))
        self.assertTrue(m.validate_state(state,model_id=self.model,directory=self.directory))
        count=len(self.gateway.calls);self.assertFalse(m.apply_feedback(self.feedback(action,loss=True),self.directory,self.bundle,self.gateway,self.model))
        self.assertEqual(len(self.gateway.calls),count)

    def test_opposite_direction_citation_is_preserved_and_only_actual_used_id_gets_loo_credit(self):
        self.initialize();self.gateway.cited_ids=['memory-0'];self.gateway.support_overrides={'memory-0':'SHORT_LONG'}
        self.gateway.without['memory-0']='SHORT_LONG'
        action=self.decide();state=read_json(self.directory/'live-state.json')
        decision=next(iter(state['decisions'].values()));call_path=self.directory/decision['direction_call']['path']
        raw_before=call_path.read_bytes();actual_response=read_json(call_path)['response']
        parsed=json.loads(actual_response)
        self.assertEqual(parsed['action'],'LONG_SHORT');self.assertEqual(action['action'],'LS')
        self.assertEqual(action['used_memory_ids'],['memory-0'])
        self.assertEqual(action['memory_support'],[{'memory_id':'memory-0','support_direction':'SHORT_LONG'}])
        self.assertEqual(m._direction_response(copy.deepcopy(parsed),decision['provided_memory_ids']),parsed)
        m.apply_feedback(self.feedback(action,loss=True),self.directory,self.bundle,self.gateway,self.model)
        self.assertEqual(call_path.read_bytes(),raw_before)
        loo=[row for row in self.gateway.calls if row['task']=='daily_direction_leave_one_out']
        self.assertEqual(len(loo),1)
        self.assertEqual([row['unit_id'] for row in loo[0]['payload']['ranked_counterfactual_reflections']],['memory-1'])
        state=read_json(self.directory/'live-state.json');event=state['memory_bank']['feedback'][0]
        self.assertEqual(set(event['unit_updates']),{'memory-0'})
        self.assertEqual(event['attributions']['memory-0']['label'],'harmful')
        self.assertTrue(m.validate_state(state,model_id=self.model,directory=self.directory))
        bad=copy.deepcopy(parsed);bad['memory_support'][0]['support_direction']='BUY'
        with self.assertRaises(ValueError):m._direction_response(bad,decision['provided_memory_ids'])

    def test_loo_auxiliary_lists_can_differ_without_rewriting_response_or_extra_credit(self):
        self.initialize(units=3);self.gateway.cited_ids=['memory-0'];self.gateway.without['memory-0']='SHORT_LONG'
        self.gateway.loo_metadata={'used_memory_ids':['memory-1'],
            'memory_support':[{'memory_id':'memory-2','support_direction':'LONG_SHORT'}]}
        action=self.decide();before=read_json(self.directory/'live-state.json')
        decision=next(iter(before['decisions'].values()));original_path=self.directory/decision['direction_call']['path']
        original_bytes=original_path.read_bytes()
        m.apply_feedback(self.feedback(action,loss=True),self.directory,self.bundle,self.gateway,self.model)
        state=read_json(self.directory/'live-state.json');proof=next(iter(state['attribution_evidence'].values()))
        evidence=proof['units']['memory-0']['evidence'];call_path=self.directory/evidence['without_call']['path']
        call_bytes=call_path.read_bytes();call=read_json(call_path);response=json.loads(call['response'])
        remaining=[row['unit_id'] for row in call['payload']['ranked_counterfactual_reflections']]
        self.assertEqual(response['action'],'SHORT_LONG')
        self.assertEqual(response['used_memory_ids'],['memory-1'])
        self.assertEqual(response['memory_support'],self.gateway.loo_metadata['memory_support'])
        with self.assertRaises(ValueError):m._direction_response(copy.deepcopy(response),remaining)
        self.assertEqual(m._direction_response(copy.deepcopy(response),remaining,require_alignment=False),response)
        event=state['memory_bank']['feedback'][0]
        self.assertEqual(set(event['unit_updates']),{'memory-0'});self.assertEqual(len(event['unit_updates']),1)
        self.assertEqual(event['attributions']['memory-0']['label'],'harmful')
        self.assertTrue(m.validate_state(state,model_id=self.model,directory=self.directory))
        self.assertEqual(original_path.read_bytes(),original_bytes);self.assertEqual(call_path.read_bytes(),call_bytes)
        calls=len(self.gateway.calls)
        self.assertFalse(m.apply_feedback(self.feedback(action,loss=True),self.directory,self.bundle,self.gateway,self.model))
        self.assertEqual(len(self.gateway.calls),calls)
        for bad in ('unknown-used','unknown-support','invalid-direction'):
            value=copy.deepcopy(response)
            if bad=='unknown-used':value['used_memory_ids']=['invented']
            elif bad=='unknown-support':value['memory_support'][0]['memory_id']='invented'
            else:value['memory_support'][0]['support_direction']='BUY'
            with self.assertRaises(ValueError,msg=bad):m._direction_response(value,remaining,require_alignment=False)

    def test_later_real_knowledge_sees_ema_and_restores_without_future_source(self):
        self.initialize();action=self.decide();uid=action['used_memory_ids'][0]
        self.gateway.without[uid]='SHORT_LONG';job_start=self.clock
        self.clock+=timedelta(minutes=1)
        m.apply_feedback(self.feedback(action,loss=True),self.directory,self.bundle,self.gateway,self.model)
        self.clock+=timedelta(minutes=1);self.decide(12)
        state=read_json(self.directory/'live-state.json');queries=state['memory_bank']['retrievals']
        self.assertEqual(queries[0]['feedback_revision'],0);self.assertEqual(queries[1]['feedback_revision'],1)
        self.assertGreater(m.aware(queries[1]['as_of']),job_start)
        self.assertEqual(m.aware(queries[1]['source_cutoff']),m.aware(self.observation(12)['timestamp']))
        hit=next(row for row in queries[1]['hits'] if row['unit_id']==uid)
        self.assertAlmostEqual(hit['importance'],.48);self.assertEqual(hit['feedback_count'],1)
        self.assertTrue(all(m.aware(unit['unit']['exit_at'])<=m.aware(queries[1]['source_cutoff']) for unit in state['memory_bank']['units']))
        self.assertTrue(m.validate_state(state,model_id=self.model,directory=self.directory))

    def test_failed_loo_does_not_half_apply_credit_and_cannot_repeat_ambiguous_call(self):
        self.initialize();action=self.decide();self.gateway.fail_removed=action['used_memory_ids'][1]
        before=read_json(self.directory/'live-state.json')
        with self.assertRaises(RuntimeError):m.apply_feedback(self.feedback(action),self.directory,self.bundle,self.gateway,self.model)
        after=read_json(self.directory/'live-state.json')
        self.assertEqual(before,after);self.assertEqual(after['memory_bank']['feedback'],[])
        count=len(self.gateway.calls)
        with self.assertRaises(RuntimeError):m.apply_feedback(self.feedback(action),self.directory,self.bundle,self.gateway,self.model)
        self.assertEqual(len(self.gateway.calls),count)
        failed=[read_json(path) for path in (self.directory/'calls').glob('*.json') if read_json(path)['status']=='failed']
        self.assertEqual(len(failed),1)

    def test_equal_payoffs_need_no_extra_model_call_but_neutral_counts_each_cited_unit(self):
        self.initialize();action=self.decide();record=self.feedback(action)
        record['exit_prices']=copy.deepcopy(record['entry_prices']);record['step_return']=0.
        count=len(self.gateway.calls);m.apply_feedback(record,self.directory,self.bundle,self.gateway,self.model)
        self.assertEqual(len(self.gateway.calls),count)
        event=read_json(self.directory/'live-state.json')['memory_bank']['feedback'][0]
        self.assertEqual(set(event['attributions']),set(action['used_memory_ids']))
        self.assertTrue(all(row['label']=='neutral' for row in event['attributions'].values()))
        self.assertTrue(all(row['feedback_count_after']==1 for row in event['unit_updates'].values()))

    def test_embedding_failure_keeps_exact_unit_and_blocks_paid_retry_without_partial_ema(self):
        self.initialize();action=self.decide();before=read_json(self.directory/'live-state.json')
        calls=[]
        def fail_embedding(text):calls.append(text);raise RuntimeError('Synthetic embedding provider failure')
        self.gateway.embed=fail_embedding
        with self.assertRaises(RuntimeError):m.apply_feedback(self.feedback(action),self.directory,self.bundle,self.gateway,self.model)
        staged=read_json(next((self.directory/'feedback-staging').glob('*.json')))
        self.assertEqual(staged['status'],'failed');self.assertEqual(read_json(self.directory/'live-state.json'),before)
        self.clock+=timedelta(minutes=1)
        with self.assertRaises(RuntimeError):m.apply_feedback(self.feedback(action),self.directory,self.bundle,self.gateway,self.model)
        self.assertEqual(len(calls),1)
        self.assertEqual(staged,read_json(next((self.directory/'feedback-staging').glob('*.json'))))

    def test_uncited_and_invalid_citations_never_receive_implicit_reward(self):
        self.initialize();self.gateway.cite=False;action=self.decide()
        count=len(self.gateway.calls);m.apply_feedback(self.feedback(action),self.directory,self.bundle,self.gateway,self.model)
        self.assertEqual(len(self.gateway.calls),count)
        event=read_json(self.directory/'live-state.json')['memory_bank']['feedback'][0]
        self.assertEqual(event['attributions'],{});self.assertEqual(event['unit_updates'],{})
        self.assertEqual(event['used_memory_ids'],[])

    def test_fabricated_memory_id_rejected_no_flat_and_no_repeat(self):
        self.initialize();self.gateway.bad=True
        with self.assertRaises(RuntimeError):self.decide()
        count=len(self.gateway.calls)
        with self.assertRaises(RuntimeError):self.decide()
        self.assertEqual(len(self.gateway.calls),count)
        self.assertEqual(read_json(self.directory/'live-state.json')['decisions'],{})

    def test_move_shadow_keeps_relative_evidence_valid_and_tampering_is_rejected(self):
        self.initialize();action=self.decide();m.apply_feedback(self.feedback(action),self.directory,self.bundle,self.gateway,self.model)
        moved=self.directory.with_name('installed');self.directory.rename(moved)
        state=read_json(moved/'live-state.json');self.assertTrue(m.validate_state(state,model_id=self.model,directory=moved))
        ref=next(iter(state['decisions'].values()))['direction_call'];path=moved/ref['path']
        data=read_json(path);data['payload']['current_observation']['prices']['NVDA']=999.;write_private(path,data)
        with self.assertRaises(ValueError):m.validate_state(state,model_id=self.model,directory=moved)

    def test_outcome_record_tamper_is_rejected_even_when_bank_and_call_hashes_stay_valid(self):
        self.initialize();action=self.decide();m.apply_feedback(self.feedback(action),self.directory,self.bundle,self.gateway,self.model)
        original=read_json(self.directory/'live-state.json')
        for field in ('record','counterfactual','unit'):
            state=copy.deepcopy(original);key=next(iter(state['feedback']))
            if field=='record':state['feedback'][key]['exit_prices'][action['stock1']]+=1.
            elif field=='counterfactual':
                proof=state['attribution_evidence'][key];proof['outcome']['actual_return']=.99
            else:
                unit=next(row for row in state['memory_bank']['units'] if row['unit']['unit_id']=='action:'+key)
                unit['unit']['actual_return']=.99
            with self.assertRaises(ValueError,msg=field):m.validate_state(state,model_id=self.model,directory=self.directory)
        self.assertEqual(m._credit(1e-14,2e-14),'harmful')
        self.assertEqual(m._credit(2e-14,1e-14),'helpful')
        self.assertEqual(m._credit(1e-14,1e-14),'neutral')

    def test_training_namespace_never_reads_live_company_context_and_break_waits1530(self):
        self.initialize(scope='training');self.decide()
        with patch.object(self.n.engine,'statistical_lifecycle_state',return_value='BREAK'):
            second=self.decide(10);self.assertTrue(read_json(self.directory/'live-state.json')['pending_break'])
            final=self.decide(15)
        self.assertNotEqual({second['stock1'],second['stock2']},{final['stock1'],final['stock2']})
        self.assertTrue(all('company_operations_context' not in row['payload'] for row in self.gateway.calls))
        self.assertEqual(sum(row['task']=='pair_selection' for row in self.gateway.calls),2)

    def test_no_future_feedback_or_changed_accepted_inputs(self):
        self.initialize();action=self.decide();record=self.feedback(action)
        record['exit_at']='2026-11-05T11:30:00-05:00';record['settled_at']=record['exit_at']
        with self.assertRaises(ValueError):m.apply_feedback(record,self.directory,self.bundle,self.gateway,self.model)
        changed=self.observation();changed['news']['NVDA']=['Changed observation.']
        with self.assertRaises(ValueError):m.decide(changed,self.model,self.directory,self.bundle,self.gateway)

    def test_native_training_whole_episodes_fresh_and_reflection_finalized_only(self):
        from live_testing.initial_training import HistoricalDataset
        from live_testing.training import Observation,Execution,train_native
        times=[datetime(2026,10,19,h,30,tzinfo=m.legacy.NY) for h in (9,10,11,12,14,15)]
        prices=[{ticker:100. for ticker in m.TICKERS} for _ in times]
        prices[2]['NVDA']=120.;prices[3]['NVDA']=108.;prices[4]['NVDA']=115.;prices[5]['NVDA']=110.
        observations=[Observation(str(i),at,prices[i],{ticker:['Synthetic historical headline.'] for ticker in m.TICKERS},at)
            for i,at in enumerate(times[:4])]
        labels=[Execution(str(i),times[i+1],times[i+2],times[i+2],prices[i+1],prices[i+2],{'source':'synthetic exact interval'}) for i in range(4)]
        dataset=HistoricalDataset.build(observations,labels,training_at=datetime(2026,11,2,12,tzinfo=m.legacy.NY),
            release_nodes=times[:4],valuation_nodes=times[4:],window_start=times[0],window_end=times[-1])
        prepared=copy.deepcopy(self.prepared);prepared['prompt_memory']=self.n.DirectionPromptMemory(
            base_prompt=self.prepared['prompt_memory']['active_prompt'],symbols=m.TICKERS).to_dict()
        adapter=m.NativeTrainingAdapter(dataset=dataset,bundle=self.bundle,directory=self.root/'training',gateway=self.gateway,model_id=self.model)
        report=train_native(adapter,dataset,prepared,version=self.model,run_id='synthetic-v2',max_cycles=1)
        episodes=list((self.root/'training'/'episodes'/self.model).iterdir())
        self.assertEqual(len(episodes),2);self.assertEqual(report['completed_cycles'],1)
        self.assertEqual(report['matured_label_count'],4);self.assertEqual(report['checkpoint']['memory_policy_protocol'],m.PROTOCOL_VERSION)
        self.assertFalse(report['cycles'][0]['accepted'])
        for path in episodes:
            state=read_json(path/'live-state.json');self.assertEqual(len(state['feedback']),4)
            self.assertEqual(len(state['memory_bank']['units']),4)
            self.assertEqual(state['memory_bank']['retrievals'][0]['hits'],[])
            self.assertTrue(m.validate_state(state,model_id=self.model,directory=path))
        reflections=[row for row in self.gateway.calls if row['task']=='direction_reflection']
        self.assertEqual(len(reflections),1);self.assertEqual(len(reflections[0]['payload']['decisions']),4)
        self.assertTrue(all('company_operations_context' not in row['payload'] for row in self.gateway.calls))


if __name__=='__main__':unittest.main()
