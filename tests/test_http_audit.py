import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from live_testing.budget import Budget,BudgetExceeded
from live_testing.llm import BudgetTransport
from live_testing.storage import digest


class HttpAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.budget=Budget(self.root,model_id='finmem',stage='inference',event_id='synthetic-release',
            clock=lambda:datetime(2026,11,2,16,tzinfo=timezone.utc))

    def transport(self,handler):
        with patch('live_testing.llm.httpx.HTTPTransport',return_value=httpx.MockTransport(handler)):
            transport=BudgetTransport(self.budget,'finmem-live')
        self.addCleanup(transport.close);return transport

    def request(self,payload,endpoint='chat/completions'):
        return httpx.Request('POST','https://api.openai.com/v1/'+endpoint+'?unused=secret-query-value',
            headers={'Authorization':'Bearer synthetic-key-never-log','X-Private':'synthetic-header-never-log'},json=payload)

    def records(self):return [json.loads(p.read_text())for p in (self.root/'api-audit/http-requests').glob('*.json')]

    def test_exact_post_override_payload_is_private_and_saved_before_http_without_auth_or_query(self):
        def handler(request):
            records=self.records();self.assertEqual(len(records),1);record=records[0]
            payload=json.loads(request.content)
            self.assertEqual(record['status'],'started');self.assertEqual(record['request_payload'],payload)
            self.assertIs(payload['store'],False);self.assertEqual(payload['service_tier'],'default')
            self.assertEqual(record['wire_body_sha256'],hashlib.sha256(request.content).hexdigest())
            self.assertEqual(record['request_sha256'],digest(payload))
            self.assertEqual(request.headers['Authorization'],'Bearer synthetic-key-never-log')
            return httpx.Response(200,json={'id':'provider-success','model':'gpt-4.1-mini',
                'usage':{'prompt_tokens':20,'completion_tokens':3,'prompt_tokens_details':{'cached_tokens':5}},
                'choices':[]})
        transport=self.transport(handler)
        transport.handle_request(self.request({'model':'gpt-4.1-mini','messages':[{'role':'user','content':'Dated evidence'}],
            'store':True,'service_tier':'priority','max_tokens':100}))
        record=self.records()[0];ledger=json.loads((self.root/'api-budget.json').read_text());identity=record['reservation_id']
        self.assertEqual(set(ledger['calls']),{identity});self.assertEqual(record['request_sha256'],ledger['calls'][identity]['request_sha256'])
        self.assertEqual((record['model_id'],record['stage'],record['event_id'],record['phase']),
            ('finmem','inference','synthetic-release','finmem-live'))
        self.assertEqual(record['request_path'],'/v1/chat/completions');self.assertEqual(record['status'],'completed')
        self.assertEqual(record['response_id'],'provider-success');self.assertEqual(record['usage']['completion_tokens'],3)
        serialized=json.dumps(record)
        for secret in ('synthetic-key-never-log','synthetic-header-never-log','secret-query-value','Authorization'):
            self.assertNotIn(secret,serialized)
        path=self.root/'api-audit/http-requests'/f'{identity}.json'
        self.assertEqual(path.stat().st_mode&0o777,0o600);self.assertEqual(path.parent.stat().st_mode&0o777,0o700)

    def test_embedding_body_is_also_audited_without_chat_only_overrides(self):
        payload={'model':'text-embedding-3-large','input':['Historical memory'],'encoding_format':'float'}
        def handler(request):
            self.assertEqual(self.records()[0]['request_payload'],payload)
            self.assertEqual(json.loads(request.content),payload)
            return httpx.Response(200,json={'model':'text-embedding-3-large','data':[],
                'usage':{'prompt_tokens':4,'total_tokens':4}})
        self.transport(handler).handle_request(self.request(payload,'embeddings'))
        record=self.records()[0];self.assertEqual(record['request_path'],'/v1/embeddings')
        self.assertEqual(record['status'],'completed');self.assertNotIn('store',record['request_payload'])
        self.assertEqual(record['usage']['total_tokens'],4)

    def test_http_failure_keeps_reservation_but_never_saves_provider_error_text(self):
        def handler(request):
            return httpx.Response(429,json={'error':{'message':'Provider echoes synthetic-key-never-log'}})
        response=self.transport(handler).handle_request(self.request({'model':'gpt-4.1-mini','input':'evidence','max_output_tokens':100},'responses'))
        self.assertEqual(response.status_code,429);record=self.records()[0]
        self.assertEqual(record['status'],'failed');self.assertEqual(record['error_type'],'HTTPError');self.assertEqual(record['status_code'],429)
        self.assertNotIn('Provider echoes',json.dumps(record));self.assertNotIn('synthetic-key-never-log',json.dumps(record))
        call=json.loads((self.root/'api-budget.json').read_text())['calls'][record['reservation_id']]
        self.assertEqual(call['guard_cost_usd'],call['reserved_upper_usd']);self.assertNotIn('actual_cost_usd',call)

    def test_transport_exception_saves_sanitized_type_and_retains_unknown_cost(self):
        def handler(request):raise httpx.ReadTimeout('Do not log synthetic-key-never-log',request=request)
        with self.assertRaises(httpx.ReadTimeout):
            self.transport(handler).handle_request(self.request({'model':'gpt-4.1-mini','input':'evidence','max_tokens':100}))
        record=self.records()[0];self.assertEqual(record['error_type'],'ReadTimeout');self.assertIsNone(record['status_code'])
        self.assertNotIn('synthetic-key-never-log',json.dumps(record));self.assertGreater(self.budget.summary()['guard_total_usd'],0)

    def test_audit_failure_prevents_http_and_does_not_refund_the_reservation(self):
        calls=[];transport=self.transport(lambda request:calls.append(request))
        with patch('live_testing.llm.write_private',side_effect=OSError('Synthetic local audit failure')):
            with self.assertRaises(OSError):transport.handle_request(self.request({'model':'gpt-4.1-mini','input':'evidence','max_tokens':100}))
        self.assertEqual(calls,[]);self.assertGreater(self.budget.summary()['guard_total_usd'],0)
        row=next(iter(json.loads((self.root/'api-budget.json').read_text())['calls'].values()))
        self.assertEqual(row['status'],'failed');self.assertEqual(row['error_type'],'OSError')

    def test_budget_rejection_has_no_transport_or_unreserved_audit(self):
        calls=[];transport=self.transport(lambda request:calls.append(request))
        with patch.object(self.budget,'reserve',side_effect=BudgetExceeded('Synthetic monthly cap')):
            with self.assertRaises(BudgetExceeded):transport.handle_request(self.request({'model':'gpt-4.1-mini','input':'evidence','max_tokens':100}))
        self.assertEqual(calls,[]);self.assertEqual(self.records(),[])

    def test_success_metadata_survives_a_post_response_accounting_guard_failure(self):
        transport=self.transport(lambda request:httpx.Response(200,json={'id':'actual-success',
            'model':'gpt-4.1-mini','usage':{'input_tokens':20,'output_tokens':3}}))
        original=self.budget.finish
        def finish(identity,**kwargs):
            if 'response' in kwargs:raise BudgetExceeded('Synthetic unexpected provider-cost guard')
            return original(identity,**kwargs)
        with patch.object(self.budget,'finish',side_effect=finish):
            with self.assertRaises(BudgetExceeded):transport.handle_request(self.request({'model':'gpt-4.1-mini','input':'evidence','max_tokens':100}))
        record=self.records()[0]
        self.assertEqual(record['status_code'],200);self.assertEqual(record['response_id'],'actual-success')
        self.assertEqual(record['usage']['output_tokens'],3);self.assertEqual(record['error_type'],'BudgetExceeded')
        self.assertGreater(self.budget.summary()['guard_total_usd'],0)


if __name__=='__main__':unittest.main()
