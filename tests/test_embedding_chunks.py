from datetime import datetime,timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

import httpx

from live_testing import llm
from live_testing.budget import Budget
from live_testing.storage import digest,write_private


class ByteEncoder:
    def encode(self,text,**_kwargs):return list(text.encode('utf-8'))
    def decode_single_token_bytes(self,token):return bytes([token])


def vector(first=0.,second=0.):return [first,second]+[0.]*(3072-2)


class EmbeddingChunkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.budget=Budget(self.root,clock=lambda:datetime(2026,11,2,16,tzinfo=timezone.utc))
        self.gateway=llm.Gateway('synthetic-key',self.budget,self.root/'api-audit');self.requests=[]
        self.encoder=ByteEncoder();self.response=lambda texts:[vector(1.,0.)for _ in texts]
        def create(**kwargs):
            self.requests.append(kwargs);values=self.response(kwargs['input'])
            return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=v)for i,v in reversed(list(enumerate(values)))])
        self.client=SimpleNamespace(embeddings=SimpleNamespace(create=create));self.context=Mock()
        self.context.__enter__=Mock(return_value=self.client);self.context.__exit__=Mock(return_value=False)
        self.sdk=self.enterContext(patch('live_testing.llm.sdk_client',return_value=self.context))
        self.enterContext(patch('tiktoken.get_encoding',return_value=self.encoder))

    def cache(self,text):return self.root/'embedding-cache'/f'{digest([llm.EMBEDDING_MODEL,text])}.json'

    def test_long_input_uses_every_character_token_weights_and_l2_normalizes(self):
        text='a'*8000+'b'*2000
        self.response=lambda texts:[vector(1.,0.)if value.startswith('a')else vector(0.,1.)for value in texts]
        result=self.gateway.embed(text);expected=math.hypot(.8,.2)
        self.assertAlmostEqual(result[0],.8/expected);self.assertAlmostEqual(result[1],.2/expected)
        self.assertAlmostEqual(math.hypot(*result),1.);self.assertEqual(len(result),3072)
        self.assertEqual(''.join(self.requests[0]['input']),text);self.assertEqual(len(self.requests),1)
        record=json.loads(self.cache(text).read_text());self.assertEqual(record['chunk_token_lengths'],[8000,2000])
        self.assertEqual(record['chunk_input_sha256'],[digest('a'*8000),digest('b'*2000)])
        self.assertEqual(record['dimension'],3072);self.assertEqual(record['embedding_protocol'],llm.EMBEDDING_PROTOCOL)
        self.assertEqual(record['embedding_sha256'],digest(result));self.assertEqual(self.cache(text).stat().st_mode&0o777,0o600)
        before=self.cache(text).read_bytes();self.assertEqual(self.gateway.embed(text),result)
        self.assertEqual(len(self.requests),1);self.assertEqual(self.cache(text).read_bytes(),before)

    def test_utf8_boundaries_preserve_unicode_and_no_chunk_exceeds_eight_thousand(self):
        text='a'*7999+'💹中文'*4+'<|endoftext|>'
        self.gateway.embed(text);chunks=[s for r in self.requests for s in r['input']]
        self.assertEqual(''.join(chunks),text);self.assertTrue(all(1<=len(self.encoder.encode(s))<=8000 for s in chunks))
        self.assertTrue(all('\ufffd'not in s for s in chunks))

    def test_large_input_batches_are_bounded_and_response_indices_restore_order(self):
        text=''.join(f'{i:04d}'+'a'*7996 for i in range(34));self.gateway.embed(text)
        self.assertEqual([len(r['input'])for r in self.requests],[31,3])
        self.assertEqual(''.join(s for r in self.requests for s in r['input']),text)
        for request in self.requests:
            self.assertEqual(request['model'],'text-embedding-3-large');self.assertEqual(request['encoding_format'],'float')
            self.assertLessEqual(len(request['input']),128)
            self.assertLessEqual(sum(len(self.encoder.encode(s))for s in request['input']),250000)

    def test_short_input_keeps_provider_vector_and_existing_legacy_cache_is_immutable(self):
        self.response=lambda texts:[vector(3.,4.)for _ in texts]
        self.assertEqual(self.gateway.embed('one source'),vector(3.,4.))
        self.assertEqual(self.requests[0]['input'],['one source'])
        text='legacy input';path=self.cache(text);write_private(path,{'model':llm.EMBEDDING_MODEL,
            'input_sha256':digest(text),'embedding':vector(2.,1.),'generated_at':'actual historical generation'})
        before=path.read_bytes();self.assertEqual(self.gateway.embed(text),vector(2.,1.))
        self.assertEqual(path.read_bytes(),before);self.assertEqual(len(self.requests),1)

    def test_failed_later_batch_retains_real_completed_chunks_for_free_resume(self):
        text='a'*8000+'b'*8000+'c'*1000;original=self.client.embeddings.create
        def create(**request):
            if request['input'][0].startswith('c'):raise RuntimeError('Synthetic transport failure')
            return original(**request)
        self.client.embeddings.create=create
        with patch.object(llm,'EMBEDDING_BATCH_TOKENS',16000):
            with self.assertRaises(RuntimeError):self.gateway.embed(text)
            self.assertFalse(self.cache(text).exists());self.assertEqual(len(list((self.root/'embedding-cache/chunks').glob('*.json'))),2)
            self.client.embeddings.create=original;self.gateway.embed(text)
        self.assertEqual([len(r['input'])for r in self.requests],[2,1]);self.assertEqual(self.requests[-1]['input'],['c'*1000])

    def test_missing_bad_or_zero_aggregate_embeddings_fail_without_placeholder(self):
        self.client.embeddings.create=lambda **_kwargs:SimpleNamespace(data=[])
        with self.assertRaisesRegex(ValueError,'omitted'):self.gateway.embed('missing actual vector')
        self.assertFalse(self.cache('missing actual vector').exists())
        self.client.embeddings.create=lambda **_kwargs:SimpleNamespace(data=[SimpleNamespace(index=0,embedding=[1.,2.])])
        with self.assertRaisesRegex(ValueError,'3072'):self.gateway.embed('wrong provider dimension')
        self.response=lambda texts:[vector(1.,0.)if text.startswith('a')else vector(-1.,0.)for text in texts]
        def create(**request):return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=v)for i,v in enumerate(self.response(request['input']))])
        self.client.embeddings.create=create
        with self.assertRaisesRegex(ValueError,'aggregate direction'):self.gateway.embed('a'*8000+'b'*8000)
        self.assertFalse(self.cache('a'*8000+'b'*8000).exists())

    def test_changed_cache_fails_closed_without_any_extra_provider_request(self):
        text='audited existing source';self.gateway.embed(text);path=self.cache(text);record=json.loads(path.read_text())
        record['embedding'][0]=2.;write_private(path,record);before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'cache protocol'):self.gateway.embed(text)
        self.assertEqual(path.read_bytes(),before);self.assertEqual(len(self.requests),1)

    def test_chunk_batches_use_provider_usage_through_the_real_budget_transport(self):
        budget=self.budget.scoped('moira2-top1','replay','synthetic-first-release')
        def handler(request):
            payload=json.loads(request.content)
            return httpx.Response(200,json={'model':llm.EMBEDDING_MODEL,'usage':{'prompt_tokens':19,'total_tokens':19},
                'data':[{'index':i,'embedding':vector(1.,0.)}for i in range(len(payload['input']))]})
        with patch('live_testing.llm.httpx.HTTPTransport',return_value=httpx.MockTransport(handler)):
            transport=llm.BudgetTransport(budget,'memory-embedding')
        self.addCleanup(transport.close)
        def create(**payload):
            response=transport.handle_request(httpx.Request('POST','https://api.openai.com/v1/embeddings',json=payload))
            return SimpleNamespace(data=[SimpleNamespace(**row)for row in response.json()['data']])
        self.client.embeddings.create=create;self.gateway.budget=budget
        self.gateway.embed('a'*8000+'b'*2000)
        summary=budget.summary();self.assertEqual(summary['prompt_tokens'],19);self.assertEqual(summary['completion_tokens'],0)
        calls=json.loads((self.root/'api-budget.json').read_text())['calls'];self.assertEqual(len(calls),1)
        call=next(iter(calls.values()));self.assertEqual(call['model_id'],'moira2-top1');self.assertEqual(call['stage'],'replay')
        self.assertEqual(call['phase'],'memory-embedding');self.assertEqual(call['status'],'completed')
        self.assertAlmostEqual(call['actual_cost_usd'],19*.13/1_000_000)


if __name__=='__main__':unittest.main()
