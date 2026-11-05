"""Pure prompt views: preserve evidence text, discard private bulk metadata."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from live_testing import prompt_projection as p
from live_testing import native_source
from live_testing.registry import TICKERS
from live_testing.storage import write_private


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.document={'schema':'private-context','generated_at':'2026-11-02T08:00:00-05:00',
            'archive':{'as_of':'2026-11-01T16:00:00-05:00','input_sha256':'private-audit-hash'},
            'training':{'checkpoint':'private-path'},'response_id':'private-response-id','companies':{}}
        for i,ticker in enumerate(TICKERS):
            prefix='source-'+ticker+'-'
            self.document['companies'][ticker]={'recent_operations_summary':'Complete synthetic company operations summary '+ticker+'.',
                'summary_evidence_uuids':[prefix+'summary'],
                'company_facts':[{'statement':'A complete reported company fact.','evidence_uuids':[prefix+'fact']}],
                'analyst_outlook':[{'statement':'A complete analyst forecast, not an observed result.','evidence_uuids':[prefix+'outlook']}],
                'limitations':'Only supplied archived evidence is available.',
                'source_date_bounds':{'first':'2026-10-29T12:00:00-04:00','last':'2026-11-01T15:30:00-05:00'},
                'source_ids':[prefix+'summary',prefix+'fact',prefix+'outlook']+[prefix+str(j) for j in range(200)],
                'coverage':{'status':'partial'},'generated_at':'2026-11-02T08:00:00-05:00',
                'output_tokens':99,'response_id':'private-response-id'}

    def test_full_semantics_preserved_without_source_ids_or_audit_metadata(self):
        before=copy.deepcopy(self.document);view=p.compact_company_context(self.document)
        self.assertEqual(self.document,before);self.assertEqual(list(view['companies']),list(TICKERS))
        self.assertEqual(set(view),{'as_of','source_basis','companies'})
        mapping=p.company_reference_map(self.document)
        self.assertEqual(len(mapping),30)
        self.assertEqual(list(mapping),sorted(mapping))
        for ticker,row in view['companies'].items():
            original=self.document['companies'][ticker]
            self.assertEqual(row['recent_operations_summary'],original['recent_operations_summary'])
            self.assertEqual(row['limitations'],original['limitations'])
            for field in ('company_facts','analyst_outlook'):
                self.assertEqual(row[field][0]['statement'],original[field][0]['statement'])
                self.assertEqual([mapping[ref] for ref in row[field][0]['evidence_refs']],original[field][0]['evidence_uuids'])
            self.assertEqual([mapping[ref] for ref in row['summary_evidence_refs']],original['summary_evidence_uuids'])
        text=json.dumps(view)
        for secret in ('source_ids','response_id','input_sha256','private-path','output_tokens','generated_at','coverage'):
            self.assertNotIn(secret,text)
        self.assertLess(len(text),len(json.dumps(self.document))*.25)

    def test_pair_scope_preserves_order_and_does_not_leak_unselected_company(self):
        pair=(TICKERS[2],TICKERS[0]);view=p.compact_company_context(self.document,tickers=pair)
        self.assertEqual(list(view['companies']),list(pair));mapping=p.company_reference_map(self.document,tickers=pair)
        self.assertEqual(len(mapping),6)
        for ticker in set(TICKERS)-set(pair):
            self.assertNotIn('source-'+ticker,json.dumps(mapping));self.assertNotIn(ticker,view['companies'])
        full=p.company_reference_map(self.document)
        self.assertTrue(all(full[key]==value for key,value in mapping.items()))

    def test_missing_repeated_unknown_or_unordered_tickers_fail_closed(self):
        invalid=[[],[TICKERS[0]],[TICKERS[0],TICKERS[0]],['UNKNOWN',TICKERS[0]],set(TICKERS),TICKERS[0]]
        for tickers in invalid:
            with self.assertRaises(ValueError,msg=str(tickers)):p.compact_company_context(self.document,tickers=tickers)
        missing=copy.deepcopy(self.document);del missing['companies'][TICKERS[0]]
        with self.assertRaises(ValueError):p.compact_company_context(missing)
        with self.assertRaises(ValueError):p.compact_company_context(missing,tickers=TICKERS[:2])

    def test_fabricated_references_and_future_publication_bounds_rejected(self):
        for field in ('summary_evidence_uuids','source_date_bounds','company_facts'):
            doc=copy.deepcopy(self.document);row=doc['companies'][TICKERS[0]]
            if field=='summary_evidence_uuids':row[field]=['unknown-source']
            elif field=='source_date_bounds':row[field]['last']='2026-11-02T09:30:00-05:00'
            else:row[field][0]['evidence_uuids']=['unknown-source']
            with self.assertRaises(ValueError,msg=field):p.compact_company_context(doc)

    def test_legacy_summary_only_context_is_accepted(self):
        doc={'as_of':'2026-11-01T12:00:00-05:00','companies':{t:{'recent_operations_summary':'Synthetic prior source.'} for t in TICKERS}}
        self.assertEqual(p.company_reference_map(doc),{})
        self.assertEqual(p.compact_company_context(doc)['companies'][TICKERS[0]],{'recent_operations_summary':'Synthetic prior source.'})

    def test_operations_for_keeps_causal_checks_and_original_file_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'company-context.json';write_private(path,self.document)
            original=path.read_bytes();budget=SimpleNamespace(root=root)
            observation={'timestamp':'2026-11-02T09:30:00-05:00'}
            view=native_source.operations_for(budget,observation,tickers=TICKERS[:2])
            self.assertEqual(list(view['companies']),list(TICKERS[:2]));self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(RuntimeError):native_source.operations_for(budget,{'timestamp':'2026-10-30T09:30:00-04:00'})
            future=copy.deepcopy(self.document);future['generated_at']='2099-01-01T00:00:00+00:00';write_private(path,future)
            with self.assertRaises(RuntimeError):native_source.operations_for(budget,observation)

    def test_reflection_view_keeps_exact_bank_id_full_text_and_retrieval_order(self):
        text='A complete reflection. '+('No sentence is truncated. '*200)
        first={'unit_id':'action:actual-full-bank-id-1','reflection':text,'provenance':{'private':'audit'},
            'retrieval':{'score':2.3,'clock':{'private':'clock'}},'actual_return':.01,'news':{'private':'news'}}
        second={**copy.deepcopy(first),'unit_id':'action:actual-full-bank-id-2'}
        originals=copy.deepcopy([first,second]);views=p.compact_reflection_hits([first,second])
        self.assertEqual(views,[{'unit_id':row['unit_id'],'reflection':text} for row in originals])
        self.assertEqual([first,second],originals)
        complete='Complete news-free historical display. '+('All finalized outcome reasoning is retained. '*50)
        self.assertEqual(p.compact_reflection_hit(first,reflection_text=complete),{'unit_id':first['unit_id'],'reflection':complete})
        self.assertEqual(p.compact_reflection_hits([first,second],reflection_renderer=lambda row:complete)[1]['reflection'],complete)
        with self.assertRaises(ValueError):p.compact_reflection_hits([first,first])
        with self.assertRaises(ValueError):p.compact_reflection_hit(first,reflection_text='')

    def observation(self):
        return {'round_id':'private-audit-round','timestamp':'2026-11-04T10:30:00-05:00',
            'prices':{t:100.+i for i,t in enumerate(TICKERS)},
            'news':{t:['Published 2026-11-04T09:45:12.000123-05:00 [full-article-id-'+t+']: Original summary: every word remains.'] for t in TICKERS},
            'news_protocol':'private-audit-protocol','news_coverage':{t:{'status':'partial','live_capped':True,
                'window_truncated':True,'content_basis':'partial_headline','invalid_publication_count':1,
                'conflict_count':2,'private_audit':'not for prompt'} for t in TICKERS}}

    def test_observation_scopes_pair_retains_exact_timestamp_prices_summaries_and_necessary_limits(self):
        observation=self.observation();before=copy.deepcopy(observation);pair=TICKERS[:2]
        view=p.compact_observation(observation,pair);mapping=p.observation_reference_map(observation,pair)
        self.assertEqual(observation,before);self.assertEqual(view['timestamp'],observation['timestamp'])
        self.assertEqual(view['prices'],{t:observation['prices'][t] for t in pair})
        self.assertEqual(set(view['news']),set(pair));self.assertEqual(len(mapping),2)
        for ticker in pair:
            short=p.short_reference('full-article-id-'+ticker)
            self.assertEqual(view['news'][ticker],['Published 2026-11-04T09:45:12.000123-05:00 ['+short+']: Original summary: every word remains.'])
            self.assertEqual(mapping[short],'full-article-id-'+ticker)
            self.assertIn('Coverage is partial.',view['news_limitations'][ticker])
            self.assertIn('Source search capped.',view['news_limitations'][ticker])
            self.assertIn('2 conflicting publication groups omitted.',view['news_limitations'][ticker])
        for audit in ('round_id','news_coverage','news_protocol','private_audit','full-article-id-'):
            self.assertNotIn(audit,json.dumps(view))

    def test_observation_future_publication_and_missing_ticker_fail_closed(self):
        for published in ('2026-11-04T10:30:00-05:00','2026-11-04T10:31:00-05:00'):
            observation=self.observation();observation['news'][TICKERS[0]]=['Published '+published+' [actual-id]: Actual summary.']
            with self.assertRaises(ValueError):p.compact_observation(observation)
        missing=self.observation();del missing['prices'][TICKERS[0]]
        with self.assertRaises(ValueError):p.compact_observation(missing)

    def test_news_compaction_only_matches_header_and_never_rewrites_reflections_or_body(self):
        exact='Published 2026-11-04T09:45:12-05:00 [original-id]: Quote [leave this untouched]: and Published 2026-01-01T10:00:00-05:00 [nested-id]: content.'
        compact=p.compact_news_text(exact)
        self.assertEqual(compact.split(']: ',1)[1],exact.split(']: ',1)[1])
        self.assertIn('[nested-id]',compact)
        self.assertEqual(p.compact_news_text('A quotation: '+exact),'A quotation: '+exact)
        payload={'active_candidate':{'news_a':[exact],'news_b':[exact]},'current_observation':{'news':{TICKERS[0]:[exact]}},
            'ranked_counterfactual_reflections':[{'unit_id':'full-memory-id','reflection':exact}], 'reason':exact}
        before=copy.deepcopy(payload);view=p.compact_payload_news(payload)
        self.assertEqual(payload,before);self.assertEqual(view['active_candidate']['news_a'],[compact])
        self.assertEqual(view['current_observation']['news'][TICKERS[0]],[compact])
        self.assertEqual(view['ranked_counterfactual_reflections'],payload['ranked_counterfactual_reflections'])
        self.assertEqual(view['reason'],exact)

    def test_unknown_coverage_and_unformatted_legacy_news_are_not_claimed_complete(self):
        observation=self.observation();observation.pop('news_coverage');observation['news'][TICKERS[0]]=['Whole legacy summary unchanged.']
        view=p.compact_observation(observation,TICKERS[:2])
        self.assertEqual(view['news'][TICKERS[0]],['Whole legacy summary unchanged.'])
        self.assertIn('Coverage information unavailable.',view['news_limitations'][TICKERS[0]])
        self.assertIn('Some news lacks an exact publication timestamp.',view['news_limitations'][TICKERS[0]])

    def test_reference_collision_between_different_tickers_is_rejected(self):
        with patch.object(p,'short_reference',return_value='same-short-ref'):
            with self.assertRaises(ValueError):p.observation_reference_map(self.observation())
            doc=copy.deepcopy(self.document)
            for company in doc['companies'].values():
                company.pop('company_facts');company.pop('analyst_outlook')
            with self.assertRaises(ValueError):p.company_reference_map(doc)

    def test_native_news_headers_preserve_memory_ids_dates_body_and_other_lines(self):
        header='Published 2026-11-04T09:45:12-05:00 [source-original-id]: Original summary [body bracket] stays exact.'
        lines=['Short memory:\r\n','7. ['+TICKERS[0]+'] '+header+'\r\n',
            '18. ['+TICKERS[1]+'] Published2026-11-04T10:30:00-05:00: '+header+'\n',
            '19. Reflection action:bank-memory-id '+header+'\n',
            'Unquoted prose: '+header+'\n','Necessary memory ID: action:actual-bank-id\n']
        original=''.join(lines);view=p.compact_native_prompt(original);short=p.short_reference('source-original-id')
        expected=''.join([lines[0],lines[1].replace('[source-original-id]','['+short+']'),
            lines[2].replace('[source-original-id]','['+short+']'),*lines[3:]])
        self.assertEqual(view,expected);self.assertEqual(p.native_prompt_reference_map(original),{short:'source-original-id'})
        self.assertIn('action:actual-bank-id',view)

    def test_json_encoded_news_fields_only_change_article_refs_not_decoded_text(self):
        header='Published 2026-11-04T09:45:12-05:00 [source-id]: Quoted "word" and all Unicode 中文 stay.'
        wrapped=json.dumps([header],ensure_ascii=True)
        payload={'active_candidate':{'news_a':wrapped,'news_b':{'items':[wrapped]},'news_A_summary':{'major_event':wrapped}},
            'reflection':wrapped,'reason':wrapped,'unit_id':'actual-full-bank-id'}
        view=p.compact_payload_news(payload)
        expected=p.compact_news_text(header)
        self.assertEqual(json.loads(view['active_candidate']['news_a']),[expected])
        self.assertEqual(json.loads(view['active_candidate']['news_b']['items'][0]),[expected])
        self.assertEqual(json.loads(view['active_candidate']['news_A_summary']['major_event']),[expected])
        self.assertEqual(view['reflection'],wrapped);self.assertEqual(view['reason'],wrapped)
        self.assertEqual(view['unit_id'],payload['unit_id'])


if __name__=='__main__':unittest.main()
