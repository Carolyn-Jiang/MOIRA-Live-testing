import copy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from live_testing import news
from live_testing.registry import TICKERS
from live_testing.storage import write_private


class NewsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.start='2026-11-02T09:30:00-05:00';self.end='2026-11-02T10:30:00-05:00'
        self.article={'uuid':'shared','link':'https://finance.yahoo.com/news/shared-123456.html',
            'title':'Company launches an updated product','providerPublishTime':datetime.fromisoformat('2026-11-02T10:00:00-05:00').timestamp()}
        self.summary='The company announced a product update, while the headline alone provides limited operating detail.'
        # Counting stub tests bounds/cache behavior without downloading BPE data.
        # Production defaults explicitly to tiktoken's real cl100k_base encoder.
        self.encoder=Mock();self.encoder.encode.side_effect=lambda text:text.split()

    def fetch(self,ticker):return [copy.deepcopy(self.article)] if ticker in ('NVDA','GOOG') else []

    def test_global_dedup_summarizes_once_and_retains_both_ticker_associations(self):
        collection=news.collect_articles(self.start,self.end,self.root,fetcher=self.fetch)
        self.assertEqual(len(collection['articles']),1);self.assertEqual(collection['articles'][0]['tickers'],['GOOG','NVDA'])
        summarize=Mock(return_value=self.summary)
        result=news.summarize_articles(collection,summarize,self.root,encoder=self.encoder)
        self.assertEqual(summarize.call_count,1);self.assertEqual(len(result['news_rows']),2)
        self.assertEqual({row['ticker'] for row in result['news_rows']},{'GOOG','NVDA'})
        self.assertTrue(all(row['content_basis']=='partial_headline' for row in result['news_rows']))
        self.assertTrue(all(10<=row['summary_tokens']<=50 for row in result['news_rows']))
        news.summarize_articles(collection,summarize,self.root,encoder=self.encoder);self.assertEqual(summarize.call_count,1)
        cached=next((self.root/'summaries').glob('*.json'));self.assertEqual(cached.stat().st_mode&0o777,0o600)
        self.assertIn('untrusted',summarize.call_args.args[0]['summary_instructions'])

    def test_cache_union_uses_source_publication_and_half_open_window(self):
        rows=[{**self.article,'uuid':'start','published_at_utc':self.start,'publication_precision':'exact'},
              {**self.article,'uuid':'end','link':'https://example.com/end','published_at_utc':self.end,'publication_precision':'exact'},
              {'uuid':'modified-only','title':'Not admissible','url':'https://example.com/modified','dateModified':self.start}]
        write_private(self.root/'archives'/'sunday.json',{'symbols':{'AAPL':{'articles':rows}}})
        result=news.collect_articles(self.start,self.end,self.root,fetcher=lambda ticker:[])
        self.assertEqual([row['article_id'] for row in result['articles']],['start'])
        self.assertEqual(result['coverage']['AAPL']['cache_used_count'],1)
        self.assertEqual(result['coverage']['AAPL']['invalid_publication_count'],1)

    def test_exact_archive_time_corrects_updated_provider_time_before_filter(self):
        archived={**self.article,'published_at_utc':'2026-10-30T12:00:00-04:00','publication_precision':'exact'}
        write_private(self.root/'archives'/'sunday.json',{'symbols':{'NVDA':{'articles':[archived]}}})
        result=news.collect_articles(self.start,self.end,self.root,fetcher=self.fetch)
        self.assertEqual(result['articles'],[])

    def test_actual_article_metadata_overrides_provider_time_and_rechecks_window(self):
        reader=Mock(return_value={'datePublished':'2026-10-30T12:00:00-04:00','body':'Actual article body.'})
        result=news.collect_articles(self.start,self.end,self.root,fetcher=self.fetch,article_fetcher=reader)
        self.assertEqual(reader.call_count,1);self.assertEqual(result['articles'],[])

    def test_source_body_is_used_but_future_modified_body_is_never_supplied(self):
        row={**self.article,'body':'Actual article text from a verified source.'}
        result=news.collect_articles(self.start,self.end,self.root,fetcher=lambda ticker:[row] if ticker=='NVDA' else [])
        summarize=Mock(return_value=self.summary);news.summarize_articles(result,summarize,self.root,encoder=self.encoder)
        self.assertEqual(summarize.call_args.args[0]['body'],row['body']);self.assertEqual(result['articles'][0]['content_basis'],'article_body')
        row['dateModified']='2026-11-03T10:00:00-05:00'
        other=Path(self.temp.name)/'separate'
        result=news.collect_articles(self.start,self.end,other,fetcher=lambda ticker:[row] if ticker=='NVDA' else [])
        self.assertIsNone(result['articles'][0]['body']);self.assertEqual(result['articles'][0]['content_basis'],'partial_headline')

    def test_capped_and_missing_feeds_remain_explicitly_partial_or_unavailable(self):
        articles=[{**self.article,'uuid':str(i),'link':f'https://example.com/{i}'} for i in range(200)]
        def fetch(ticker):
            if ticker=='NVDA':return {'articles':articles,'capped':True}
            raise RuntimeError('fetch unavailable')
        result=news.collect_articles(self.start,self.end,self.root,fetcher=fetch)
        self.assertTrue(result['coverage']['NVDA']['live_capped']);self.assertTrue(result['coverage']['NVDA']['window_truncated'])
        self.assertEqual(result['coverage']['NVDA']['status'],'partial');self.assertEqual(result['coverage']['META']['status'],'unavailable')
        self.assertEqual(result['coverage']['META']['live_error_type'],'RuntimeError')

    def test_invalid_summary_never_becomes_padded_or_cached_success(self):
        collection=news.collect_articles(self.start,self.end,self.root,fetcher=self.fetch)
        for text in ('Too short.',self.summary*20):
            with self.assertRaisesRegex(ValueError,'10–50'):news.summarize_articles(collection,lambda article,stage:text,self.root,encoder=self.encoder)
        self.assertFalse((self.root/'summaries').exists())

    def test_html_extracts_datePublished_and_body_never_modified_only(self):
        html='<script type="application/ld+json">'+json.dumps({'@type':'NewsArticle','datePublished':self.start,'dateModified':self.end,'articleBody':'Source article text.'})+'</script>'
        result=news.parse_article_html(html);self.assertEqual(result['datePublished'],self.start);self.assertEqual(result['body'],'Source article text.')
        with self.assertRaises(ValueError):news.parse_article_html('<script type="application/ld+json">{"@type":"NewsArticle","dateModified":"'+self.end+'"}</script><time datetime="'+self.end+'">Updated</time>')
        result=news.parse_article_html('<meta property="article:published_time" content="'+self.start+'"><div class="caas-body"><p>First paragraph.</p><p>Second paragraph.</p></div>')
        self.assertIn('First paragraph.',result['body']);self.assertIn('Second paragraph.',result['body'])

    def test_normalized_provider_basis_never_upgrades_to_trusted_exact(self):
        end=news.aware(self.end)
        provider=news._normalize(self.article,'NVDA',{'source':'yahoo_search_max_200'},end)
        again=news._normalize(provider,'NVDA',{'source':'local_archive'},end)
        self.assertEqual(again['publication_basis'],'providerPublishTime')
        legacy={**self.article,'published_at':self.start,'source_provenance':[{'source':'live_yahoo_search_provider_timestamp'}]}
        legacy.pop('providerPublishTime')
        self.assertEqual(news._normalize(legacy,'NVDA',{'source':'local_archive'},end)['publication_basis'],'providerPublishTime')
        legacy['source_provenance']=[{'source':'yahoo_browser_archive_exact_article_publication'}]
        self.assertEqual(news._normalize(legacy,'NVDA',{'source':'local_archive'},end)['publication_basis'],'exact_source_publication')

    def test_legacy_exact_and_provider_url_bridge_uses_actual_archive_publication(self):
        old_url='https://example.com/original';new_url='https://example.com/updated';end=news.aware(self.end)
        exact={'uuid':'bridge-id','link':old_url,'title':'Archived exact publication','published_at':self.start,
            'source_provenance':[{'source':'yahoo_browser_archive_exact_article_publication'}]}
        old_provider={'link':new_url,'title':'Provider alias','published_at':'2026-11-02T10:00:00-05:00',
            'source_provenance':[{'source':'live_yahoo_search_provider_timestamp'}]}
        bridge={**self.article,'uuid':'bridge-id','link':new_url}
        normalized=[news._normalize(row,'NVDA',{'source':'local_archive'},end) for row in (exact,old_provider,bridge)]
        conflicts=[];result=news._union(normalized,conflicts=conflicts)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['published_at'],self.start)
        self.assertEqual(result[0]['publication_basis'],'exact_source_publication');self.assertEqual(conflicts,[])

    def test_genuine_conflict_component_is_quarantined_without_blocking_independent_news(self):
        rows=[{**self.article,'uuid':'bad-a','link':'https://example.com/old','datePublished':self.start},
              {**self.article,'uuid':'bad-b','link':'https://example.com/new','datePublished':'2026-11-02T10:01:00-05:00'},
              {**self.article,'uuid':'bad-a','link':'https://example.com/new'},
              {**self.article,'uuid':'good','link':'https://example.com/good'}]
        result=news.collect_articles(self.start,self.end,self.root,fetcher=lambda ticker:rows if ticker=='NVDA' else [])
        self.assertEqual([row['article_id'] for row in result['articles']],['good'])
        self.assertEqual(result['coverage']['NVDA']['conflict_count'],1);self.assertEqual(result['coverage']['NVDA']['status'],'partial')
        path=next((self.root/'publication-conflicts').glob('*.json'));audit=json.loads(path.read_text())
        self.assertEqual(len(audit['quarantined_groups']),1);self.assertEqual(len(audit['quarantined_groups'][0]['records']),3)
        self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_unresolvable_provider_timestamp_conflicts_are_never_published(self):
        rows=[self.article,{**self.article,'providerPublishTime':datetime.fromisoformat(self.start).timestamp()}]
        result=news.collect_articles(self.start,self.end,self.root,fetcher=lambda ticker:rows if ticker=='NVDA' else [])
        self.assertEqual(result['articles'],[]);self.assertEqual(result['coverage']['NVDA']['conflict_count'],1)


if __name__=='__main__':unittest.main()
