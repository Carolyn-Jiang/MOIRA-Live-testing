import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from live_testing.publish import publish_release,PublicationConflict
from live_testing.registry import TICKERS


class FakeHub:
    token='fake-never-published-token'
    def __init__(self):self.private=True;self.sha='initial';self.files={};self.calls=[];self.snapshots={self.sha:{}}
    def repo_info(self,*args,**kwargs):return SimpleNamespace(private=self.private,sha=self.sha)
    def auth_check(self,**kwargs):
        if kwargs.get('write') is not True:raise AssertionError('Write authentication must be checked')
    def read(self,name,revision):return self.snapshots[revision].get(name)
    def create_commit(self,**kwargs):
        self.calls.append(kwargs)
        if kwargs['parent_commit']!=self.sha:raise PublicationConflict('CAS conflict')
        self.files=dict(self.files)
        for operation in kwargs['operations']:self.files[operation.path_in_repo]=operation.path_or_fileobj
        self.sha='commit-'+str(len(self.calls));self.snapshots[self.sha]=dict(self.files)
        return SimpleNamespace(oid=self.sha)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.api=FakeHub()
        self.start='2026-11-02T09:30:00-05:00';self.end='2026-11-02T10:30:00-05:00'
        self.prices=[{'ticker':ticker,'timestamp':self.end,'price':100.+i} for i,ticker in enumerate(TICKERS)]
        self.news=[{'ticker':'NVDA','article_id':'article-1','published_at':'2026-11-02T10:00:00-05:00',
            'summary':'The company announced a product update, while the headline alone provides limited operating detail.',
            'window_start':self.start,'window_end':self.end,'content_basis':'partial_headline','summary_tokens':18}]
    def publish(self,**kwargs):return publish_release(self.prices,self.news,release_id='2026-11-02_10-30_NY',window_start=self.start,
        window_end=self.end,cache_root=self.root,api=self.api,reader=self.api.read,**kwargs)

    def test_one_private_CAS_commit_contains_both_parquets_release_and_manifest(self):
        result=self.publish();self.assertFalse(result['idempotent']);self.assertEqual(len(self.api.calls),1)
        request=self.api.calls[0];self.assertEqual(request['parent_commit'],'initial');self.assertEqual(request['repo_type'],'dataset')
        self.assertEqual(set(self.api.files),{'data/2026-11-02_10-30_NY/price.parquet','data/2026-11-02_10-30_NY/news.parquet','data/2026-11-02_10-30_NY/release.json','releases.json'})
        self.assertTrue(self.api.files['data/2026-11-02_10-30_NY/price.parquet'].startswith(b'PAR1'))
        self.assertNotIn(b'fake-never-published-token',b''.join(self.api.files.values()))
        audit=next((self.root/'publication-audit').glob('*.json'));self.assertEqual(audit.stat().st_mode&0o777,0o600)
        self.assertEqual(json.loads(audit.read_text())['commit'],result['commit'])

    def test_identical_retry_verifies_bytes_without_another_commit(self):
        first=self.publish();second=self.publish();self.assertTrue(second['idempotent']);self.assertEqual(second['commit'],first['commit']);self.assertEqual(len(self.api.calls),1)
        self.prices[0]['price']+=1
        with self.assertRaisesRegex(PublicationConflict,'immutable'):self.publish()
        self.assertEqual(len(self.api.calls),1)

    def test_public_repo_and_wrong_dataset_are_rejected_before_write(self):
        self.api.private=False
        with self.assertRaisesRegex(RuntimeError,'must remain private'):self.publish()
        self.assertEqual(self.api.calls,[])
        with self.assertRaises(ValueError):self.publish(dataset='someone/public')

    def test_incomplete_prices_and_outside_source_publication_never_publish(self):
        self.prices.pop()
        with self.assertRaises(ValueError):self.publish()
        self.prices.append({'ticker':TICKERS[-1],'timestamp':self.end,'price':100.})
        self.news[0]['published_at']=self.end
        with self.assertRaisesRegex(ValueError,'publication window'):self.publish()
        self.assertEqual(self.api.calls,[])

    def test_corrupted_existing_remote_bytes_and_concurrent_HEAD_are_not_success(self):
        self.publish();self.api.snapshots[self.api.sha]['data/2026-11-02_10-30_NY/news.parquet']=b'corrupt'
        with self.assertRaisesRegex(PublicationConflict,'byte verification'):self.publish()
        self.api=FakeHub();original=self.api.repo_info;count=0
        def racing(*args,**kwargs):
            nonlocal count
            count+=1
            info=original(*args,**kwargs)
            if count==2:info.sha='concurrent'
            return info
        self.api.repo_info=racing
        with self.assertRaisesRegex(PublicationConflict,'advanced'):self.publish()
        self.assertEqual(self.api.calls,[])

    def test_true_zero_article_window_has_empty_parquet_and_explicit_coverage(self):
        self.news=[];result=self.publish(coverage={'NVDA':{'status':'partial','article_count':0}})
        self.assertEqual(result['news_rows'],0);self.assertEqual(result['coverage']['NVDA']['status'],'partial')


if __name__=='__main__':unittest.main()
