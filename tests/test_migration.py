import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from live_testing.accounting import new_book
from live_testing.calendar import DEFAULT_CALENDAR as C
from live_testing.registry import MODEL_IDS,TICKERS,BACKBONES
from live_testing.migration import import_legacy_actions,_Source
from live_testing.projection import build_feed


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()}


def legacy(directory):
    state={"schema":"finreason.task2.phrase3-private/1","phase_id":"phrase-iii-2026-11-02",
           "published":{},"sealed":{},"teams":{"private":"DO NOT COPY LEGACY PARTICIPANTS"}}
    budget={"calls":{}}
    runtime={model:{"attempts":{},"decisions":{}} for model in ['finmem','moira']}
    for i,at in enumerate(C.release_nodes[:2]):
        rid=at.strftime('%Y-%m-%d_%H:%M');pub=at+timedelta(minutes=1);generated=at+timedelta(minutes=2)
        rows=[{"ticker":t,"timestamp":at.isoformat(),"price":100.+i,"currency":"USD","price_basis":"09:30 minute open" if i==0 else "previous minute close","bar_start":(at if i==0 else at-timedelta(minutes=1)).isoformat(),"status":"available"} for t in TICKERS]
        news=[{"ticker":t,"timestamp":at.isoformat(),"summary":"Original ticker-level summary protocol."} for t in TICKERS]
        release={"timestamp":at.isoformat(),"prices":{r['ticker']:r['price'] for r in rows},"price_rows":rows,"news_rows":news}
        write(directory/f'release-{i+1}.json',release)
        record={"published_at":pub.isoformat(),"file_sha256":{}}
        for kind,data in [('price',rows),('news',news)]:
            name=f'2026-11-02_{at:%H-%M}_NY_{kind}.parquet';pd.DataFrame(data).to_parquet(directory/name,index=False)
            record[kind+'_url']='https://huggingface.co/datasets/example/resolve/main/folder/'+name
            record['file_sha256'][kind]=hashlib.sha256((directory/name).read_bytes()).hexdigest()
        state['published'][rid]=record;state['sealed'][rid]={}
        for model,old in [('codex','Codex'),('finmem','FinMem'),('moira','Moira')]:
            seal={"action":{"stock1":TICKERS[0],"stock2":TICKERS[1],"action":"LS"},"sealed_at":generated.isoformat(),"model":BACKBONES[model]}
            state['sealed'][rid][old]=seal
            if model!='codex':
                started=at+timedelta(seconds=70)
                runtime[model]['attempts'][rid]={"status":"complete","started_at":started.isoformat()}
                runtime[model]['decisions'][rid]={"action":{**seal['action'],"generated_at":generated.isoformat(),"model":BACKBONES[model]}}
                budget['calls'][f'{model}-{i}']={"phase":model+'-live',"model":"gpt-4.1-mini","started_at":(started+timedelta(seconds=1)).isoformat(),"status":"completed","usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":10}},"actual_cost_usd":.001}
    write(directory/'state.json',state);write(directory/'native/api-budget.json',budget)
    for model,record in runtime.items():write(directory/f'native/{model}/runtime-records.json',record)
    write(directory/'native/finmem/qdrant-live/DO-NOT-LOAD.json',{"private":"not a migration source"})
    return state


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name);self.state=legacy(self.source);self.book=new_book(MODEL_IDS,TICKERS)

    def test_six_real_actions_and_two_prices_copied_sources_unchanged(self):
        before=hashes(self.source);report=import_legacy_actions(self.book,self.source)
        self.assertEqual(hashes(self.source),before)
        self.assertEqual(report['copied_decisions'],6)
        self.assertEqual(len(self.book['valuations']),2)
        for model,old in [('codex','Codex'),('finmem','FinMem'),('moira','Moira')]:
            self.assertEqual(len(self.book['decisions'][model]),2)
            for at in C.release_nodes[:2]:
                record=self.book['decisions'][model][at.isoformat()];rid=at.strftime('%Y-%m-%d_%H:%M')
                self.assertEqual(record['generated_at'],self.state['sealed'][rid][old]['sealed_at'])
                self.assertEqual(record['mode'],'live')
                self.assertEqual(record['audit']['input_protocol'],'legacy_input_summary_protocol')
        self.assertTrue(all(self.book['decisions'][m]=={} for m in MODEL_IDS[3:]))
        self.assertTrue(all(a['history']==[] and a['equity']==1. for a in self.book['accounts'].values()))
        self.assertNotIn('DO NOT COPY LEGACY PARTICIPANTS',json.dumps(self.book))
        self.assertFalse(any('qdrant' in path for path in report['source_file_sha256']))

    def test_unknown_imported_codex_usage_stays_unavailable(self):
        import_legacy_actions(self.book,self.source)
        calls=self.book['imported_usage'];self.assertEqual(len(calls),6)
        self.assertTrue(all(r['status']=='usage-unavailable' for r in calls if r['model_id']=='codex'))
        self.assertTrue(all(r['input_tokens']==100 and r['cached_input_tokens']==10 and r['output_tokens']==20 for r in calls if r['model_id'] in ['finmem','moira']))
        feed=build_feed(self.book,asof=C.release_nodes[1]+timedelta(minutes=5))
        rows={r['model_id']:r for r in feed['weeks'][0]['tokens']}
        self.assertIsNone(rows['codex']['total_tokens'])
        self.assertEqual(rows['finmem']['total_tokens'],240)
        self.assertEqual(feed['weeks'][0]['tokens_coverage'],'partial')

    def test_import_is_idempotent_and_cannot_extend_to_wrong_third_slot(self):
        report=import_legacy_actions(self.book,self.source);before=copy.deepcopy(self.book)
        self.assertEqual(import_legacy_actions(self.book,self.source),report)
        self.assertEqual(self.book,before)
        with self.assertRaises(ValueError):import_legacy_actions(self.book,self.source,limit=3)

    def test_missing_usage_source_never_invents_tokens(self):
        (self.source/'native/api-budget.json').unlink()
        import_legacy_actions(self.book,self.source)
        self.assertTrue(all(r['status']=='usage-unavailable' for r in self.book['imported_usage']))

    def test_real_publisher_folder_urls_map_to_existing_prefixed_local_files(self):
        state=json.loads((self.source/'state.json').read_text())
        for at in C.release_nodes[:2]:
            rid=at.strftime('%Y-%m-%d_%H:%M');folder=f'2026-11-02_{at:%H-%M}_NY'
            for kind in ['price','news']:
                state['published'][rid][kind+'_url']=f'https://huggingface.co/datasets/example/resolve/main/{folder}/{kind}.parquet'
        write(self.source/'state.json',state);before=hashes(self.source)
        report=import_legacy_actions(self.book,self.source)
        self.assertEqual(report['copied_decisions'],6)
        self.assertEqual(hashes(self.source),before)
        self.assertFalse((self.source/'price.parquet').exists())
        self.assertFalse((self.source/'news.parquet').exists())
        for source in report['sources']:
            self.assertTrue(source['published_file_proofs']['price']['filename'].startswith('2026-11-02_'))

    def test_json_parquet_mismatch_rejects_without_new_book_mutation(self):
        path=self.source/'release-1.json';r=json.loads(path.read_text());r['prices'][TICKERS[0]]=999;r['price_rows'][0]['price']=999;write(path,r)
        before=copy.deepcopy(self.book)
        with self.assertRaises(ValueError):import_legacy_actions(self.book,self.source)
        self.assertEqual(self.book,before)

    def test_published_parquet_hash_mismatch_rejected(self):
        path=next(self.source.glob('*_price.parquet'));path.write_bytes(path.read_bytes()+b'changed')
        before=copy.deepcopy(self.book)
        with self.assertRaises(ValueError):import_legacy_actions(self.book,self.source)
        self.assertEqual(self.book,before)

    def test_late_or_wrong_backbone_cannot_be_copied_as_live(self):
        rid=C.release_nodes[0].strftime('%Y-%m-%d_%H:%M');state=json.loads((self.source/'state.json').read_text())
        state['sealed'][rid]['Codex']['sealed_at']=C.action_window(C.release_nodes[0]).entry_at.isoformat();write(self.source/'state.json',state)
        with self.assertRaises(ValueError):import_legacy_actions(self.book,self.source)
        self.assertTrue(all(not rows for rows in self.book['decisions'].values()))

    def test_advancing_source_aborts_atomically(self):
        before=copy.deepcopy(self.book)
        with patch.object(_Source,'unchanged',return_value=False):
            with self.assertRaises(ValueError):import_legacy_actions(self.book,self.source)
        self.assertEqual(self.book,before)


if __name__=='__main__':unittest.main()
