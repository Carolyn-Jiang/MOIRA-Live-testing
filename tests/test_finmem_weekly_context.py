"""Causal producer tests use only synthetic local books and fake native files."""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from live_testing.accounting import (new_book, record_decision, record_valuation,
    settle_due, matured_feedback, risk_metrics)
from live_testing.calendar import DEFAULT_CALENDAR as C, NY
from live_testing.data import round_id
from live_testing.finmem_weekly_context import build_context, validate_context
from live_testing.registry import TICKERS
from live_testing.storage import digest, read_json, write_private


class WeeklyContextProducerTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.epoch=self.root/'models/finmem/epochs/synthetic-source'
        self.state_path=self.epoch/'live-state.json'
        self.cutoff=datetime(2026,11,6,16,tzinfo=NY)
        self.observation=C.release_nodes[29]
        self.book=new_book(['finmem'],TICKERS)
        for i,at in enumerate(C.release_nodes[:30]):
            record_decision(self.book,'finmem',at,
                {'stock1':TICKERS[0],'stock2':TICKERS[1],'action':'LS'},
                generated_at=at+timedelta(seconds=5),input_asof=at,
                audit={'native_generation':str(self.epoch)})
            values={ticker:100.+j*5+i for j,ticker in enumerate(TICKERS)}
            record_valuation(self.book,at,values,observed_at=at+timedelta(seconds=1))
        settle_due(self.book,self.cutoff)
        self.native={'decisions':{},'feedback':{}}
        for i,(stamp,decision) in enumerate(self.book['decisions']['finmem'].items()):
            reason='Synthetic matured reason '+str(i) if i<28 else 'PENDING_REASON_SENTINEL_'+str(i)
            self.native['decisions'][round_id(stamp)]={'action':{**decision['action'],
                'generated_at':decision['generated_at'],'reason':reason},'evidence_ids':[100+i]}
        self.matured=matured_feedback(self.book,'finmem',self.cutoff,observation_asof=self.cutoff)
        for row in self.matured:
            self.native['feedback'][round_id(row['release_at'])]={**copy.deepcopy(row),'reward':row['step_return']}
        write_private(self.state_path,self.native)

    def build(self,book=None):
        return build_context(self.root,book or self.book,self.cutoff,self.observation)

    def test_thirty_real_calendar_decisions_yield_only_twenty_eight_mature_experiences(self):
        before=copy.deepcopy(self.book);native_before=self.state_path.read_bytes()
        context,proof=self.build()
        self.assertEqual(len(self.book['decisions']['finmem']),30)
        self.assertEqual(len(self.book['accounts']['finmem']['history']),28)
        self.assertEqual(context['known_labels'],28)
        self.assertEqual(len(context['finalized_memory']),28)
        self.assertEqual(context['weekly_return'],risk_metrics(row['step_return'] for row in self.matured))
        self.assertEqual(context,validate_context(context))
        actual={row['release_at'] for row in context['finalized_memory']}
        self.assertEqual(actual,{row['release_at'] for row in self.matured})
        self.assertTrue(all('PENDING_REASON_SENTINEL' not in row['decision_reason'] for row in context['finalized_memory']))
        self.assertNotIn('PENDING_REASON_SENTINEL',json.dumps(context))
        self.assertEqual(len(proof['cases']),28)
        self.assertEqual(proof['context_sha256'],digest(context))
        self.assertEqual(proof['source_book_sha256'],digest(self.book))
        self.assertEqual(self.book,before);self.assertEqual(self.state_path.read_bytes(),native_before)
        self.assertNotIn('native_generation',json.dumps(context))

    def test_tampered_native_reward_or_quote_cannot_enter_a_selector_prompt(self):
        first=next(iter(self.native['feedback']))
        for kind in ('reward','entry'):
            native=copy.deepcopy(self.native)
            if kind=='reward':native['feedback'][first]['reward']+=.01
            else:native['feedback'][first]['entry_prices'][TICKERS[0]]+=1
            write_private(self.state_path,native)
            with self.assertRaises(RuntimeError):self.build()

    def test_native_action_or_actual_reason_must_match_the_sealed_source(self):
        first=next(iter(self.native['feedback']))
        for kind in ('action','generated_at','reason'):
            native=copy.deepcopy(self.native)
            if kind=='action':native['decisions'][first]['action']['action']='SL'
            elif kind=='generated_at':native['decisions'][first]['action']['generated_at']='2026-11-02T09:30:06-05:00'
            else:native['decisions'][first]['action']['reason']=''
            write_private(self.state_path,native)
            with self.assertRaises((ValueError,RuntimeError)):self.build()

    def test_unexited_late_available_or_invented_metrics_are_rejected(self):
        context,_=self.build()
        for kind in ('late','exit','return','metrics','count'):
            bad=copy.deepcopy(context)
            if kind=='late':bad['finalized_memory'][0]['available_at']='2026-11-06T16:00:01-05:00'
            elif kind=='exit':bad['finalized_memory'][0]['exit_at']='2026-11-09T09:30:00-05:00'
            elif kind=='return':bad['finalized_memory'][0]['step_return']+=.01
            elif kind=='metrics':bad['weekly_return']['cumulative_return']+=.01
            else:bad['known_labels']+=2
            with self.assertRaises(ValueError):validate_context(bad)

    def legacy_book(self):
        book=copy.deepcopy(self.book)
        for stamp in list(book['decisions']['finmem'])[:2]:
            decision=book['decisions']['finmem'][stamp]
            seal={'action':copy.deepcopy(decision['action']),'sealed_at':decision['generated_at']}
            decision['audit']={'input_protocol':'legacy_input_summary_protocol','source_original_seal':seal,
                'source_seal_sha256':digest(seal),'source_round_id':round_id(stamp)}
        return book

    def test_legacy_first_two_require_an_exact_seal_and_unique_native_match(self):
        book=self.legacy_book();context,proof=self.build(book)
        self.assertEqual(context['known_labels'],28);self.assertEqual(len(proof['cases']),28)
        bad=copy.deepcopy(book);first=next(iter(bad['decisions']['finmem']))
        bad['decisions']['finmem'][first]['audit']['source_original_seal']['sealed_at']='2026-11-02T09:30:07-05:00'
        with self.assertRaises(RuntimeError):self.build(bad)
        copy_path=self.root/'models/finmem/epochs/ambiguous-copy/live-state.json'
        write_private(copy_path,self.native)
        with self.assertRaises(RuntimeError):self.build(book)

    def test_missing_mature_native_feedback_is_not_an_empty_memory_fallback(self):
        native=copy.deepcopy(self.native);native['feedback'].pop(next(iter(native['feedback'])))
        write_private(self.state_path,native)
        with self.assertRaises(RuntimeError):self.build()


if __name__=='__main__':unittest.main()
