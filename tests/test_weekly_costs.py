import copy
import unittest

from live_testing.accounting import new_book
from live_testing.calendar import DEFAULT_CALENDAR as C
from live_testing.projection import build_feed
from live_testing.registry import MODEL_IDS
from tests.test_projection import populated, budget, call


def model_row(feed, model, week=0):
    return next(row for row in feed['weeks'][week]['tokens'] if row['model_id']==model)


class WeeklyCostTests(unittest.TestCase):
    def test_per_model_cost_includes_training_inference_embeddings_but_not_shared_news(self):
        book,asof=populated()
        ledger=budget({'train':call('moira2-top1','training',actual_cost_usd=.2,guard_cost_usd=.2),
            'infer':call('moira2-top1','inference',actual_cost_usd=.3,guard_cost_usd=.3),
            'embedding':call('moira2-top1','training',requested_model='text-embedding-3-large',
                actual_cost_usd=.04,guard_cost_usd=.04),
            'other':call('finmem',actual_cost_usd=.1,guard_cost_usd=.1),
            'shared':call('data','data_summary',actual_cost_usd=20.,guard_cost_usd=20.)})
        original=copy.deepcopy((book,ledger));feed=build_feed(book,asof=asof,budget_ledger=ledger)
        self.assertAlmostEqual(model_row(feed,'moira2-top1')['estimated_cost_usd'],.54)
        self.assertEqual(model_row(feed,'moira2-top1')['cost_coverage'],'complete')
        self.assertAlmostEqual(model_row(feed,'finmem')['estimated_cost_usd'],.1)
        self.assertEqual((book,ledger),original)
        self.assertNotIn('estimated_cost_usd',feed['weeks'][0]['shared_tokens'])

    def test_unknown_transport_reservation_is_separate_from_measured_cost(self):
        book,asof=populated();ledger=budget({'known':call(actual_cost_usd=.05,guard_cost_usd=.05),
            'failed':call(status='failed',actual_cost_usd=None,guard_cost_usd=.15)})
        row=model_row(build_feed(book,asof=asof,budget_ledger=ledger),'finmem')
        self.assertAlmostEqual(row['estimated_cost_usd'],.05)
        self.assertAlmostEqual(row['reserved_cost_usd'],.15)
        self.assertEqual(row['cost_coverage'],'partial')
        self.assertIsNone(row['total_tokens'])
        self.assertEqual(ledger['calls']['failed']['guard_cost_usd'],.15)

    def test_unmeasured_import_never_becomes_a_free_call_or_invented_token_estimate(self):
        book,asof=populated();book['imported_usage']=[{'source_call_id':'old-unknown',
            'model_id':'codex','stage':'inference','started_at':C.release_nodes[0].isoformat(),
            'status':'usage-unavailable'}]
        row=model_row(build_feed(book,asof=asof,budget_ledger=budget()),'codex')
        self.assertIsNone(row['estimated_cost_usd']);self.assertEqual(row['cost_coverage'],'unavailable')
        ledger=budget({'known':call('codex',actual_cost_usd=.02,guard_cost_usd=.02)})
        row=model_row(build_feed(book,asof=asof,budget_ledger=ledger),'codex')
        self.assertEqual(row['estimated_cost_usd'],.02);self.assertEqual(row['cost_coverage'],'partial')

    def test_rates_use_the_recorded_provider_model_and_cached_tokens_without_double_counting(self):
        book,asof=populated()
        ledger=budget({'priced':call('moira',requested_model='gpt-4.1-mini',actual_cost_usd=None,
            guard_cost_usd=.01,input_tokens=1000,cached_input_tokens=800,output_tokens=10,total_tokens=1010)})
        row=model_row(build_feed(book,asof=asof,budget_ledger=ledger),'moira')
        self.assertAlmostEqual(row['estimated_cost_usd'],.000176)
        self.assertEqual(row['cost_coverage'],'complete')
        ledger['calls']['priced']['cached_input_tokens']=None
        row=model_row(build_feed(book,asof=asof,budget_ledger=ledger),'moira')
        self.assertAlmostEqual(row['estimated_cost_usd'],.000416)
        self.assertEqual(row['cost_coverage'],'partial')

    def test_costs_follow_actual_new_york_request_week_and_exclude_future_requests(self):
        book=new_book(MODEL_IDS,['AAA','BBB'])
        ledger=budget({'prior':call(started_at='2026-11-08T23:59:00-05:00',actual_cost_usd=.1,guard_cost_usd=.1),
            'new':call(started_at='2026-11-09T00:00:00-05:00',actual_cost_usd=.2,guard_cost_usd=.2),
            'future':call(started_at='2026-11-09T10:00:00-05:00',actual_cost_usd=.3,guard_cost_usd=.3)})
        feed=build_feed(book,asof='2026-11-09T09:30:00-05:00',budget_ledger=ledger)
        self.assertEqual([model_row(feed,'finmem',week)['estimated_cost_usd'] for week in (0,1)],[.1,.2])

    def test_verified_zero_differs_from_missing_cost_evidence_and_invalid_fees_reject(self):
        book,asof=populated()
        no_journal=model_row(build_feed(book,asof=asof),'finmem')
        verified=model_row(build_feed(book,asof=asof,budget_ledger=budget()),'finmem')
        self.assertIsNone(no_journal['estimated_cost_usd']);self.assertEqual(no_journal['cost_coverage'],'unavailable')
        self.assertEqual(verified['estimated_cost_usd'],0.);self.assertEqual(verified['cost_coverage'],'complete')
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):build_feed(book,asof=asof,budget_ledger=budget({'bad':call(actual_cost_usd=value)}))


if __name__=='__main__':unittest.main()
