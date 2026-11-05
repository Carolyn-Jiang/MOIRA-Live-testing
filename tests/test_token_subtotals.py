"""Known provider evidence stays visible without fabricating missing usage."""
import unittest

from live_testing.projection import build_feed, TOKEN_FIELDS
from tests.test_projection import populated, budget, call


def row(feed, model):
    return next(r for r in feed['weeks'][0]['tokens'] if r['model_id'] == model)


class TokenSubtotalTests(unittest.TestCase):
    def test_two_unknown_imports_do_not_hide_later_measured_codex_subtotal(self):
        book, asof = populated()
        book['imported_usage'] = [dict(source_call_id='synthetic-old-'+str(i),model_id='codex',stage='inference',
            started_at='2026-11-02T09:49:00-05:00',status='usage-unavailable') for i in range(2)]
        result = row(build_feed(book,asof=asof,budget_ledger=budget({'new':call('codex')})), 'codex')
        self.assertIsNone(result['total_tokens'])
        self.assertEqual(result['recorded_tokens']['total_tokens'],130)
        self.assertEqual(result['token_coverage']['total_tokens'],'partial')
        self.assertEqual(result['recorded_tokens']['training_tokens'],0)
        self.assertEqual(result['token_coverage']['training_tokens'],'complete')

    def test_one_failed_inference_preserves_complete_training_and_known_inference(self):
        book, asof = populated()
        ledger = budget({'train':call('moira2-top1','training'), 'infer':call('moira2-top1'),
            'unknown':call('moira2-top1',status='failed',input_tokens=None,cached_input_tokens=None,
                           output_tokens=None,total_tokens=None)})
        result = row(build_feed(book,asof=asof,budget_ledger=ledger),'moira2-top1')
        self.assertIsNone(result['total_tokens'])
        self.assertEqual(result['recorded_tokens']['total_tokens'],260)
        self.assertEqual(result['recorded_tokens']['training_tokens'],130)
        self.assertEqual(result['recorded_tokens']['inference_tokens'],130)
        self.assertEqual(result['token_coverage']['training_tokens'],'complete')
        self.assertEqual(result['token_coverage']['inference_tokens'],'partial')

    def test_missing_cached_field_does_not_invent_zero_or_hide_other_counts(self):
        book, asof = populated()
        ledger = budget({'old':call(cached_input_tokens=None),'new':call(cached_input_tokens=0)})
        result = row(build_feed(book,asof=asof,budget_ledger=ledger),'finmem')
        self.assertEqual(result['total_tokens'],260)
        self.assertIsNone(result['cached_input_tokens'])
        self.assertEqual(result['recorded_tokens']['cached_input_tokens'],0)
        self.assertEqual(result['token_coverage']['cached_input_tokens'],'partial')
        self.assertEqual(result['token_coverage']['total_tokens'],'complete')

    def test_entirely_unknown_usage_has_no_numeric_subtotal(self):
        book, asof = populated()
        result = row(build_feed(book,asof=asof,budget_ledger=budget({'unknown':call('codex',status='failed')})),'codex')
        for field in TOKEN_FIELDS[:4]:
            self.assertIsNone(result['recorded_tokens'][field])
            self.assertEqual(result['token_coverage'][field],'unavailable')

    def test_unknown_purpose_cannot_be_counted_as_zero_training_or_inference(self):
        book, asof = populated()
        result = row(build_feed(book,asof=asof,budget_ledger=budget({'ambiguous':call('codex','unknown-stage')})),'codex')
        for field in ('training_tokens','inference_tokens'):
            self.assertIsNone(result[field]);self.assertIsNone(result['recorded_tokens'][field])
            self.assertEqual(result['token_coverage'][field],'unavailable')
        self.assertEqual(result['recorded_tokens']['total_tokens'],130)

    def test_zero_requires_available_journal_or_measured_zero(self):
        book, asof = populated()
        unknown = row(build_feed(book,asof=asof),'codex')
        verified = row(build_feed(book,asof=asof,budget_ledger=budget({})),'codex')
        self.assertTrue(all(v is None for v in unknown['recorded_tokens'].values()))
        self.assertTrue(all(verified['recorded_tokens'][field] == 0 for field in TOKEN_FIELDS))
        self.assertTrue(all(verified['token_coverage'][field] == 'complete' for field in TOKEN_FIELDS))
        self.assertIsNone(verified['trading_tokens'])
        self.assertEqual(verified['token_coverage']['trading_tokens'], 'unavailable')

    def test_partial_week_is_recognized_even_if_every_full_total_is_unknown(self):
        from live_testing.registry import MODEL_IDS
        book, asof = populated();calls={}
        for model in MODEL_IDS:
            calls[model+'-known']=call(model,'unknown-stage')
            calls[model+'-missing']=call(model,'unknown-stage',status='failed')
        feed=build_feed(book,asof=asof,budget_ledger=budget(calls))
        self.assertTrue(all(row[field] is None for row in feed['weeks'][0]['tokens'] for field in TOKEN_FIELDS))
        self.assertEqual(feed['weeks'][0]['tokens_coverage'],'partial')


if __name__ == '__main__':
    unittest.main()
