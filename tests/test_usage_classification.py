import copy
import json
from pathlib import Path
import tempfile
import unittest

from live_testing.projection import _calls, build_feed
from live_testing.registry import LEGACY_BACKBONES, MODEL_IDS
from live_testing.storage import digest
from live_testing.usage_classification import (build_usage_catalog, build_catalog_from_state,
                                             FIELDS, validate_catalog)
from tests.test_projection import populated, budget, call


class UsageClassificationTests(unittest.TestCase):
    def legacy_codex_usage(self, stamp, decision):
        rid = stamp[:10]+'_'+stamp[11:16]
        seal = {'action': copy.deepcopy(decision['action']),
                'sealed_at': decision['generated_at'], 'model': LEGACY_BACKBONES['codex']}
        identity = f'legacy-action:codex:{rid}:{digest(seal)}'
        audit = {'input_protocol': 'legacy_input_summary_protocol', 'source_round_id': rid,
                 'source_seal_sha256': digest(seal), 'source_original_seal': seal}
        row = {'source_call_id': identity, 'model_id': 'codex', 'stage': 'inference',
               'started_at': seal['sealed_at'], 'status': 'usage-unavailable',
               'input_tokens': None, 'cached_input_tokens': None, 'output_tokens': None,
               'total_tokens': None, 'source_protocol': 'legacy_input_summary_protocol',
               'source_round_id': rid}
        return identity, audit, row

    def fixture(self):
        book, at = populated(1)
        stamp = next(iter(book['decisions']['moira']))
        ledger = budget({
            'accepted': call('moira', 'replay'),
            'replaced': call('moira', 'replay'),
            'monitor': call('moira'), 'loo': call('moira'),
            'embedding': call('moira', requested_model='text-embedding-3-large'),
            'training': call('moira', 'training'),
            'failed': call('moira', status='failed', actual_cost_usd=None, guard_cost_usd=.2)})
        proof = [{'source_call_id': 'accepted', 'model_id': 'moira', 'kind': 'direction_decision',
                  'release_at': stamp, 'decision_sha256': digest(book['decisions']['moira'][stamp])},
                 {'source_call_id': 'monitor', 'model_id': 'moira', 'kind': 'relationship_monitor'},
                 {'source_call_id': 'loo', 'model_id': 'moira', 'kind': 'loo_attribution'}]
        return book, at, ledger, proof

    def test_accepted_only_trading_preserves_every_other_token_and_reservation(self):
        book, at, ledger, proof = self.fixture()
        original = copy.deepcopy((book, ledger))
        catalog = build_usage_catalog(book, _calls(book, ledger, None), evidence=proof)
        feed = build_feed(book, asof=at, budget_ledger=ledger, usage_catalog=catalog)
        row = next(x for x in feed['weeks'][0]['tokens'] if x['model_id'] == 'moira')
        self.assertEqual(row['trading_tokens'], 130)
        self.assertEqual(row['direction_decision_tokens'], 130)
        self.assertEqual(row['pair_reselection_tokens'], 0)
        self.assertEqual(row['training_tokens'], 130)
        self.assertEqual(row['relationship_monitor_tokens'], 130)
        self.assertEqual(row['loo_attribution_tokens'], 130)
        self.assertIsNone(row['other_inference_tokens'])
        self.assertEqual(row['recorded_tokens']['other_inference_tokens'], 260)
        self.assertEqual(row['token_coverage']['other_inference_tokens'], 'partial')
        self.assertEqual(row['recorded_tokens']['total_tokens'], 780)
        self.assertAlmostEqual(row['estimated_cost_usd'], .06)
        self.assertEqual(row['reserved_cost_usd'], .2)
        self.assertEqual((book, ledger), original)
        public = json.dumps(feed)
        for private in ('source_call_id', 'decision_sha256', 'private_prompt', 'replaced', 'accepted'):
            self.assertNotIn(private, public)

    def test_missing_catalog_never_relabels_all_inference_as_trading(self):
        book, at, ledger, _ = self.fixture()
        row = next(x for x in build_feed(book, asof=at, budget_ledger=ledger)['weeks'][0]['tokens']
                   if x['model_id'] == 'moira')
        for field in FIELDS:
            self.assertIsNone(row[field]); self.assertEqual(row['token_coverage'][field], 'unavailable')
        self.assertEqual(row['recorded_tokens']['inference_tokens'], 650)

    def test_missing_current_decision_proof_is_partial_not_zero(self):
        book, at, ledger, proof = self.fixture()
        stamp = next(iter(book['decisions']['moira']))
        catalog = build_usage_catalog(book, _calls(book, ledger, None), evidence=proof[1:],
            unresolved={'moira': {'direction_decision': [stamp]}})
        row = next(x for x in build_feed(book, asof=at, budget_ledger=ledger, usage_catalog=catalog)['weeks'][0]['tokens']
                   if x['model_id'] == 'moira')
        self.assertIsNone(row['trading_tokens'])
        self.assertEqual(row['recorded_tokens']['trading_tokens'], 0)
        self.assertEqual(row['token_coverage']['trading_tokens'], 'partial')

    def test_stale_book_mixed_model_training_and_non1530_reselection_are_rejected(self):
        book, at, ledger, proof = self.fixture()
        calls = _calls(book, ledger, None)
        for change in ('model', 'training', 'stale', 'reselection'):
            item = copy.deepcopy(proof[0])
            if change == 'model': item['model_id'] = 'finmem'
            if change == 'training': item['source_call_id'] = 'training'
            if change == 'stale': item['decision_sha256'] = '0' * 64
            if change == 'reselection': item['kind'] = 'pair_reselection'
            with self.assertRaises(ValueError): build_usage_catalog(book, calls, evidence=[item])
        catalog = build_usage_catalog(book, calls, evidence=proof)
        altered = copy.deepcopy(book); altered['decisions']['moira'][proof[0]['release_at']]['mode'] = 'changed'
        with self.assertRaises(ValueError): validate_catalog(catalog, altered, calls)

    def test_filesystem_reader_requires_real_request_hash_and_accepted_native_result(self):
        stamp = '2026-11-02T09:30:00-05:00'; generated = '2026-11-02T14:31:02+00:00'
        action = {'stock1': 'AAA', 'stock2': 'BBB', 'action': 'LS'}
        accepted = {'release_at': stamp, 'generated_at': generated, 'action': action}
        book = {'decisions': {model: {} for model in MODEL_IDS}}
        book['decisions']['moira'][stamp] = accepted
        payload = {'current_observation': {'timestamp': stamp}}
        request = {'model': 'gpt-4.1-mini', 'instructions': 'fixed system', 'input': json.dumps(payload)}
        row = call('moira', started_at='2026-11-02T14:31:01+00:00', requested_model='gpt-4.1-mini',
                   request_sha256=digest(request), response_id='provider-receipt',
                   phase='not-a-purpose-proof')
        ledger = budget({'receipt': row})
        native = {'decisions': {'native-round': {'timestamp': stamp,
                  'action': {**action, 'generated_at': generated, 'reason': 'actual reason'}}}}
        audit = {'model_id': 'moira', 'scope': 'live', 'task_type': 'daily_direction',
                 'started_at': '2026-11-02T14:31:00+00:00', 'generated_at': generated,
                 'system_prompt': 'fixed system', 'payload': payload, 'payload_sha256': digest(payload),
                 'response': json.dumps({'action': 'LONG_SHORT', 'direction_reason': 'actual reason'})}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'models/moira').mkdir(parents=True)
            (root/'api-audit/http-requests').mkdir(parents=True)
            (root/'models/moira/live-state.json').write_text(json.dumps(native))
            (root/'models/moira/model-calls.jsonl').write_text(json.dumps(audit)+'\n')
            path = root/'api-audit/http-requests/receipt.json'
            wire = {'request_payload': request, 'model_id': 'moira', 'response_id': 'provider-receipt'}
            path.write_text(json.dumps(wire))
            catalog = build_catalog_from_state(root, book, ledger)
            self.assertEqual(catalog['calls']['receipt']['kind'], 'direction_decision')
            wire['request_payload']['instructions'] = 'altered'
            path.write_text(json.dumps(wire))
            broken = build_catalog_from_state(root, book, ledger)
            self.assertNotIn('receipt', broken['calls'])
            self.assertEqual(broken['unresolved']['moira']['direction_decision'], [stamp])

    def test_failed_real_loo_request_is_unknown_attribution_not_free_other_usage(self):
        book = {'decisions': {model: {} for model in MODEL_IDS}}
        request = {'model': 'gpt-4.1-mini', 'input': '{}'}
        ledger = budget({'failed-loo': call('moira2-top1', status='failed', request_sha256=digest(request))})
        audit = {'scope': 'live', 'task_type': 'daily_direction_leave_one_out',
                 'model_id': 'moira2-top1', 'request': request, 'request_sha256': digest(request),
                 'started_at': '2026-11-02T10:30:59-05:00', 'generated_at': '2026-11-02T10:31:02-05:00',
                 'status': 'failed', 'payload': {}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = root/'models/moira2-top1/calls'; directory.mkdir(parents=True)
            (directory/'attempt.json').write_text(json.dumps(audit))
            catalog = build_catalog_from_state(root, book, ledger)
            self.assertEqual(catalog['calls']['failed-loo']['kind'], 'loo_attribution')

    def test_finmem_reads_proved_receipts_from_both_historical_and_active_epochs(self):
        book = {'decisions': {model: {} for model in MODEL_IDS}}
        ledger = budget({})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, stamp in enumerate(('2026-11-02T09:30:00-05:00', '2026-11-02T10:30:00-05:00')):
                epoch = root/'models/finmem/epochs'/str(i); epoch.mkdir(parents=True)
                generated = stamp.replace(':30:00', ':31:02')
                started = stamp.replace(':30:00', ':31:01')
                action = {'stock1': 'AAA', 'stock2': 'BBB', 'action': 'LS'}
                book['decisions']['finmem'][stamp] = {'release_at': stamp, 'generated_at': generated,
                    'action': action, 'audit': {'native_generation': str(epoch)}}
                policy = {'round_id': stamp[:10]+'_'+stamp[11:16], 'generated_at': generated,
                          'started_at': stamp, 'accepted': True, 'returned_action': 'LS'}
                (epoch/'live-policy-attempts.jsonl').write_text(json.dumps(policy)+'\n')
                request = {'model': 'gpt-4.1-mini', 'messages': [{'role': 'user', 'content': 'frozen'}],
                           'response_format': {'json_schema': {'name': 'finmem_pair_action'}}}
                identity = 'receipt-'+str(i)
                ledger['calls'][identity] = call('finmem', requested_model='gpt-4.1-mini', started_at=started,
                    request_sha256=digest(request), response_id='actual-'+str(i))
                directory = root/'api-audit/http-requests'; directory.mkdir(parents=True, exist_ok=True)
                (directory/(identity+'.json')).write_text(json.dumps({'request_payload': request,
                    'model_id': 'finmem', 'response_id': 'actual-'+str(i)}))
            (root/'models/finmem/active-epoch.json').write_text(json.dumps({'directory': str(epoch)}))
            catalog = build_catalog_from_state(root, book, ledger)
            self.assertEqual(set(catalog['calls']), {'receipt-0', 'receipt-1'})
            self.assertFalse(catalog['unresolved']['finmem']['direction_decision'])
            (root/'api-audit/http-requests/receipt-0.json').unlink()
            missing = build_catalog_from_state(root, book, ledger)
            self.assertNotIn('receipt-0', missing['calls'])
            self.assertEqual(missing['unresolved']['finmem']['direction_decision'], ['2026-11-02T09:30:00-05:00'])

    def test_reissued_codex_first_two_use_new_receipts_without_losing_old_usage(self):
        book, at = populated(2)
        ledger = budget({}); audits = []; book['imported_usage'] = []
        old_ids = set(); new_ids = set()
        for i, (stamp, decision) in enumerate(book['decisions']['codex'].items()):
            identity, _, row = self.legacy_codex_usage(stamp, decision)
            old_ids.add(identity); book['imported_usage'].append(row)
            # This is a new accepted retrospective generation, not the original
            # imported seal, even when the selected pair/action happen to match.
            decision['mode'] = 'retrospective_point_in_time'
            decision['audit'] = {'authorization': 'Explicit model migration', 'input_sha256': 'a'*64}
            receipt = f'new-receipt-{i}'; new_ids.add(receipt)
            ledger['calls'][receipt] = call('codex', 'replay', started_at=decision['generated_at'],
                requested_model='gpt-6.1-sol', response_id=f'provider-{i}')
            audits.append({'model_id': 'codex', 'payload': {'observation': {'timestamp': stamp}},
                'response': copy.deepcopy(decision['action']), 'generated_at': decision['generated_at'],
                'response_id': f'provider-{i}'})
        original = copy.deepcopy((book, ledger))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'api-audit').mkdir()
            (root/'api-audit/model-calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in audits))
            catalog = build_catalog_from_state(root, book, ledger)
        self.assertEqual(set(catalog['calls']), new_ids)
        self.assertFalse(catalog['unresolved']['codex']['direction_decision'])
        self.assertEqual({x['id'] for x in _calls(book, ledger, None)}, old_ids | new_ids)
        row = next(x for x in build_feed(book, asof=at, budget_ledger=ledger,
                   usage_catalog=catalog)['weeks'][0]['tokens'] if x['model_id'] == 'codex')
        self.assertEqual(row['trading_tokens'], 260)
        self.assertEqual(row['token_coverage']['trading_tokens'], 'complete')
        self.assertIsNone(row['total_tokens'])
        self.assertEqual(row['recorded_tokens']['total_tokens'], 260)
        self.assertEqual(row['token_coverage']['total_tokens'], 'partial')
        self.assertIsNone(row['other_inference_tokens'])
        self.assertEqual(row['token_coverage']['other_inference_tokens'], 'partial')
        self.assertEqual((book, ledger), original)

    def test_original_codex_import_keeps_unknown_trading_usage_partial(self):
        book, at = populated(1)
        stamp, decision = next(iter(book['decisions']['codex'].items()))
        identity, audit, row = self.legacy_codex_usage(stamp, decision)
        decision['audit'] = audit; book['imported_usage'] = [row]
        ledger = budget({})
        with tempfile.TemporaryDirectory() as temporary:
            catalog = build_catalog_from_state(Path(temporary), book, ledger)
        self.assertEqual(catalog['calls'][identity]['kind'], 'direction_decision')
        self.assertFalse(catalog['unresolved']['codex']['direction_decision'])
        tokens = next(x for x in build_feed(book, asof=at, budget_ledger=ledger,
                      usage_catalog=catalog)['weeks'][0]['tokens'] if x['model_id'] == 'codex')
        self.assertIsNone(tokens['trading_tokens'])
        self.assertEqual(tokens['token_coverage']['trading_tokens'], 'partial')
        self.assertEqual(tokens['recorded_tokens']['trading_tokens'], 0)
        self.assertEqual(tokens['other_inference_tokens'], 0)
        # Broken provenance cannot turn an unrelated old import into a receipt.
        for field in ('source_round_id', 'source_seal_sha256', 'generation', 'row_generation', 'action'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                altered = copy.deepcopy(book)
                proof = altered['decisions']['codex'][stamp]['audit']
                if field in ('source_round_id', 'source_seal_sha256'):
                    proof[field] = 'mismatch'
                elif field == 'generation':
                    altered['decisions']['codex'][stamp]['generated_at'] = '2026-11-02T09:31:00-05:00'
                elif field == 'row_generation':
                    altered['imported_usage'][0]['started_at'] = '2026-11-02T09:31:00-05:00'
                else:
                    altered['decisions']['codex'][stamp]['action']['action'] = 'F'
                rejected = build_catalog_from_state(Path(temporary), altered, ledger)
                self.assertNotIn(identity, rejected['calls'])
                self.assertEqual(rejected['unresolved']['codex']['direction_decision'], [stamp])

    def test_pure_catalog_automatically_marks_unproved_accepted_decisions(self):
        book, _, ledger, _ = self.fixture()
        catalog = build_usage_catalog(book, _calls(book, ledger, None))
        for model in MODEL_IDS:
            self.assertEqual(set(catalog['unresolved'][model]['direction_decision']), set(book['decisions'][model]))


if __name__ == '__main__':
    unittest.main()
