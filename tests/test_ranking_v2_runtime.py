"""Read-only v2 routing tests; all model/gateway calls are mocks."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from live_testing import ranking_v2_runtime as policy, runner
from live_testing.accounting import new_book
from live_testing.calendar import DEFAULT_CALENDAR
from live_testing.memory_v2 import PROTOCOL_VERSION
from live_testing.registry import MODEL_IDS, TICKERS
from live_testing.storage import file_hash, read_json, write_private


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve(); self.model = 'moira2-top3'
        self.directory = self.root/'models'/self.model
        prompt = 'Actual evaluated transferable prompt.'
        self.prepared = {'model_id': self.model, 'model': 'gpt-4.1-mini', 'status': 'ready',
            'memory_policy_protocol': PROTOCOL_VERSION, 'prompt_memory': {'active_prompt': prompt},
            'pair_memory': [],
            'frozen_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        self.report = {'objective': 'sample_sharpe_without_annualization', 'completed_cycles': 1, 'max_cycles': 8,
            'checkpoint': copy.deepcopy(self.prepared), 'cycles': [{'accepted': True,
                'baseline_metrics': {'sharpe_ratio': .1}, 'candidate_metrics': {'sharpe_ratio': .2}}]}
        self.report_path = self.root/'training/model/training-result.json'
        self.write_ready()
        self.runtime = runner.Runtime.__new__(runner.Runtime); self.runtime.root = self.root
        self.runtime.book = new_book(MODEL_IDS, TICKERS)
        self.runtime.state = {'feedback_applied': {m: [] for m in MODEL_IDS}}
        self.runtime.save = Mock(); self.runtime._gateway = SimpleNamespace(scoped=Mock(return_value=object()))
        self.at = DEFAULT_CALENDAR.release_nodes[0]
        self.obs = {'timestamp': self.at.isoformat(), 'round_id': 'synthetic-frozen-observation'}
        self.action = {'stock1': TICKERS[0], 'stock2': TICKERS[1], 'action': 'LS',
                       'generated_at': self.at.isoformat()}
    def write_ready(self):
        write_private(self.directory/'prepared.json', self.prepared)
        write_private(self.report_path, self.report)
        write_private(self.directory/'memory-policy.json', {'protocol': PROTOCOL_VERSION, 'status': 'ready',
            'model_id': self.model, 'checkpoint_sha256': file_hash(self.directory/'prepared.json'),
            'training_result_path': str(self.report_path), 'training_result_sha256': file_hash(self.report_path)})
    def refresh_report_hash(self):
        write_private(self.report_path, self.report)
        marker = read_json(self.directory/'memory-policy.json')
        marker['training_result_sha256'] = file_hash(self.report_path)
        write_private(self.directory/'memory-policy.json', marker)
    def test_ready_v2_bypasses_only_legacy_initial_training_marker(self):
        self.assertTrue(policy.ready(self.root, self.model))
        self.assertIsNone(self.runtime.require_variant_training(self.model))
        self.assertFalse((self.directory/'initial-training-ready.json').exists())
    def test_real_v2_decision_routes_with_same_frozen_observation_and_no_legacy_call(self):
        with patch('live_testing.moira_v2_native.decide', return_value=self.action) as current, \
                patch('live_testing.moira_native.decide', side_effect=AssertionError('No legacy fallback')) as old:
            self.assertEqual(self.runtime.decide(self.model, self.obs, replay=True), self.action)
        self.assertEqual(current.call_args.args[:3], (self.obs, self.model, self.directory))
        self.assertTrue(current.call_args.kwargs['replay']); old.assert_not_called()
    def test_v2_feedback_forwards_market_cutoff_and_never_uses_legacy_feedback(self):
        feedback = {'release_at': self.at.isoformat()}
        def rows(book, model, *args, **kwargs): return [feedback] if model==self.model else []
        with patch.object(runner, 'matured_feedback', side_effect=rows), \
                patch('live_testing.moira_v2_native.apply_feedback') as current, \
                patch('live_testing.moira_native.apply_feedback', side_effect=AssertionError('No legacy feedback')) as old:
            self.assertEqual(self.runtime.feedback(self.at), {})
        self.assertEqual(current.call_args.kwargs, {'source_cutoff': self.at})
        self.assertEqual(self.runtime.state['feedback_applied'][self.model], [self.at.isoformat()]); old.assert_not_called()
    def test_partial_or_wrong_marker_blocks_before_gateway_without_fallback(self):
        for change in ({'status': 'pending'}, {'model_id': 'moira2-top1'}, {'protocol': 'old-policy'},
                       {'checkpoint_sha256': '0'*64}, {'training_result_sha256': '0'*64},
                       {'training_result_path': 'relative/training-result.json'}):
            with self.subTest(change=change):
                self.write_ready(); marker=read_json(self.directory/'memory-policy.json'); marker.update(change)
                write_private(self.directory/'memory-policy.json', marker)
                with self.assertRaises(RuntimeError): self.runtime.decide(self.model, self.obs)
        self.runtime._gateway.scoped.assert_not_called()
    def test_v2_prepared_without_marker_and_orphan_v2_native_never_fall_back(self):
        (self.directory/'memory-policy.json').unlink()
        with self.assertRaises(RuntimeError): policy.ready(self.root, self.model)
        write_private(self.directory/'prepared.json', {'legacy': True})
        write_private(self.directory/'live-state.json', {'memory_policy_protocol': PROTOCOL_VERSION})
        with self.assertRaises(RuntimeError): policy.ready(self.root, self.model)
        write_private(self.directory/'live-state.json', {'memory_bank': {'schema': 'moira.counterfactual-memory/2'}})
        with self.assertRaises(RuntimeError): policy.ready(self.root, self.model)
        (self.directory/'live-state.json').unlink()
        write_private(self.directory/'prepared.json', {'schema': 'moira.quarter-ranking-checkpoint/2'})
        with self.assertRaises(RuntimeError): policy.ready(self.root, self.model)
    def test_ready_marker_requires_actual_sharpe_objective_and_correct_acceptance(self):
        for mutation in ('objective', 'count', 'max', 'prompt', 'accepted', 'model'):
            with self.subTest(mutation=mutation):
                self.set_report_valid()
                if mutation=='objective': self.report['objective']='cumulative_return'
                elif mutation=='count': self.report['completed_cycles']=0
                elif mutation=='max': self.report['max_cycles']=9
                elif mutation=='prompt': self.report['checkpoint']['frozen_prompt_sha256']='0'*64
                elif mutation=='accepted': self.report['cycles'][0]['accepted']=False
                elif mutation=='model': self.report['checkpoint']['model_id']='moira2-top1'
                self.refresh_report_hash()
                with self.assertRaises(RuntimeError): policy.ready(self.root, self.model)
    def set_report_valid(self):
        self.report = {'objective': 'sample_sharpe_without_annualization', 'completed_cycles': 1, 'max_cycles': 8,
            'checkpoint': copy.deepcopy(self.prepared), 'cycles': [{'accepted': True,
                'baseline_metrics': {'sharpe_ratio': .1}, 'candidate_metrics': {'sharpe_ratio': .2}}]}
        self.write_ready()
    def test_invalid_v2_feedback_stays_pending_without_teaching_any_framework(self):
        (self.directory/'memory-policy.json').unlink(); feedback={'release_at': self.at.isoformat()}
        def rows(book, model, *args, **kwargs): return [feedback] if model==self.model else []
        with patch.object(runner, 'matured_feedback', side_effect=rows), \
                patch('live_testing.moira_v2_native.apply_feedback') as current, \
                patch('live_testing.moira_native.apply_feedback') as old:
            self.assertEqual(self.runtime.feedback(self.at), {self.model: 'RuntimeError'})
        self.assertEqual(self.runtime.state['feedback_applied'][self.model], [])
        old.assert_not_called(); current.assert_not_called(); self.runtime._gateway.scoped.assert_not_called()
    def test_original_moira_and_real_legacy_variant_keep_original_route(self):
        with patch('live_testing.moira_native.decide', return_value=self.action) as old, \
                patch('live_testing.moira_v2_native.decide', side_effect=AssertionError('No v2 migration')) as current:
            self.assertEqual(self.runtime.decide('moira', self.obs), self.action)
        old.assert_called_once(); current.assert_not_called()
        legacy='moira2-top1'; directory=self.root/'models'/legacy
        write_private(directory/'prepared.json', {'old': 'actual legacy'})
        write_private(directory/'initial-training-checkpoint.json', {'old': 'frozen'})
        write_private(directory/'initial-training-ready.json', {'status':'ready','model_id':legacy,
            'backbone':'gpt-4.1-mini','cycles_completed':2,
            'checkpoint_sha256':file_hash(directory/'initial-training-checkpoint.json')})
        self.assertFalse(policy.ready(self.root, legacy))
        with patch('live_testing.moira_native.decide', return_value=self.action) as old:
            self.assertEqual(self.runtime.decide(legacy, self.obs), self.action)
        old.assert_called_once()
    def test_new_install_hold_blocks_before_config_book_key_or_kernel(self):
        operation=self.root/'repairs/ranking-v2/interrupted'
        write_private(operation/'status.json', {'status':'committing'})
        write_private(self.root/'repairs/ranking-v2/current.json', {'operation':str(operation)})
        with patch.object(runner, 'read_json', side_effect=AssertionError('No config/book reads')), \
                patch.object(runner, 'create_guarded_gateway') as gateway, \
                patch.object(runner, 'write_private') as write:
            with self.assertRaisesRegex(RuntimeError, 'three-model installation'): runner.Runtime(self.root)
        gateway.assert_not_called(); write.assert_not_called()


if __name__ == '__main__': unittest.main()
