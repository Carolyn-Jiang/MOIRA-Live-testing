"""Final selector wire preserves validated finalized evidence, without API calls."""
import copy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from live_testing import finmem_native as native
from live_testing.accounting import risk_metrics


class WeeklySelectionWireTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.budget=SimpleNamespace(root=Path(self.temp.name))
        self.bundle={'candidates':[{'pair':['AAA', 'BBB']} for _ in range(5)]}
        self.observation={'timestamp':'2026-11-06T15:30:00-05:00'}
        self.context={'schema':'moira.finmem.weekly-selection-context/1',
            'as_of':'2026-11-06T16:00:00-05:00','week_start':'2026-11-02T00:00:00-05:00',
            'observation_at':self.observation['timestamp'],'known_labels':0,
            'weekly_return':risk_metrics([]),'finalized_memory':[]}
        self.validator=Mock(side_effect=lambda value:copy.deepcopy(value))
        module=ModuleType('live_testing.finmem_weekly_context');module.validate_context=self.validator
        self.module=patch.dict(sys.modules,{'live_testing.finmem_weekly_context':module})
        self.module.start();self.addCleanup(self.module.stop)
        self.create=Mock(return_value=SimpleNamespace(output_text=json.dumps({
            'candidate_index':0,'reason':'Synthetic actual provider result.'}),id='synthetic-receipt'))
        self.sdk=Mock(return_value=SimpleNamespace(responses=SimpleNamespace(create=self.create)))

    @staticmethod
    def projection(source, observation, **kwargs):
        # Model the production observation whitelist: unknown observation extras
        # disappear, while explicit original-payload fields survive.
        return {**copy.deepcopy(source),'current_observation':{'timestamp':observation['timestamp']}}

    def run_selection(self, observation):
        with patch.object(native,'sdk_client',self.sdk),patch.object(native,'project_payload',side_effect=self.projection) as projector:
            result=native.select_pair(self.bundle,observation,'synthetic-not-a-key',self.budget)
        return result,projector

    def test_weekly_context_survives_the_final_serialized_responses_input(self):
        observation={**self.observation,'weekly_reselection_context':self.context}
        before=copy.deepcopy((observation,self.bundle))
        (_,metadata),projector=self.run_selection(observation)
        wire=self.create.call_args.kwargs;payload=json.loads(wire['input'])
        self.assertEqual(payload['weekly_reselection_context'],self.context)
        self.assertEqual(projector.call_args.args[0]['weekly_reselection_context'],self.context)
        self.assertEqual(metadata['response_id'],'synthetic-receipt')
        self.assertEqual((observation,self.bundle),before)
        self.validator.assert_called_once_with(self.context)
        self.assertIn('future epoch decisions',wire['instructions'])
        self.assertIn('Pending actions have no realized reward',wire['instructions'])
        self.assertNotIn('initial fixed relative-value pair',wire['instructions'])
        self.assertEqual(self.sdk.call_args.args[2],'finmem-live-weekly-reselection')
        self.assertEqual(wire['max_output_tokens'],500)

    def test_initial_selection_keeps_its_existing_prompt_and_has_no_weekly_evidence(self):
        self.run_selection(copy.deepcopy(self.observation))
        wire=self.create.call_args.kwargs;payload=json.loads(wire['input'])
        self.assertNotIn('weekly_reselection_context',payload)
        self.assertIn('Select ONE initial fixed relative-value pair',wire['instructions'])
        self.assertIn('fixed for this entire live phase',wire['instructions'])
        self.assertEqual(self.sdk.call_args.args[2],'finmem-live-initial-selection')
        self.validator.assert_not_called()

    def test_old_implicit_weekly_fields_fail_before_sdk_or_projection(self):
        for field in ('weekly_return','known_labels','finalized_memory'):
            with patch.object(native,'sdk_client',self.sdk),patch.object(native,'project_payload') as projector:
                with self.assertRaises(ValueError):
                    native.select_pair(self.bundle,{**self.observation,field:[]},'synthetic',self.budget)
            projector.assert_not_called()
        self.sdk.assert_not_called()

    def test_invalid_context_aborts_before_any_paid_sdk_client_is_constructed(self):
        self.validator.side_effect=ValueError('Synthetic malformed or future weekly evidence')
        with patch.object(native,'sdk_client',self.sdk),patch.object(native,'project_payload') as projector:
            with self.assertRaises(ValueError):
                native.select_pair(self.bundle,{**self.observation,'weekly_reselection_context':None},
                    'synthetic',self.budget)
        self.sdk.assert_not_called();projector.assert_not_called();self.create.assert_not_called()

    def test_dropped_or_changed_context_fails_closed_before_sdk(self):
        for returned in ({'candidates':[]},{'weekly_reselection_context':{'known_labels':0}}):
            with patch.object(native,'sdk_client',self.sdk),patch.object(native,'project_payload',return_value=returned):
                with self.assertRaises(RuntimeError):
                    native.select_pair(self.bundle,{**self.observation,'weekly_reselection_context':self.context},
                        'synthetic',self.budget)
        self.sdk.assert_not_called();self.create.assert_not_called()


if __name__=='__main__':unittest.main()
