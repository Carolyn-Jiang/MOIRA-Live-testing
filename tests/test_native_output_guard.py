import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from live_testing import llm
from live_testing.budget import Budget
from live_testing.native_output_guard import create_guarded_gateway


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.gateway=create_guarded_gateway('synthetic-not-a-key',Budget(self.root),self.root/'audit')

    def test_compatible_constructor_and_scope_preserve_guard_and_shared_budget(self):
        child=self.gateway.scoped('moira2-top5','training','weekly-source')
        self.assertEqual(type(child),type(self.gateway));self.assertTrue(callable(child.native_text))
        self.assertEqual(child.budget.root,self.gateway.budget.root)
        self.assertEqual(child.budget.model_id,'moira2-top5');self.assertEqual(child.budget.stage,'training')

    def test_no_cached_response_and_schema_controls_actual_model_choice(self):
        payload={'top5_candidates':[{'pair':['NVDA','GOOG']},{'pair':['META','JPM']}]}
        selected={'selected_pair':'META/JPM','pair_selection_reason':'Synthetic actual result.'}
        with patch.object(llm.Gateway,'json',return_value=selected) as call:
            result=json.loads(self.gateway.native_text(system_prompt='native original',payload=payload,task_type='pair_selection'))
        self.assertEqual(result,selected);self.assertEqual(call.call_count,1)
        self.assertEqual(call.call_args.kwargs['schema']['properties']['selected_pair']['enum'],['NVDA/GOOG','META/JPM'])
        self.assertEqual(call.call_args.kwargs['payload'],payload)

    def test_daily_allowed_actions_and_native_pair_are_not_substituted(self):
        payload={'active_pair':['NVDA','GOOG'],'current_observation':{'timestamp':'2026-11-02T09:30:00-05:00'}}
        actual={'action':'SHORT_LONG','direction_reason':'Synthetic actual result.'}
        with patch.object(llm.Gateway,'json',return_value=actual) as call:
            result=json.loads(self.gateway.native_text(system_prompt='native original',payload=payload,task_type='daily_direction'))
        self.assertEqual(result,actual)
        self.assertEqual(call.call_args.kwargs['schema']['properties']['action']['enum'],['LONG_SHORT','SHORT_LONG','FLAT'])
        self.assertEqual(call.call_args.kwargs['payload']['active_pair'],payload['active_pair'])


if __name__=='__main__':unittest.main()
