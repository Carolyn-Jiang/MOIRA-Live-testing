import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from live_testing.prompt_inputs import project_payload
from live_testing.registry import TICKERS
from live_testing.storage import read_json, write_private


class PromptInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.gateway = SimpleNamespace(budget=SimpleNamespace(root=self.root))
        self.identity = '00000000-0000-4000-a000-123456789abc'
        self.context = {'generated_at':'2026-11-01T12:00:00-05:00', 'as_of':'2026-11-01T11:00:00-05:00',
            'companies':{ticker:{'recent_operations_summary':ticker+' has a supplied operating update.',
                'summary_evidence_uuids':[self.identity], 'source_ids':[self.identity],
                'company_facts':[{'statement':'Reported operating facts.', 'evidence_uuids':[self.identity]}]}
                for ticker in TICKERS}}
        write_private(self.root/'company-context.json', self.context)
        self.observation = {'timestamp':'2026-11-02T09:30:00-05:00', 'round_id':'local-audit-id',
            'prices':{ticker:100. for ticker in TICKERS},
            'news':{ticker:[f'Published 2026-11-01T10:00:00-05:00 [{self.identity}]: Reported supplied revenue growth.'] for ticker in TICKERS}}

    def test_pair_only_wire_and_complete_private_sources_preserved(self):
        pair = list(TICKERS[:2]); payload = {'active_pair':pair, 'memory_policy':{'audit_clock':'private'},
            'ranked_counterfactual_reflections':[{'unit_id':'actual-unit-id', 'reflection':'Complete experiential lesson.'}]}
        before = copy.deepcopy((self.context, self.observation, payload))
        result = project_payload(payload, self.observation, gateway=self.gateway, directory=self.root/'model', task_type='daily_direction')
        self.assertEqual(set(result['company_operations_context']['companies']), set(pair))
        self.assertEqual(set(result['current_observation']['news']), set(pair))
        self.assertEqual(set(result['current_observation']['prices']), set(pair))
        self.assertNotIn(self.identity, json.dumps(result)); self.assertNotIn('memory_policy', result)
        self.assertEqual(result['ranked_counterfactual_reflections'], payload['ranked_counterfactual_reflections'])
        self.assertEqual((self.context, self.observation, payload), before)
        sources = list((self.root/'model/prompt-source-projections').glob('*.json'))
        self.assertEqual(len(sources), 1)
        local = read_json(sources[0]); self.assertEqual(local['original_observation'], self.observation)
        self.assertIn(self.identity, local['company_references'].values())
        archive = list((self.root/'prompt-source-archive').glob('*.json'))
        self.assertEqual(read_json(archive[0]), self.context)
        project_payload(payload, self.observation, gateway=self.gateway, directory=self.root/'model', task_type='daily_direction')
        self.assertEqual(len(list((self.root/'model/prompt-source-projections').glob('*.json'))),1)

    def test_selection_has_full_universe_but_round_reflection_has_no_market_news(self):
        view = project_payload({}, self.observation, gateway=self.gateway, directory=self.root/'model', task_type='pair_selection')
        self.assertEqual(set(view['company_operations_context']['companies']), set(TICKERS))
        payload = {'counterfactual_summary':{'graded':1}, 'decisions':[{'realized_pnl':.01}]}
        reflection = project_payload(payload, self.observation, gateway=self.gateway, directory=self.root/'model')
        self.assertEqual(reflection, payload)

    def test_missing_selected_pair_fails_before_direction_model(self):
        with self.assertRaises(ValueError):
            project_payload({}, self.observation, gateway=self.gateway, directory=self.root/'model', task_type='daily_direction')


if __name__ == '__main__': unittest.main()
