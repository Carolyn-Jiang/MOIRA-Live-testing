"""Real native PolicyAgent path with explicitly injected fake transport only."""
import copy
from dataclasses import fields
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from live_testing import moira_native as m
from live_testing.memory import MemoryUnit
from live_testing.storage import read_json, write_private


class Gateway:
    def __init__(self, root):
        self.budget = SimpleNamespace(root=Path(root)); self.calls = []; self.embeddings = []; self.bad = False
    def embed(self, text): self.embeddings.append(text); return [1., 0.]
    def native_text(self, *, system_prompt, payload, task_type):
        self.calls.append((task_type, copy.deepcopy(payload)))
        if 'counterfactual_summary' in payload:
            return 'Synthetic finalized outcome reflection.\nPrinciples:\nCheck whether relative news agrees with spread evidence before choosing a direction.'
        if task_type == 'pair_selection': return json.dumps({'selected_pair': payload['top5_candidates'][0]['pair'], 'pair_selection_reason': 'Synthetic evidence.'})
        if task_type == 'daily_direction': return 'malformed' if self.bad else json.dumps({'action': 'LONG_SHORT', 'direction_reason': 'Synthetic reason.'})
        if task_type == 'critical_news_verification': return json.dumps({'severity': 'LOW', 'affects_pair_relationship': False})
        return json.dumps({'news_break_level': 'HEALTHY', 'is_relationship_break_signal': False})


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        external = Path(__file__).resolve().parents[2]/'pair-trading-fix'
        if not os.environ.get('PAIR_TRADING_FIX_ROOT') and external.exists(): os.environ['PAIR_TRADING_FIX_ROOT'] = str(external)
        try: cls.n = m.native_types()
        except RuntimeError as error: raise unittest.SkipTest(str(error))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.gateway = Gateway(self.root)
        write_private(self.root/'company-context.json', {'generated_at': '2026-11-02T08:00:00-05:00',
            'as_of': '2026-11-01T12:00:00-05:00', 'companies': {t: {'recent_operations_summary': 'Synthetic prior source.'} for t in m.TICKERS}})
        prompt = 'Synthetic frozen native instruction.'
        self.prepared = {'model': m.MODEL, 'status': 'ready', 'pair_memory': [],
            'prompt_memory': {'active_prompt': prompt}, 'frozen_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        pairs = [('NVDA', 'GOOG'), ('META', 'JPM'), ('GOOG', 'JPM'), ('GOOG', 'WMT'), ('NVDA', 'WMT')]
        candidates = []
        for pair in pairs:
            values = {f.name: .1 for f in fields(self.n.StatArbPair) if f.default is f.default_factory}
            # Explicit complete dataclass defaults; no statistical/model call.
            values = dict(pair=pair, n_obs=2520, corr_10y=.9, corr_5y=.9, corr_1y=.9,
                rolling_corr_mean=.9, rolling_corr_std=.01, coint_pvalue=.01, coint_stat=-4.,
                adf_pvalue=.01, adf_stat=-4., hedge_ratio=1., alpha=0., spread_mean=0., spread_std=.1,
                current_z=.1, spread_vol=.1, z_p05=-2., z_p50=0., z_p95=2., half_life=3., half_life_ok=True,
                n_entry_1=10, n_entry_2=5, hit_rate=.5, hist_sharpe=.5, win_rate=.5,
                rolling_corr_stability=.9, spread_vol_stability=.9, breakdown_flag=False)
            candidates.append(self.n.StatArbPair(**values).to_dict())
        prices = {t: 100.+i for i, t in enumerate(m.TICKERS)}
        history = {(datetime(2026, 10, 1, 16, tzinfo=m.NY)+timedelta(days=i)).isoformat():
            {'price': {t: p+(i*.05 if j%2 else i*.1) for j, (t, p) in enumerate(prices.items())}}
            for i in range(20)}
        self.bundle = {'candidates': candidates, 'history': history, 'embedding_dimension': 2, 'embedding_id': 'synthetic-vector'}

    def observation(self, hour=9):
        at = datetime(2026, 11, 2, hour, 30, tzinfo=m.NY)
        return {'round_id': at.isoformat(), 'timestamp': at.isoformat(),
            'prices': {t: 103.+i for i, t in enumerate(m.TICKERS)},
            'news': {t: ['Synthetic current published headline.'] for t in m.TICKERS}}

    def initialize(self, model='moira', units=()):
        directory = self.root/model
        m.initialize(self.prepared, self.bundle, directory, units, self.gateway, model_id=model)
        return directory

    def test_real_native_select_direction_context_and_idempotent_current_input(self):
        directory = self.initialize(); obs = self.observation()
        first = m.decide(obs, 'moira', directory, self.bundle, self.gateway, asof=datetime(2026, 11, 2, 18, tzinfo=timezone.utc), replay=True)
        self.assertEqual(first['action'], 'LS'); self.assertEqual(len(self.gateway.calls), 2)
        self.assertTrue(all('company_operations_context' in payload for _, payload in self.gateway.calls))
        self.assertEqual(m.decide(obs, 'moira', directory, self.bundle, self.gateway), first)
        self.assertEqual(len(self.gateway.calls), 2)
        wrong = copy.deepcopy(obs); wrong['prices']['NVDA'] += 1
        with self.assertRaises(ValueError): m.decide(wrong, 'moira', directory, self.bundle, self.gateway)

    def test_break_pending_until_real1530_and_no_same_pair_reselection(self):
        directory = self.initialize(); asof = datetime(2026, 11, 2, 22, tzinfo=timezone.utc)
        m.decide(self.observation(9), 'moira', directory, self.bundle, self.gateway, asof=asof, replay=True)
        with patch.object(self.n.engine, 'statistical_lifecycle_state', return_value='BREAK'):
            second = m.decide(self.observation(10), 'moira', directory, self.bundle, self.gateway, asof=asof, replay=True)
            state = read_json(directory/'live-state.json'); self.assertTrue(state['pending_break'])
            self.assertEqual(sum(task=='pair_selection' for task, _ in self.gateway.calls), 1)
            third = m.decide(self.observation(15), 'moira', directory, self.bundle, self.gateway, asof=asof, replay=True)
        self.assertNotEqual({second['stock1'], second['stock2']}, {third['stock1'], third['stock2']})
        self.assertFalse(read_json(directory/'live-state.json')['pending_break'])

    def test_ranked_whole_memory_replaces_recent3_and_completed_feedback_persists(self):
        units = []
        for i in range(4):
            exited = datetime(2026, 10, 23, 12+i, tzinfo=m.NY)
            units.append(MemoryUnit(str(i), 'historical-'+str(i), 'NVDA', 'GOOG', 'LS', exited-timedelta(hours=2),
                exited-timedelta(hours=1), exited, exited, .01, {'LS': .01, 'SL': -.01, 'F': 0.},
                'One complete synthetic historical counterfactual reflection.', ('NVDA', 'GOOG'), {'source': 'prior historical episode'}))
        directory = self.initialize('moira2-top3', units); obs = self.observation()
        action = m.decide(obs, 'moira2-top3', directory, self.bundle, self.gateway, asof=datetime(2026, 11, 2, 15, tzinfo=timezone.utc), replay=True)
        payload = next(value for task, value in self.gateway.calls if task=='daily_direction')
        self.assertNotIn('past_3_days_direction_memory', payload)
        self.assertEqual(len(payload['ranked_counterfactual_reflections']), 3)
        self.assertTrue(all(set(v['counterfactual_returns'])=={'LS','SL','F'} for v in payload['ranked_counterfactual_reflections']))
        entry = {t: 100. for t in m.TICKERS}; exit_prices = {**entry, 'NVDA': 102.}
        record = {'release_at': obs['timestamp'], 'entry_at': '2026-11-02T10:30:00-05:00',
            'exit_at': '2026-11-02T11:30:00-05:00', 'settled_at': '2026-11-02T11:31:00-05:00',
            'entry_prices': entry, 'exit_prices': exit_prices,
            'action': {k: action[k] for k in ('stock1','stock2','action')}, 'step_return': .01}
        class Later(datetime):
            @classmethod
            def now(cls, tz=None): return datetime(2026,11,2,18,tzinfo=timezone.utc)
        with patch.object(m, 'datetime', Later): m.apply_feedback(record, directory, self.bundle, self.gateway, 'moira2-top3')
        state = read_json(directory/'live-state.json')
        self.assertEqual(len(state['memory_bank']['units']), 5)
        self.assertEqual(len(state['memory_bank']['feedback']), 1)

    def test_model_failure_is_not_flat_and_does_not_repeat_paid_attempt(self):
        directory = self.initialize(); self.gateway.bad = True; obs = self.observation()
        with self.assertRaises(RuntimeError): m.decide(obs, 'moira', directory, self.bundle, self.gateway)
        calls = len(self.gateway.calls)
        self.assertEqual(read_json(directory/'live-state.json')['decisions'], {})
        with self.assertRaises(RuntimeError): m.decide(obs, 'moira', directory, self.bundle, self.gateway)
        self.assertEqual(len(self.gateway.calls), calls)

    def test_real_native_weekly_training_uses_variable_matured_subset_and_no_live_context(self):
        from live_testing.training import Observation, Execution, WeeklyDataset
        times = [datetime(2026,11,2,h,30,tzinfo=m.NY) for h in (9,10,11,12)]
        prices = [{t:100. for t in m.TICKERS} for _ in times]
        prices[2]['NVDA'] = 120.; prices[3]['NVDA'] = 108.
        rows = [Observation(str(i), at, prices[i], {t:['Synthetic historical current news.'] for t in m.TICKERS}, at)
            for i, at in enumerate(times)]
        labels = [Execution(str(i), times[i+1], times[i+2], times[i+2], prices[i+1], prices[i+2],
            {'source':'shared immutable synthetic valuations'}) for i in (0,1)]
        data = WeeklyDataset.build(rows, labels, training_at=datetime(2026,11,9,8,tzinfo=m.NY))
        prepared = copy.deepcopy(self.prepared)
        prepared['prompt_memory'] = self.n.DirectionPromptMemory(
            base_prompt=prepared['prompt_memory']['active_prompt'], symbols=m.TICKERS).to_dict()
        live = self.initialize(); live_before = (live/'live-state.json').read_bytes()
        result = m.train_week(data, prepared, self.bundle, self.root/'weekly', self.gateway, 'moira', max_cycles=1)
        self.assertEqual((result['observation_count'],result['matured_label_count']), (4,2))
        self.assertEqual(result['pending_action_ids'], ['2','3'])
        self.assertEqual(result['completed_cycles'], 1)
        self.assertEqual(len(list((self.root/'weekly'/'episodes'/'moira').iterdir())), 2)
        reflections = [p for _,p in self.gateway.calls if 'counterfactual_summary' in p]
        self.assertEqual(len(reflections), 1); self.assertEqual(len(reflections[0]['decisions']), 2)
        self.assertFalse(any('company_operations_context' in p for _,p in self.gateway.calls))
        self.assertFalse(result['cycles'][0]['accepted'])
        self.assertEqual(result['checkpoint']['prompt_memory']['active_prompt'], self.prepared['prompt_memory']['active_prompt'])
        self.assertEqual((live/'live-state.json').read_bytes(), live_before)

    def test_native_superclass_chat_path_uses_explicit_budgeted_sdk_and_standard_tier(self):
        requests = []; closed = []
        def complete(**request):
            requests.append(copy.deepcopy(request))
            payload = json.loads(request['messages'][1]['content'])
            task = 'pair_selection' if 'top5_candidates' in payload else 'daily_direction'
            content = self.gateway.native_text(system_prompt=request['messages'][0]['content'], payload=payload, task_type=task)
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5,total_tokens=15),
                system_fingerprint='synthetic', choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)), close=lambda:closed.append(True))
        explicit = SimpleNamespace(key='synthetic-not-a-secret', budget=self.gateway.budget, embed=self.gateway.embed)
        directory = self.initialize()
        with patch('live_testing.llm.sdk_client', return_value=client) as factory:
            action = m.decide(self.observation(), 'moira', directory, self.bundle, explicit)
        self.assertEqual(action['action'], 'LS'); self.assertEqual(len(requests),2)
        self.assertEqual(factory.call_count,1); self.assertEqual(closed,[True])
        self.assertTrue(all(r['store'] is False and r['service_tier']=='default' and r['model']==m.MODEL for r in requests))

    def test_explicit_pre_live_historical_training_uses_native_ranked_memory_inside_episode(self):
        from live_testing.initial_training import HistoricalDataset
        from live_testing.training import Observation, Execution, train_native
        times=[datetime(2026,10,19,h,30,tzinfo=m.NY) for h in (9,10,11,12,14,15)]
        prices=[{t:100. for t in m.TICKERS} for _ in times];prices[2]['NVDA']=120.;prices[3]['NVDA']=108.
        observations=[Observation(str(i),at,prices[i],{t:['Synthetic old headline.'] for t in m.TICKERS},at) for i,at in enumerate(times[:4])]
        labels=[Execution(str(i),times[i+1],times[i+2],times[i+2],prices[i+1],prices[i+2],{'source':'synthetic exact old grid'}) for i in range(4)]
        dataset=HistoricalDataset.build(observations,labels,training_at=datetime(2026,11,2,12,tzinfo=m.NY),
            release_nodes=times[:4],valuation_nodes=times[4:],window_start=times[0],window_end=times[-1])
        prepared=copy.deepcopy(self.prepared);prepared['prompt_memory']=self.n.DirectionPromptMemory(
            base_prompt=self.prepared['prompt_memory']['active_prompt'],symbols=m.TICKERS).to_dict()
        adapter=m.NativeTrainingAdapter(dataset=dataset,bundle=self.bundle,directory=self.root/'initial',gateway=self.gateway,model_id='moira2-top1')
        result=train_native(adapter,dataset,prepared,version='moira2-top1',run_id='historical',max_cycles=1)
        self.assertEqual(result['matured_label_count'],4);self.assertEqual(len(result['pending_action_ids']),0)
        self.assertTrue(all('company_operations_context' not in payload for _,payload in self.gateway.calls))
        directions=[p for task,p in self.gateway.calls if task=='daily_direction']
        self.assertTrue(all('ranked_counterfactual_reflections' in p for p in directions))
        self.assertEqual([len(p['ranked_counterfactual_reflections']) for p in directions[:4]],[0,0,1,1])
        self.assertTrue(all(m.aware(u['exit_at'])<=m.aware(p['current_observation']['timestamp'])
            for p in directions for u in p['ranked_counterfactual_reflections']))
        state=read_json(adapter.last_policy.directory/'live-state.json')
        self.assertEqual(len(state['memory_bank']['units']),4)


if __name__ == '__main__': unittest.main()
