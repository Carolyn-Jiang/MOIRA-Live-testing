"""Offline previous-week/matured-label and strict Sharpe objective checks."""
import copy
from datetime import datetime, timedelta
import unittest

from live_testing.calendar import NY, DEFAULT_CALENDAR, session_nodes
from live_testing.training import Observation, Execution, WeeklyDataset, objective, simulate_episode, train_native
from live_testing.memory import Namespace


def checkpoint(text='base'):
    return {'prompt_memory': {'active_prompt': text}, 'pair_memory': []}


def dataset(constant_baseline=False):
    times = [datetime(2026, 11, 2, h, 30, tzinfo=NY) for h in (9, 10, 11, 12)]
    price_rows = [dict(NVDA=100., GOOG=100., META=100., MSFT=100.),
        dict(NVDA=100., GOOG=100., META=100., MSFT=100.),
        dict(NVDA=120., GOOG=100., META=102., MSFT=100.),
        dict(NVDA=144. if constant_baseline else 108., GOOG=100., META=104.244, MSFT=100.)]
    rows = [Observation(str(i), at, price_rows[i], {t: ['Synthetic current news.'] for t in price_rows[i]}, at)
            for i, at in enumerate(times)]
    outcomes = [Execution(str(i), times[i+1], times[i+2], times[i+2], price_rows[i+1], price_rows[i+2], {'source': 'shared prices'}) for i in (0, 1)]
    return WeeklyDataset.build(rows, outcomes, training_at=datetime(2026, 11, 9, 8, tzinfo=NY))


class Adapter:
    def __init__(self): self.policies = []; self.reflections = []; self.rejections = []
    def begin_episode(self, source, namespace):
        outer = self
        class Policy:
            def __init__(self): self.current = []; self.credits = []; self.seen = []
            def decide(self, observation):
                self.seen.append((observation.action_id, tuple(self.current)))
                a, b = ('META', 'MSFT') if source['prompt_memory']['active_prompt'] == 'candidate' else ('NVDA', 'GOOG')
                return {'stock1': a, 'stock2': b, 'action': 'LS', 'reason': 'Synthetic explicit policy.'}
            def on_feedback(self, feedback): self.current.append(feedback.action_id); self.credits.append(feedback)
            def checkpoint(self): return copy.deepcopy(source)
        policy = Policy(); policy.namespace = namespace; outer.policies.append(policy); return policy
    def reflect(self, finalized, source): self.reflections.append(finalized); return {'lessons': [{'lesson_text': 'Synthetic evidence.'}]}
    def propose(self, source, reflection, cycle, baseline_metrics): return checkpoint('candidate')
    def accept(self, candidate, reference, reflection, cycle, before, after): return copy.deepcopy(candidate)
    def reject(self, candidate, reference, reflection, cycle, before, after, reason):
        self.rejections.append(reason); return copy.deepcopy(reference)


class TrainingTests(unittest.TestCase):
    def test_week_boundary_and_pending_labels_never_become_zero_rewards(self):
        data = dataset(); self.assertEqual(len(data.observations), 4); self.assertEqual(len(data.executions), 2)
        self.assertEqual(data.pending_action_ids, ('2', '3'))
        with self.assertRaises(ValueError): WeeklyDataset.build(data.observations, [], training_at=datetime(2026, 11, 9, 9, tzinfo=NY))
        # Future tail labels are discarded as objects, not supplied to models.
        row = data.observations[-1]; window = DEFAULT_CALENDAR.action_window(row.timestamp)
        future = Execution(row.action_id, window.entry_at, datetime(2026,11,9,9,30,tzinfo=NY), datetime(2026,11,9,9,30,tzinfo=NY), row.prices, row.prices, {'source':'future'})
        rebuilt = WeeklyDataset.build(data.observations, list(data.executions.values())+[future], training_at=data.training_at)
        self.assertNotIn(row.action_id, rebuilt.executions)

    def test_feedback_only_at_exit_and_reflection_receives_matured_subset(self):
        adapter = Adapter(); result = simulate_episode(adapter, checkpoint(), dataset(), Namespace('moira', 'training', 'episode'))
        self.assertEqual(adapter.policies[0].seen, [('0', ()), ('1', ()), ('2', ('0',)), ('3', ('0', '1'))])
        self.assertEqual(len(result.decisions), 4); self.assertEqual(len(result.finalized), 2)
        self.assertEqual(result.pending_action_ids, ('2','3'))
        self.assertAlmostEqual(sum(f.actual_return for f in result.finalized), .05)

    def test_strict_sharpe_accepts_lower_cr_higher_sharpe(self):
        adapter = Adapter(); report = train_native(adapter, dataset(), checkpoint(), version='moira', run_id='week', max_cycles=1)
        cycle = report['cycles'][0]
        self.assertLess(cycle['candidate_metrics']['cumulative_return'], cycle['baseline_metrics']['cumulative_return'])
        self.assertGreater(cycle['candidate_metrics']['sharpe_ratio'], cycle['baseline_metrics']['sharpe_ratio'])
        self.assertTrue(cycle['accepted']); self.assertEqual(report['checkpoint']['prompt_memory']['active_prompt'], 'candidate')
        self.assertEqual(len(adapter.reflections[0]), 2)
        self.assertTrue(all(p.namespace.scope=='training' for p in adapter.policies))
        self.assertEqual(len({p.namespace for p in adapter.policies}), len(adapter.policies))

    def test_undefined_variance_rejects_without_fake_zero_and_rolls_back(self):
        self.assertIsNone(objective([0., 0.])['sharpe_ratio'])
        self.assertIsNone(objective([.1, .1])['sharpe_ratio'])
        adapter = Adapter(); report = train_native(adapter, dataset(constant_baseline=True), checkpoint(), version='moira', run_id='undefined', max_cycles=2)
        self.assertEqual(report['accepted_updates'], 0)
        self.assertEqual(report['checkpoint']['prompt_memory']['active_prompt'], 'base')
        self.assertTrue(all(c['reason']=='undefined_sharpe_comparison' and c['baseline_metrics']['sharpe_ratio'] is None for c in report['cycles']))
        self.assertEqual(len(adapter.rejections), 2)

    def test_holiday_halfday_variable_count_has_no_invented1530(self):
        times = [at for day in range(23, 28) for at in session_nodes(datetime(2026,11,day).date())]
        self.assertEqual(len(times), 22)
        prices = dict(NVDA=100., GOOG=100.)
        rows = [Observation(str(i), at, prices, {t:['Synthetic news.'] for t in prices}, at) for i, at in enumerate(times)]
        labels = [Execution(str(i), times[i+1], times[i+2], times[i+2], prices, prices, {'source':'synthetic'}) for i in range(len(times)-2)]
        data = WeeklyDataset.build(rows, labels, training_at=datetime(2026,11,30,8,tzinfo=NY))
        self.assertEqual((len(data.observations), len(data.executions), len(data.pending_action_ids)), (22,20,2))
        self.assertFalse(any(at.day==27 and at.hour==15 for at in times))

    def test_wrong_shared_price_or_execution_grid_rejected(self):
        data = dataset(); item = data.executions['0']
        wrong = Execution(item.action_id, item.entry_at, item.exit_at, item.available_at,
            {**item.entry_prices,'NVDA':999.}, item.exit_prices, item.provenance)
        with self.assertRaises(ValueError): WeeklyDataset.build(data.observations, [wrong], training_at=data.training_at)


if __name__ == '__main__': unittest.main()
