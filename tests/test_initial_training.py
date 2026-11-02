"""Explicit old source grid never weakens the new weekly/live calendar."""
import copy
from datetime import datetime,timedelta
import unittest

from live_testing.calendar import NY
from live_testing.initial_training import HistoricalDataset,phase_two_dataset
from live_testing.memory import Namespace
from live_testing.registry import TICKERS
from live_testing.training import Observation,Execution,WeeklyDataset,simulate_episode


def fixture():
    releases=[datetime(2026,10,d,h,30,tzinfo=NY) for d in range(19,24)
        for h in ((9,10,11,12,14) if d==23 else (9,10,11,12,14,15))]
    grid=releases+[datetime(2026,10,23,15,30,tzinfo=NY),datetime(2026,10,23,16,tzinfo=NY)]
    prices=[{t:100.+i*(.2 if j%2 else .1) for j,t in enumerate(TICKERS)} for i in range(len(grid))]
    rows=[{'round_id':at.strftime('%Y-%m-%d_%H:%M'),'timestamp':at.isoformat(),'prices':prices[i],
        'news':{t:['Synthetic historical headline.'] for t in TICKERS}} for i,at in enumerate(releases)]
    intervals={r['round_id']:{'entry_timestamp':grid[i+1].isoformat(),'exit_timestamp':grid[i+2].isoformat(),
        'entry_prices':prices[i+1],'exit_prices':prices[i+2]} for i,r in enumerate(rows)}
    return {'observations':rows,'intervals':intervals}


class HistoricalTests(unittest.TestCase):
    def test_exact_authorized_29_and_actual_terminal_marks_without_synthetic_action(self):
        data=phase_two_dataset(fixture(),training_at=datetime(2026,11,2,11,40,tzinfo=NY))
        self.assertEqual((len(data.observations),len(data.executions),data.pending_action_ids),(29,29,()))
        self.assertEqual(data.observations[-1].timestamp.hour,14)
        self.assertEqual(tuple(v.strftime('%H:%M') for v in data.valuation_nodes),('15:30','16:00'))
        self.assertFalse(any(row.timestamp.date().month==11 for row in data.observations))

    def test_delayed_feedback_not_seen_early_and_last_two_close_after_last_action(self):
        class Adapter:
            def begin_episode(self,checkpoint,namespace):
                outer=self
                class Policy:
                    def __init__(self):self.known=[];self.seen=[]
                    def decide(self,row):
                        self.seen.append((row.timestamp,tuple(self.known)))
                        return {'stock1':'NVDA','stock2':'GOOG','action':'LS'}
                    def on_feedback(self,value):self.known.append(value.execution.exit_at)
                    def checkpoint(self):return copy.deepcopy(checkpoint)
                outer.policy=Policy();return outer.policy
        data=phase_two_dataset(fixture(),training_at=datetime(2026,11,2,11,40,tzinfo=NY));adapter=Adapter()
        result=simulate_episode(adapter,{'prompt_memory':{'active_prompt':'frozen'}},data,Namespace('moira2-top3','training','historical'))
        self.assertEqual(len(result.decisions),29);self.assertEqual(len(result.finalized),29)
        self.assertEqual(len(adapter.policy.seen[0][1]),0);self.assertEqual(len(adapter.policy.seen[1][1]),0)
        self.assertEqual(len(adapter.policy.seen[2][1]),1)
        self.assertEqual(len(adapter.policy.seen[-1][1]),27)
        self.assertTrue(all(exit_at<=at for at,known in adapter.policy.seen for exit_at in known))

    def test_missing_future_or_november_observations_rejected(self):
        for mutation in ('missing','november','price'):
            bundle=fixture()
            if mutation=='missing':bundle['observations'].pop()
            if mutation=='november':bundle['observations'][0]['timestamp']='2026-11-02T09:30:00-05:00'
            if mutation=='price':
                label=next(iter(bundle['intervals'].values()));label['exit_prices']=copy.deepcopy(label['exit_prices']);label['exit_prices']['NVDA']+=9
            with self.assertRaises(ValueError):phase_two_dataset(bundle,training_at=datetime(2026,11,2,12,tzinfo=NY))

    def test_wrong_delayed_boundary_rejected(self):
        bundle=fixture();last=next(reversed(bundle['intervals'].values()));last['exit_timestamp']='2026-10-23T16:30:00-04:00'
        with self.assertRaises(ValueError):phase_two_dataset(bundle,training_at=datetime(2026,11,2,12,tzinfo=NY))

    def test_weekly_gate_stays_strict_and_historical_context_cannot_enter_new_week(self):
        data=phase_two_dataset(fixture(),training_at=datetime(2026,11,2,11,40,tzinfo=NY))
        with self.assertRaises(ValueError):WeeklyDataset.build(data.observations,data.executions.values(),training_at=data.training_at)
        with self.assertRaises(ValueError):WeeklyDataset.build(data.observations,data.executions.values(),training_at=datetime(2026,11,2,8,tzinfo=NY))


if __name__=='__main__':unittest.main()
