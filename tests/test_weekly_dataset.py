"""Pure holiday-fallback, unchanged source-week and delayed-label checks."""
import copy
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
import unittest

from live_testing.calendar import DEFAULT_CALENDAR, NY, session_nodes, week_bounds
from live_testing.training import Observation, Execution, WeeklyDataset as FrozenWeeklyDataset, simulate_episode
from live_testing.weekly_dataset import WeeklyDataset
from live_testing.memory import Namespace


def at(day,hour=8,minute=0):
    return datetime.combine(date.fromisoformat(day),time(hour,minute),NY)


def fixture(monday):
    first=date.fromisoformat(monday)
    times=[node for offset in range(5) for node in session_nodes(first+timedelta(days=offset))]
    def prices(stamp):
        index=DEFAULT_CALENDAR.valuation_nodes.index(stamp)
        return {'NVDA':100.+index/10,'GOOG':100.}
    observations=[Observation(str(i),stamp,prices(stamp),{'NVDA':['One historical item.'],'GOOG':[]},stamp)
                  for i,stamp in enumerate(times)]
    executions=[]
    for row in observations:
        window=DEFAULT_CALENDAR.action_window(row.timestamp)
        executions.append(Execution(row.action_id,window.entry_at,window.exit_at,window.exit_at,
            prices(window.entry_at),prices(window.exit_at),{'source':'synthetic exact immutable valuations'}))
    return observations,executions


class WeeklyDatasetTests(unittest.TestCase):
    def test_normal_monday_is_compatible_with_frozen_native_dataset(self):
        rows,labels=fixture('2026-11-02');cutoff=at('2026-11-09')
        actual=WeeklyDataset.build(rows,labels,training_at=cutoff)
        original=FrozenWeeklyDataset.build(rows,labels,training_at=cutoff)
        self.assertIsInstance(actual,FrozenWeeklyDataset)
        self.assertEqual(vars(actual),vars(original))
        self.assertEqual((len(actual.observations),len(actual.executions),len(actual.pending_action_ids)),(30,28,2))

    def test_holiday_tuesday_uses_previous_mon_sun_without_monday_leakage(self):
        for prior,closed,tuesday in [('2027-01-11','2027-01-18','2027-01-19'),('2027-02-08','2027-02-15','2027-02-16')]:
            rows,labels=fixture(prior)
            # Even deliberately supplied current-week Monday material is excluded;
            # there is no actual market node on that closed date.
            outside=Observation('current-Monday',at(closed,9,30),{'NVDA':999.,'GOOG':100.},
                {'NVDA':['Must not enter the preceding week.'],'GOOG':[]},at(closed,9,30))
            dataset=WeeklyDataset.build(rows+[outside],labels,training_at=at(tuesday))
            self.assertEqual(dataset.training_at,at(tuesday))
            self.assertEqual(len(dataset.observations),30)
            self.assertNotIn(outside.action_id,{r.action_id for r in dataset.observations})
            self.assertEqual(dataset.pending_action_ids,('28','29'))
            self.assertEqual(len(dataset.executions),28)
            self.assertTrue(all(r.timestamp<at(closed,0) for r in dataset.observations))

    def test_actual_tuesday_cutoff_accepts_monday_late_availability(self):
        rows,labels=fixture('2027-01-11');monday_late=at('2027-01-18',17)
        rows[0]=replace(rows[0],available_at=monday_late)
        labels[0]=replace(labels[0],available_at=monday_late)
        dataset=WeeklyDataset.build(rows,labels,training_at=at('2027-01-19'))
        self.assertEqual(dataset.observations[0].available_at,monday_late)
        self.assertEqual(dataset.executions['0'].available_at,monday_late)
        self.assertGreater(monday_late,at('2027-01-18'))

    def test_wrong_day_time_or_holiday_monday_cannot_train(self):
        rows,labels=fixture('2027-01-11')
        for cutoff in [at('2027-01-18'),at('2027-01-19',9),at('2027-01-19',8,1),
                       at('2027-01-20'),at('2027-01-19')+timedelta(microseconds=1)]:
            with self.subTest(cutoff=cutoff),self.assertRaises(ValueError):
                WeeklyDataset.build(rows,labels,training_at=cutoff)
        normal_rows,normal_labels=fixture('2026-11-02')
        with self.assertRaises(ValueError):
            WeeklyDataset.build(normal_rows,normal_labels,training_at=at('2026-11-10'))
        # Input in UTC names the same real Tuesday 08:00 boundary.
        self.assertEqual(WeeklyDataset.build(rows,labels,training_at=at('2027-01-19').astimezone(timezone.utc)).training_at,
                         at('2027-01-19'))

    def test_late_observation_rejected_future_labels_remain_absent(self):
        rows,labels=fixture('2027-01-11');cutoff=at('2027-01-19')
        late=replace(rows[0],available_at=cutoff+timedelta(seconds=1))
        with self.assertRaises(ValueError):WeeklyDataset.build([late]+rows[1:],labels,training_at=cutoff)
        late_label=replace(labels[0],available_at=cutoff+timedelta(seconds=1))
        data=WeeklyDataset.build(rows,[late_label]+labels[1:],training_at=cutoff)
        self.assertNotIn('0',data.executions)
        self.assertEqual(data.pending_action_ids,('0','28','29'))
        self.assertEqual(len(data.executions),27)

    def test_full_canonical_grid_prices_provenance_and_duplicates_checked(self):
        rows,labels=fixture('2027-01-11');cutoff=at('2027-01-19');first=labels[0]
        invalid=[replace(first,entry_at=rows[0].timestamp),
                 replace(first,exit_at=first.exit_at+timedelta(hours=1),available_at=first.exit_at+timedelta(hours=1)),
                 replace(first,entry_prices={**first.entry_prices,'NVDA':999.}),
                 replace(first,entry_prices={**first.entry_prices,'NVDA':float('nan')}),
                 replace(first,provenance={})]
        for item in invalid:
            with self.subTest(item=item.action_id),self.assertRaises(ValueError):
                WeeklyDataset.build(rows,[item]+labels[1:],training_at=cutoff)
        with self.assertRaises(ValueError):WeeklyDataset.build(rows,labels+[first],training_at=cutoff)
        with self.assertRaises(ValueError):WeeklyDataset.build(rows+[rows[-1]],labels,training_at=cutoff)
        with self.assertRaises(ValueError):WeeklyDataset.build(list(reversed(rows)),labels,training_at=cutoff)

    def test_native_simulation_keeps_unexited_tail_pending_and_inputs_immutable(self):
        rows,labels=fixture('2027-01-11');before=copy.deepcopy((rows,labels));cutoff=at('2027-01-19')
        dataset=WeeklyDataset.build(rows,labels,training_at=cutoff);seen=[];credits=[]
        class Adapter:
            def begin_episode(self,checkpoint,namespace):
                class Policy:
                    def decide(self,observation):
                        self.at=observation.timestamp
                        seen.append((observation.action_id,tuple(credits)))
                        return {'stock1':'NVDA','stock2':'GOOG','action':'LS'}
                    def on_feedback(self,feedback):
                        credits.append(feedback.action_id)
                    def checkpoint(self):return copy.deepcopy(checkpoint)
                return Policy()
        result=simulate_episode(Adapter(),{'prompt_memory':{'active_prompt':'frozen'}},dataset,Namespace('moira2-top1','training','offline'))
        self.assertEqual(seen[0],('0',()))
        self.assertEqual(seen[1],('1',()))
        self.assertEqual(seen[2],('2',('0',)))
        self.assertEqual(result.pending_action_ids,('28','29'))
        self.assertEqual(len(result.finalized),28)
        self.assertEqual((rows,labels),before)

    def test_dst_week_still_has_previous_calendar_week_window(self):
        rows,labels=fixture('2027-03-08');cutoff=at('2027-03-15')
        dataset=WeeklyDataset.build(rows,labels,training_at=cutoff)
        begin,end=week_bounds(rows[0].timestamp)
        self.assertEqual((end.timestamp()-begin.timestamp())/3600,167)
        self.assertEqual(cutoff.astimezone(timezone.utc).hour,12)
        self.assertEqual(len(dataset.observations),30)
        self.assertTrue(all(begin<=r.timestamp<end for r in dataset.observations))


if __name__=='__main__':unittest.main()
