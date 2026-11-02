import unittest
from datetime import date, datetime, timedelta, timezone

from live_testing.calendar import (DEFAULT_CALENDAR as C, NY, LONDON, FULL_HOLIDAYS,
    HALF_DAYS, START_AT, FINAL_EXIT_AT, session_nodes, week_bounds, month_bounds)


def at(day, hour, minute=30):
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").replace(tzinfo=NY)


class CalendarTests(unittest.TestCase):
    def test_complete_grid_and_terminal_valuation_only(self):
        self.assertEqual(len(C.release_nodes), 830)
        self.assertEqual(len(C.valuation_nodes), 832)
        self.assertEqual(C.release_nodes[0], START_AT)
        last=C.action_window(C.release_nodes[-1])
        self.assertEqual(last.entry_at, at("2027-05-24",9))
        self.assertEqual(last.exit_at, FINAL_EXIT_AT)
        with self.assertRaises(ValueError): C.action_window(last.entry_at)
        self.assertEqual([e.kind for e in C.events() if e.at.date()==date(2027,5,24)], ["valuation","valuation"])

    def test_every_action_uses_next_two_actual_nodes(self):
        for i, release in enumerate(C.release_nodes):
            w=C.action_window(release)
            self.assertEqual((w.step,w.entry_at,w.exit_at),(i+1,C.valuation_nodes[i+1],C.valuation_nodes[i+2]))
            self.assertLess(w.release_at,w.entry_at)
            self.assertLess(w.entry_at,w.exit_at)

    def test_six_holidays_have_no_market_or_maintenance_training(self):
        self.assertEqual(len(FULL_HOLIDAYS),6)
        for day in FULL_HOLIDAYS:
            self.assertEqual(session_nodes(day),())
            self.assertFalse(any(e.kind in {"release","training","pair_reselection"} and e.at.date()==day for e in C.events()))

    def test_half_days_cancel_afternoon_and_never_invent_1300(self):
        for day in HALF_DAYS:
            self.assertEqual([d.strftime("%H:%M") for d in session_nodes(day)], ["09:30","10:30","11:30","12:30"])
        w=C.action_window(at("2026-11-27",12))
        self.assertEqual((w.entry_at,w.exit_at),(at("2026-11-30",9),at("2026-11-30",10)))
        self.assertTrue(any(e.kind=="pair_reselection" and e.at==at("2026-11-27",16,0) for e in C.events()))
        self.assertNotIn(at("2026-11-27",16,0),C.valuation_nodes)

    def test_overnight_weekend_holiday_and_midday_gap(self):
        cases=[("2026-11-02",12,"2026-11-02",14,"2026-11-02",15),
               ("2026-11-25",15,"2026-11-27",9,"2026-11-27",10),
               ("2026-12-24",12,"2026-12-28",9,"2026-12-28",10),
               ("2027-01-15",15,"2027-01-19",9,"2027-01-19",10),
               ("2027-03-25",15,"2027-03-29",9,"2027-03-29",10)]
        for d,h,ed,eh,xd,xh in cases:
            w=C.action_window(at(d,h));self.assertEqual((w.entry_at,w.exit_at),(at(ed,eh),at(xd,xh)))

    def test_first_training_is_november9_and_holiday_monday_skipped(self):
        training=[e.at for e in C.events() if e.kind=="training"]
        self.assertEqual(training[0],at("2026-11-09",8,0))
        self.assertNotIn(at("2027-01-18",8,0),training)
        self.assertFalse(any(e.at> C.last_release_at and e.kind in {"training","pair_reselection","cache"} for e in C.events()))

    def test_cache_may_run_on_sunday_and_exception_saturday(self):
        cache={e.at for e in C.events() if e.kind=="cache"}
        self.assertIn(at("2026-12-26",9),cache)
        self.assertIn(at("2026-12-27",9),cache)
        self.assertFalse(any(e.at in C.valuation_nodes for e in C.events() if e.kind=="cache"))

    def test_us_and_uk_dst_change_on_different_dates(self):
        cases=[("2027-03-12",14,14),("2027-03-15",13,13),("2027-03-29",13,14)]
        for day,utc_hour,uk_hour in cases:
            value=at(day,9)
            self.assertEqual(value.astimezone(timezone.utc).hour,utc_hour)
            self.assertEqual(value.astimezone(LONDON).hour,uk_hour)
        self.assertEqual(len(C.utc_segments()),3)

    def test_due_resolution_never_dispatches_early_or_replays_stale(self):
        before=START_AT-timedelta(seconds=1)
        self.assertEqual(C.due_events(before),())
        due=C.due_events(START_AT+timedelta(seconds=3))
        self.assertEqual(len(due),1)
        self.assertEqual(due[0].at,START_AT)
        self.assertEqual(C.due_events(START_AT+timedelta(minutes=26)),())
        self.assertEqual(C.due_events(START_AT,completed_ids=[due[0].id]),())
        self.assertEqual(C.next_node(START_AT),at("2026-11-02",10))
        self.assertEqual(C.next_node(START_AT,inclusive=True),START_AT)

    def test_ny_week_month_and_dst_elapsed_intervals(self):
        begin,end=week_bounds(at("2027-03-14",9))
        self.assertEqual((end.timestamp()-begin.timestamp())/3600,167)
        self.assertEqual(begin.weekday(),0)
        self.assertEqual(month_bounds(at("2026-12-31",15))[1],at("2027-01-01",0,0))
        self.assertEqual(month_bounds(at("2027-03-15",9))[0],at("2027-03-01",0,0))


if __name__=="__main__":unittest.main()
