import copy
from datetime import datetime,timedelta
import math
import statistics
import unittest

from live_testing.calendar import ExperimentCalendar,NY,month_bounds
from live_testing.accounting import (new_book,record_decision,record_valuation,settle_due,
    matured_feedback,risk_metrics,account_metrics,validate_book,record_usage,weekly_usage,monthly_usage,pair_return)

MODELS=["Moira","Codex","FinMem","Moira2-K1","Moira2-K3","Moira2-K5"]


def at(hour,minute=30,day="2026-11-02"):
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").replace(tzinfo=NY)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.calendar=ExperimentCalendar(start_at=at(9),last_release_at=at(10))
        self.book=new_book(MODELS,["AAA","BBB"])
        self.action={"stock1":"AAA","stock2":"BBB","action":"LS"}

    def decision(self,model="Moira",hour=9,action=None,**kwargs):
        return record_decision(self.book,model,at(hour),action or self.action,
            generated_at=kwargs.pop("generated_at",at(hour)+timedelta(seconds=15)),
            input_asof=kwargs.pop("input_asof",at(hour)),calendar=self.calendar,**kwargs)

    def quote(self,hour,prices,**kwargs):
        return record_valuation(self.book,at(hour),prices,
            observed_at=kwargs.pop("observed_at",at(hour)+timedelta(seconds=20)),calendar=self.calendar,**kwargs)

    def settle(self,when):
        return settle_due(self.book,when,calendar=self.calendar)

    def test_six_accounts_start_at_one_and_are_independent(self):
        self.assertEqual(len(self.book["accounts"]),6)
        self.assertTrue(all(a=={"equity":1.,"history":[]} for a in self.book["accounts"].values()))
        self.decision();self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":110,"BBB":100})
        result=self.settle(at(11)+timedelta(minutes=1))
        self.assertAlmostEqual(self.book["accounts"]["Moira"]["equity"],1.05)
        self.assertEqual(self.book["accounts"]["Codex"]["equity"],1)
        self.assertEqual(result["pending"]["Codex"]["reason"],"missing_decision")

    def test_next_node_entry_following_exit_not_release_price(self):
        self.decision();self.quote(9,{"AAA":1,"BBB":100});self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":110,"BBB":90})
        self.assertEqual(self.settle(at(10)+timedelta(minutes=1))["settlements"],[])
        self.assertEqual(self.book["accounts"]["Moira"]["history"],[])
        self.settle(at(11)+timedelta(minutes=1))
        self.assertAlmostEqual(self.book["accounts"]["Moira"]["history"][0]["step_return"],.1)

    def test_prices_are_immutable_and_identical_retry_keeps_original_availability(self):
        original=self.quote(10,{"AAA":100,"BBB":100})
        again=self.quote(10,{"AAA":100,"BBB":100},observed_at=at(10)+timedelta(minutes=3))
        self.assertEqual(original,again)
        before=copy.deepcopy(self.book)
        with self.assertRaises(ValueError):self.quote(10,{"AAA":101,"BBB":100})
        self.assertEqual(self.book,before)
        with self.assertRaises(ValueError):self.quote(11,{"AAA":None,"BBB":100})
        self.assertNotIn(at(11).isoformat(),self.book["valuations"])

    def test_missing_exit_stays_pending_instead_of_zero(self):
        self.decision();self.quote(10,{"AAA":100,"BBB":100})
        report=self.settle(at(12))
        self.assertEqual(report["pending"]["Moira"]["reason"],"missing_valuation")
        self.assertEqual(self.book["accounts"]["Moira"]["history"],[])

    def test_later_complete_round_cannot_jump_over_missing_first_action(self):
        self.decision(hour=10);self.quote(11,{"AAA":100,"BBB":100});self.quote(12,{"AAA":110,"BBB":100})
        result=self.settle(at(12)+timedelta(minutes=1))
        self.assertEqual(result["pending"]["Moira"]["reason"],"missing_decision")
        self.assertEqual(self.book["accounts"]["Moira"]["history"],[])

    def test_genuine_flat_is_distinct_from_missing_action(self):
        self.decision(action={**self.action,"action":"F"});self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":110,"BBB":50})
        self.settle(at(11)+timedelta(minutes=1))
        self.assertEqual(len(self.book["accounts"]["Moira"]["history"]),1)
        self.assertEqual(self.book["accounts"]["Moira"]["history"][0]["step_return"],0)
        self.assertEqual(len(self.book["accounts"]["Codex"]["history"]),0)

    def test_short_long_and_compounding(self):
        self.decision();self.decision(hour=10,action={**self.action,"action":"SL"})
        self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":120,"BBB":100});self.quote(12,{"AAA":108,"BBB":100})
        self.settle(at(12)+timedelta(minutes=1))
        self.assertAlmostEqual(self.book["accounts"]["Moira"]["equity"],1.1*1.05)
        metrics=account_metrics(self.book,"Moira",calendar=self.calendar)
        self.assertAlmostEqual(metrics["cumulative_return"],.155)

    def test_no_future_or_late_live_decision(self):
        for kwargs in [{"generated_at":at(9)-timedelta(seconds=1)},
                       {"generated_at":at(10)}, {"input_asof":at(10)}]:
            with self.assertRaises(ValueError):self.decision(**kwargs)
        self.assertEqual(self.book["decisions"]["Moira"],{})

    def test_explicit_replay_keeps_actual_generation_timestamp(self):
        late=at(12)
        with self.assertRaises(ValueError):self.decision(generated_at=late,mode="retrospective_point_in_time")
        record=self.decision(generated_at=late,mode="retrospective_point_in_time",
            audit={"authorization":"User approved first two historical point-in-time decisions","input_sha256":"a"*64})
        self.assertEqual(record["generated_at"],late.isoformat())
        self.assertEqual(record["release_at"],at(9).isoformat())
        self.assertNotEqual(record["generated_at"],record["release_at"])

    def test_decision_idempotence_and_replacement_rejected(self):
        first=self.decision();self.assertEqual(self.decision(),first)
        with self.assertRaises(ValueError):self.decision(action={**self.action,"action":"SL"})

    def test_feedback_waits_for_exit_and_observed_prices(self):
        self.decision();self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":110,"BBB":100})
        report=self.settle(at(11)+timedelta(seconds=10))
        self.assertEqual(report["pending"]["Moira"]["reason"],"valuation_or_decision_not_yet_available")
        self.assertEqual(matured_feedback(self.book,"Moira",at(11),calendar=self.calendar),[])
        settled_at=at(11)+timedelta(seconds=30);self.settle(settled_at)
        self.assertEqual(matured_feedback(self.book,"Moira",at(11),calendar=self.calendar),[])
        self.assertEqual(len(matured_feedback(self.book,"Moira",settled_at,observation_asof=at(11),calendar=self.calendar)),1)
        self.assertEqual(matured_feedback(self.book,"Moira",settled_at,observation_asof=at(10),calendar=self.calendar),[])

    def test_repeat_settlement_is_idempotent(self):
        self.decision();self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":110,"BBB":100});self.settle(at(11)+timedelta(minutes=1))
        before=copy.deepcopy(self.book)
        self.assertEqual(self.settle(at(11)+timedelta(minutes=2))["settlements"],[])
        self.assertEqual(before,self.book)

    def test_corrupted_history_cannot_pass_validation(self):
        self.decision();self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":110,"BBB":100});self.settle(at(11)+timedelta(minutes=1))
        self.book["accounts"]["Moira"]["history"][0]["step_return"]=0
        with self.assertRaises(ValueError):validate_book(self.book,calendar=self.calendar)

    def test_invalid_quotes_and_repeated_stock_rejected(self):
        for bad in [float("nan"),float("inf"),0,-1,True]:
            with self.assertRaises(ValueError):self.quote(10,{"AAA":bad,"BBB":100})
        with self.assertRaises(ValueError):self.decision(action={"stock1":"AAA","stock2":"AAA","action":"F"})

    def test_metrics_sample_sharpe_initial_peak_and_recovery(self):
        values=[-.1,.2,-.05]
        metrics=risk_metrics(values)
        self.assertAlmostEqual(metrics["cumulative_return"],.9*1.2*.95-1)
        self.assertAlmostEqual(metrics["sharpe_ratio"],statistics.mean(values)/statistics.stdev(values))
        self.assertAlmostEqual(metrics["max_drawdown"],.1)
        self.assertEqual(risk_metrics([]),{"completed_rounds":0,"cumulative_return":0.,"sharpe_ratio":None,"max_drawdown":0.})
        self.assertIsNone(risk_metrics([0.,0.])["sharpe_ratio"])
        self.assertIsNone(risk_metrics([.01,.01])["sharpe_ratio"])
        self.assertIsNone(risk_metrics([.01])["sharpe_ratio"])
        with self.assertRaises(ValueError):risk_metrics([-1])

    def test_usage_ny_iso_week_natural_month_and_duplicate_identity(self):
        record_usage(self.book,"Moira",call_id="sunday",at=at(23,59,day="2026-11-01"),cost_usd=.5,input_tokens=10)
        record_usage(self.book,"Moira",call_id="monday",at=at(0,0),cost_usd=.2)
        record_usage(self.book,None,call_id="shared",at=at(9),cost_usd=.1,purpose="news_summary")
        self.assertEqual(weekly_usage(self.book,at(9))["calls"],2)
        self.assertAlmostEqual(weekly_usage(self.book,at(9))["cost_usd"],.3)
        self.assertAlmostEqual(monthly_usage(self.book,at(9))["cost_usd"],.8)
        before=copy.deepcopy(self.book)
        with self.assertRaises(ValueError):record_usage(self.book,"Moira",call_id="monday",at=at(0,0),cost_usd=.3)
        self.assertEqual(before,self.book)

    def test_invalid_usage_cannot_lower_cost_totals(self):
        for value in [-1,True,float("nan"),float("inf")]:
            with self.assertRaises(ValueError):record_usage(self.book,"Moira",call_id="bad",at=at(9),cost_usd=value)
        self.assertEqual(self.book["usage"],{})
        record_usage(self.book,"Moira",call_id="ok",at=at(9),cost_usd=.1)
        self.book["usage"]["ok"]["cost_usd"]=-100
        with self.assertRaises(ValueError):monthly_usage(self.book,at(9))

    def test_usage_is_partitioned_at_new_york_month_boundary(self):
        record_usage(self.book,"Moira",call_id="oct",at="2026-11-01T03:59:00+00:00",cost_usd=.4)
        record_usage(self.book,"Moira",call_id="nov",at="2026-11-01T04:00:00+00:00",cost_usd=.2)
        self.assertAlmostEqual(monthly_usage(self.book,at(9))["cost_usd"],.2)

    def test_nonpositive_equity_errors_atomically(self):
        self.decision(action={**self.action,"action":"SL"});self.quote(10,{"AAA":100,"BBB":100});self.quote(11,{"AAA":400,"BBB":100})
        before=copy.deepcopy(self.book)
        with self.assertRaises(ValueError):self.settle(at(11)+timedelta(minutes=1))
        self.assertEqual(self.book,before)

    def test_audit_metadata_is_finite_and_does_not_alias_caller(self):
        audit={"source":{"label":"immutable"}}
        self.decision(audit=audit);audit["source"]["label"]="changed"
        self.assertEqual(self.book["decisions"]["Moira"][at(9).isoformat()]["audit"]["source"]["label"],"immutable")
        with self.assertRaises(ValueError):self.decision(hour=10,audit={"bad":float("nan")})

    def test_pair_return_cannot_mask_missing_action_or_prices_as_flat(self):
        for action in [None,{"action":"F"},{"stock1":"AAA","stock2":"AAA","action":"F"}]:
            with self.assertRaises(ValueError):pair_return(action,{"AAA":100,"BBB":100},{"AAA":110,"BBB":100})
        with self.assertRaises(ValueError):pair_return({**self.action,"action":"F"},{"AAA":100},{"AAA":100})

    def test_final_action_enters_monday_and_exits_final_1030(self):
        release=at(15,day="2027-05-21");cal=ExperimentCalendar(start_at=release,last_release_at=release)
        book=new_book(MODELS,["AAA","BBB"])
        record_decision(book,"Moira",release,self.action,generated_at=release+timedelta(seconds=1),input_asof=release,calendar=cal)
        entry=at(9,day="2027-05-24");exit=at(10,day="2027-05-24")
        record_valuation(book,entry,{"AAA":100,"BBB":100},observed_at=entry,calendar=cal)
        record_valuation(book,exit,{"AAA":104,"BBB":100},observed_at=exit,calendar=cal)
        self.assertEqual(settle_due(book,entry,calendar=cal)["settlements"],[])
        settled=settle_due(book,exit,calendar=cal)
        self.assertEqual(len(settled["settlements"]),1)
        self.assertAlmostEqual(book["accounts"]["Moira"]["equity"],1.02)


if __name__=="__main__":unittest.main()
