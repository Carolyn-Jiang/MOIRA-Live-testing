import copy
from datetime import datetime,timedelta
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from live_testing.accounting import new_book,record_decision,record_valuation,settle_due
from live_testing.calendar import DEFAULT_CALENDAR as C,NY
from live_testing.projection import build_feed,TOKEN_FIELDS,_view
from live_testing.registry import MODEL_IDS


def budget(calls=None):
    return {"schema":"moira.live-testing.budget/1","limit_usd":100.,"warning_usd":20.,"calls":calls or {}}


def call(model="finmem",stage="inference",**kwargs):
    return {"model_id":model,"stage":stage,"started_at":"2026-11-02T10:31:00-05:00",
            "month":"2026-11","status":"completed","input_tokens":100,"cached_input_tokens":20,
            "output_tokens":30,"total_tokens":130,"actual_cost_usd":.01,"guard_cost_usd":.01,**kwargs}


def populated(count=2,missing_second=False):
    book=new_book(MODEL_IDS,["AAA","BBB"])
    for i,at in enumerate(C.release_nodes[:count]):
        for model in MODEL_IDS:
            if missing_second and model==MODEL_IDS[-1] and i==1:continue
            record_decision(book,model,at,{"stock1":"AAA","stock2":"BBB","action":"LS"},
                generated_at=at+timedelta(seconds=5),input_asof=at,
                audit={"private_prompt":"DO NOT PUBLISH","source_response_id":"private-id"})
    for i,at in enumerate(C.valuation_nodes[:count+2]):
        record_valuation(book,at,{"AAA":100*1.01**i,"BBB":100},observed_at=at+timedelta(seconds=10))
    asof=C.valuation_nodes[count+1]+timedelta(seconds=20)
    settle_due(book,asof)
    return book,asof


class ProjectionTests(unittest.TestCase):
    def test_ranking_uses_sharpe_with_negative_values_before_undefined_and_stable_ties(self):
        returns = [[.2,-.1],[.01,.011],[0.,0.],[-.01,-.02],[0.,.01],[0.,.01]]
        histories = {model:[{"step_return":value, "entry_at":C.action_window(C.release_nodes[i]).entry_at.isoformat(),
                            "exit_at":C.action_window(C.release_nodes[i]).exit_at.isoformat()} for i,value in enumerate(values)]
            for model,values in zip(MODEL_IDS,returns)}
        original = copy.deepcopy(histories)
        view = _view(histories,range(2))
        self.assertEqual([row["model_id"] for row in view["rows"]],
            ["finmem","moira2-top3","moira2-top5","codex","moira2-top1","moira"])
        self.assertEqual([row["model_id"] for row in _view(histories,[0])["rows"]],list(MODEL_IDS))
        self.assertEqual(histories,original)

    def test_only_six_model_common_settled_prefix_is_public(self):
        book,asof=populated(missing_second=True)
        self.assertEqual([len(book["accounts"][m]["history"]) for m in MODEL_IDS],[2,2,2,2,2,1])
        original=copy.deepcopy(book);feed=build_feed(book,asof=asof)
        self.assertEqual(feed["all_time"]["completed_rounds"],1)
        self.assertEqual(len(feed["all_time"]["rows"]),6)
        self.assertTrue(all(len(s["points"])==2 for s in feed["all_time"]["series"]))
        self.assertEqual(sum(w["completed_rounds"] for w in feed["weeks"]),1)
        self.assertEqual(feed["status"],"pending")
        self.assertEqual(book,original)

    def test_no_fake_zero_returns_when_any_model_is_unsettled(self):
        book,asof=populated()
        book["accounts"][MODEL_IDS[-1]]={"equity":1.,"history":[]}
        feed=build_feed(book,asof=asof)
        self.assertEqual(feed["all_time"],{"completed_rounds":0,"rows":[],"series":[]})

    def test_weekly_return_uses_exit_week_and_rebases_to_one(self):
        book,asof=populated(count=30)
        feed=build_feed(book,asof=asof)
        self.assertEqual(feed["all_time"]["completed_rounds"],30)
        self.assertEqual([w["completed_rounds"] for w in feed["weeks"]],[28,2])
        self.assertEqual([w["week_id"] for w in feed["weeks"]],["2026-W45","2026-W46"])
        for week,n in zip(feed["weeks"],[28,2]):
            self.assertTrue(all(s["points"][0]=={"step":0,"equity":1.} and len(s["points"])==n+1 for s in week["series"]))
            self.assertAlmostEqual(week["rows"][0]["cumulative_return"],1.005**n-1)
            self.assertIsNone(week["rows"][0]["sharpe_ratio"])

    def test_dates_are_actual_entry_and_exit_boundaries_including_cross_week_baseline(self):
        book,asof=populated(count=30);original=copy.deepcopy(book)
        feed=build_feed(book,asof=asof)
        history=book["accounts"][MODEL_IDS[0]]["history"]
        self.assertEqual(feed["all_time"]["timestamps"],
            [history[0]["entry_at"],*(row["exit_at"] for row in history)])
        for week in feed["weeks"]:
            rows=[row for row in history if week["start"]<=row["exit_at"]<week["end"]]
            self.assertEqual(week["timestamps"],[rows[0]["entry_at"],*(row["exit_at"] for row in rows)])
        self.assertLess(feed["weeks"][1]["timestamps"][0],feed["weeks"][1]["start"])
        self.assertEqual(book,original)

    def test_dates_retain_holiday_half_day_and_dst_calendar_boundaries(self):
        for release in ("2026-11-25T14:30:00-05:00","2026-11-27T11:30:00-05:00", "2027-03-12T14:30:00-05:00"):
            window=C.action_window(datetime.fromisoformat(release))
            histories={model:[{"step_return":.01,"entry_at":window.entry_at.isoformat(),"exit_at":window.exit_at.isoformat()}] for model in MODEL_IDS}
            view=_view(histories,[0])
            self.assertEqual(view["timestamps"],[window.entry_at.isoformat(),window.exit_at.isoformat()])
        self.assertEqual(view["timestamps"],["2027-03-12T15:30:00-05:00","2027-03-15T09:30:00-04:00"])

    def test_future_availability_never_appears_in_earlier_snapshot(self):
        book,asof=populated()
        feed=build_feed(book,asof=asof-timedelta(minutes=1))
        self.assertEqual(feed["all_time"]["completed_rounds"],0)

    def test_measured_usage_and_unknown_codex_are_not_fabricated_zero(self):
        book,asof=populated()
        book["imported_usage"]=[{"source_call_id":"original-codex","model_id":"codex","stage":"inference",
            "started_at":"2026-11-02T09:49:00-05:00","status":"usage-unavailable"}]
        ledger=budget({"f":call(),"m":call("moira","training",input_tokens=200,cached_input_tokens=0,output_tokens=10,total_tokens=210),
                       "d":call("data",input_tokens=50,cached_input_tokens=0,output_tokens=5,total_tokens=55)})
        feed=build_feed(book,asof=asof,budget_ledger=ledger)
        week=feed["weeks"][0];tokens={r["model_id"]:r for r in week["tokens"]}
        self.assertEqual(len(tokens),6)
        self.assertIsNone(tokens["codex"]["total_tokens"])
        self.assertIsNone(tokens["codex"]["inference_tokens"])
        self.assertEqual(tokens["finmem"]["total_tokens"],130)
        self.assertEqual(tokens["finmem"]["cached_input_tokens"],20)
        self.assertEqual(tokens["moira"]["training_tokens"],210)
        self.assertEqual(week["shared_tokens"]["total_tokens"],55)
        self.assertEqual(week["tokens_coverage"],"partial")
        self.assertNotIn("data",tokens)

    def test_unavailable_journal_is_not_zero_but_empty_verified_journal_is(self):
        book=new_book(MODEL_IDS,["AAA","BBB"])
        unavailable=build_feed(book,asof=C.release_nodes[0])
        self.assertTrue(all(all(r[field] is None for field in TOKEN_FIELDS) for r in unavailable["weeks"][0]["tokens"]))
        verified=build_feed(book,asof=C.release_nodes[0],budget_ledger=budget())
        self.assertTrue(all(all(r[field]==0 for field in TOKEN_FIELDS) for r in verified["weeks"][0]["tokens"]))

    def test_actual_usage_week_and_future_requests(self):
        book=new_book(MODEL_IDS,["AAA","BBB"])
        ledger=budget({"past":call(started_at="2026-11-08T23:59:00-05:00"),
                       "new":call(started_at="2026-11-09T00:00:00-05:00"),
                       "future":call(started_at="2026-11-09T15:00:00-05:00")})
        feed=build_feed(book,asof="2026-11-09T09:30:00-05:00",budget_ledger=ledger)
        self.assertEqual([next(x for x in w["tokens"] if x["model_id"]=="finmem")["total_tokens"] for w in feed["weeks"]],[130,130])

    def test_pending_provider_usage_stays_null(self):
        book,asof=populated();ledger=budget({"one":call(status="reserved",actual_cost_usd=None,guard_cost_usd=.2)})
        feed=build_feed(book,asof=asof,budget_ledger=ledger)
        row=next(r for r in feed["weeks"][0]["tokens"] if r["model_id"]=="finmem")
        self.assertIsNone(row["input_tokens"]);self.assertIsNone(row["total_tokens"])
        self.assertEqual(feed["budget"]["reserved_usd"],.2)

    def test_budget_uses_ny_month_and_source_guard(self):
        book,asof=populated();ledger=budget({"warning":call(actual_cost_usd=20,guard_cost_usd=20.5)})
        feed=build_feed(book,asof=asof,budget_ledger=ledger)
        self.assertEqual(feed["budget"]["status"],"warning")
        self.assertEqual(feed["budget"]["reserved_usd"],.5)
        ledger["calls"]["warning"]["guard_cost_usd"]=100
        self.assertEqual(build_feed(book,asof=asof,budget_ledger=ledger)["budget"]["status"],"blocked")
        ledger["calls"]["warning"]["guard_cost_usd"]=20.5
        ledger["blocks"]={"2026-11":{"reason":"private reservation block reason"}}
        blocked=build_feed(book,asof=asof,budget_ledger=ledger)
        self.assertEqual(blocked["budget"]["status"],"blocked")
        self.assertNotIn("private reservation block reason",json.dumps(blocked))

    def test_manifest_counts_only_genuinely_published_data(self):
        book,asof=populated();records=[{"timestamp":C.release_nodes[0].isoformat(),"published_at":(C.release_nodes[0]+timedelta(minutes=1)).isoformat(),"coverage":"partial","private_prices":{"AAA":100}},
            {"timestamp":C.release_nodes[1].isoformat(),"published_at":(asof+timedelta(hours=1)).isoformat(),"coverage":"complete"}]
        feed=build_feed(book,asof=asof,release_manifests=records)
        self.assertEqual(feed["data_status"]["published_batches"],1)
        self.assertEqual(feed["data_status"]["coverage"],"partial")
        self.assertEqual(feed["data_status"]["expected_release_points"],830)
        self.assertNotIn("private_prices",json.dumps(feed))

    def test_public_whitelist_and_input_immutability(self):
        book,asof=populated();ledger=budget({"f":call(request_payload={"stock1":"SECRET","api_key":"private-api-key"})})
        original=copy.deepcopy((book,ledger));feed=build_feed(book,asof=asof,budget_ledger=ledger)
        self.assertEqual((book,ledger),original)
        text=json.dumps(feed)
        for forbidden in ["stock1","stock2","action","private_prompt","source_response_id","private-api-key","request_payload"]:
            self.assertNotIn(forbidden,text)
        self.assertTrue(all(set(row)=={"model_id","cumulative_return","sharpe_ratio","max_drawdown"} for row in feed["all_time"]["rows"]))

    @unittest.skipUnless(shutil.which("node"),"Node is needed only to verify the actual frontend contract")
    def test_real_node_frontend_accepts_generated_feed(self):
        book,asof=populated(count=30)
        feed=build_feed(book,asof=asof,budget_ledger=budget())
        code="import {validateState} from './docs/app.js';let s='';for await(const x of process.stdin)s+=x;validateState(JSON.parse(s));process.stdout.write('accepted');"
        done=subprocess.run(["node","--input-type=module","-e",code],cwd=Path(__file__).resolve().parents[1],input=json.dumps(feed),text=True,capture_output=True)
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertEqual(done.stdout,"accepted")


if __name__=="__main__":unittest.main()
