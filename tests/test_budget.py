import json
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from live_testing.budget import Budget,BudgetExceeded,PricingUnavailable,model_rate,cost,RATES,PRICING_SOURCE

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.at=datetime(2026,11,2,16,tzinfo=timezone.utc)
        self.b=Budget(self.root,model_id='moira',clock=lambda:self.at)
    def tearDown(self): self.temp.cleanup()
    def payload(self): return {'model':'gpt-4.1-mini','input':'hello','max_output_tokens':100}
    def test_cached_tokens_are_subset_not_added_twice(self):
        key=self.b.reserve(self.payload(),'inference')
        self.b.finish(key,response={'id':'r','model':'gpt-4.1-mini-2025-04-14','usage':{
            'input_tokens':1000,'input_tokens_details':{'cached_tokens':800},'output_tokens':10}})
        self.assertAlmostEqual(self.b.summary()['spent_usd'],.000176)
        self.assertEqual(self.b.summary()['prompt_tokens'],1000)
    def test_unknown_failure_keeps_upper_reservation(self):
        key=self.b.reserve(self.payload(),'inference');before=self.b.summary()['guard_total_usd']
        self.b.finish(key,error_type='Timeout')
        self.assertEqual(self.b.summary()['guard_total_usd'],before)
    def test_monthly_cap_and_ny_boundary(self):
        key=self.b.reserve(self.payload(),'inference');p=self.root/'api-budget.json';s=json.loads(p.read_text())
        s['calls'][key]['guard_cost_usd']=100;p.write_text(json.dumps(s))
        with self.assertRaises(BudgetExceeded): self.b.reserve(self.payload(),'inference')
        self.at=datetime(2026,12,1,4,59,tzinfo=timezone.utc)
        with self.assertRaises(BudgetExceeded): self.b.reserve(self.payload(),'inference')
        self.at=datetime(2026,12,1,5,1,tzinfo=timezone.utc)
        self.b.reserve(self.payload(),'inference');self.assertEqual(self.b.summary()['month'],'2026-12')
    def test_no_unbounded_codex_or_unknown_model(self):
        with self.assertRaises(BudgetExceeded): self.b.reserve({'model':'gpt-5.3-codex','input':'hi'},'inference')
        with self.assertRaises(RuntimeError): model_rate('unpriced-new-model')
    def test_warning_is_persisted_once(self):
        key=self.b.reserve(self.payload(),'inference');p=self.root/'api-budget.json';s=json.loads(p.read_text())
        s['calls'][key]['actual_cost_usd']=20;s['calls'][key]['guard_cost_usd']=20;p.write_text(json.dumps(s))
        key2=self.b.reserve(self.payload(),'inference');self.b.finish(key2,response={'usage':{'input_tokens':10,'output_tokens':1}})
        self.assertTrue(json.loads(p.read_text())['warnings']['2026-11']['notification_pending'])

    def test_conservative_overrun_persists_block_even_after_reservation_cost_falls(self):
        key=self.b.reserve(self.payload(),'inference');p=self.root/'api-budget.json';state=json.loads(p.read_text())
        state['calls'][key]['guard_cost_usd']=99.99;p.write_text(json.dumps(state))
        with patch('live_testing.budget.cost',return_value=.02):
            with self.assertRaises(BudgetExceeded):self.b.reserve(self.payload(),'inference')
        state=json.loads(p.read_text());self.assertIn('2026-11',state['blocks']);self.assertEqual(len(state['calls']),1)
        self.b.finish(key,response={'usage':{'input_tokens':10,'output_tokens':1}})
        self.assertLess(self.b.summary()['guard_total_usd'],100);self.assertEqual(self.b.summary()['status'],'blocked')
        self.b.ack_warning('2026-11')
        with self.assertRaises(BudgetExceeded):self.b.reserve(self.payload(),'inference')
        self.at=datetime(2026,12,1,4,59,tzinfo=timezone.utc)
        with self.assertRaises(BudgetExceeded):self.b.reserve(self.payload(),'inference')
        self.at=datetime(2026,12,1,5,0,tzinfo=timezone.utc)
        self.b.reserve(self.payload(),'inference');self.assertEqual(self.b.summary()['status'],'ok')
        self.assertIn('2026-11',json.loads(p.read_text())['blocks'])

    def test_concurrent_reservations_cannot_cross_cap_and_block_all_later_calls(self):
        key=self.b.reserve(self.payload(),'inference');p=self.root/'api-budget.json';state=json.loads(p.read_text())
        state['calls'][key]['guard_cost_usd']=96.5;p.write_text(json.dumps(state))
        def reserve(_):
            try:return self.b.scoped('codex').reserve(self.payload(),'parallel')
            except BudgetExceeded:return None
        with patch('live_testing.budget.cost',return_value=1.):
            with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(reserve,range(24)))
        self.assertEqual(sum(value is not None for value in results),3)
        summary=self.b.summary();self.assertEqual(summary['guard_total_usd'],99.5);self.assertEqual(summary['status'],'blocked')
        self.assertEqual(len(json.loads(p.read_text())['calls']),4)

    def test_warning_ack_is_once_only_and_does_not_modify_calls_or_limits(self):
        key=self.b.reserve(self.payload(),'inference');p=self.root/'api-budget.json';state=json.loads(p.read_text())
        state['calls'][key].update(actual_cost_usd=20.,guard_cost_usd=20.);p.write_text(json.dumps(state))
        another=self.b.reserve(self.payload(),'inference');self.b.finish(another,response={'usage':{'input_tokens':10,'output_tokens':1}})
        self.assertTrue(self.b.summary()['warning_pending']);before=json.loads(p.read_text());created=before['warnings']['2026-11']['created_at']
        first=self.b.ack_warning('2026-11');second=self.b.ack_warning('2026-11');after=json.loads(p.read_text())
        self.assertEqual(first,second);self.assertEqual(before['calls'],after['calls'])
        self.assertEqual(after['limit_usd'],100);self.assertEqual(after['warning_usd'],20);self.assertFalse(self.b.summary()['warning_pending'])
        another=self.b.reserve(self.payload(),'inference');self.b.finish(another,response={'usage':{'input_tokens':10,'output_tokens':1}})
        self.assertEqual(json.loads(p.read_text())['warnings']['2026-11']['created_at'],created)
        self.assertFalse(self.b.summary()['warning_pending']);self.assertEqual(self.b.summary()['status'],'warning')
        self.assertFalse(self.b.ack_warning('2026-12')['acknowledged'])
        with self.assertRaises(ValueError):self.b.ack_warning('2026-13')

    def test_stale_pricing_fails_closed_without_mutating_any_budget_metadata(self):
        self.b.reserve(self.payload(),'inference');p=self.root/'api-budget.json';before=p.read_bytes()
        self.at=datetime(2026,12,6,16,tzinfo=timezone.utc)
        with self.assertRaises(PricingUnavailable):self.b.reserve(self.payload(),'inference')
        self.assertEqual(p.read_bytes(),before)

    def test_actual_pricing_review_cannot_reset_cap_or_change_rates(self):
        key=self.b.reserve(self.payload(),'inference');p=self.root/'api-budget.json';state=json.loads(p.read_text())
        state['blocks']['2026-11']={'reason':'cap'};p.write_text(json.dumps(state))
        evidence={'source_url':PRICING_SOURCE,'verified_at':self.at.isoformat(),
                  'rates':{k:list(v) for k,v in RATES.items()}}
        before=json.loads(p.read_text());self.b.confirm_pricing(evidence);after=json.loads(p.read_text())
        self.assertEqual(before['calls'],after['calls']);self.assertEqual(before['blocks'],after['blocks'])
        with self.assertRaises(BudgetExceeded):self.b.reserve(self.payload(),'inference')
        evidence['rates']['gpt-4.1-mini'][0]=0
        with self.assertRaises(PricingUnavailable):self.b.confirm_pricing(evidence)

if __name__=='__main__':unittest.main()
