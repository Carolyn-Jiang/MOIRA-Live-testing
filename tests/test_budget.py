import json
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from live_testing.budget import Budget,BudgetExceeded,PricingUnavailable,model_rate,cost,usage_cost,tokens,RATES,PRICING_SOURCE

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

    def sol_usage(self, *, incoming=12000, cached=4000, short=3000, long=4000, outgoing=100):
        return {'input_tokens':incoming,'output_tokens':outgoing,'total_tokens':incoming+outgoing,
                'input_tokens_details':{'cached_tokens':cached,'cache_write_tokens':short+long,
                    'cache_write_tokens_details':{'ttl_30m_tokens':short,'ttl_12h_tokens':long}}}

    def test_sol_cache_reads_and_each_write_lifetime_partition_input_exactly_once(self):
        usage=self.sol_usage()
        self.assertEqual(tokens(usage),(12000,4000,100))
        # Ordinary 1000*2 + read 4000*.1 + 30m 3000*2.5 + 12h 4000*5 + output 100*10.
        self.assertEqual(usage_cost('gpt-6.1-sol',usage),.0309)
        self.assertEqual(model_rate('gpt-6.1-sol-2026-11-05'),(2.,.1,10.))
        self.assertEqual(usage_cost('gpt-6.1-sol-2026-11-05',usage),.0309)
        # The total is not a third write charge; it only validates the two subsets.
        without_total=json.loads(json.dumps(usage));del without_total['input_tokens_details']['cache_write_tokens']
        self.assertEqual(usage_cost('gpt-6.1-sol',without_total),.0309)

    def test_sol_long_context_surcharge_applies_to_whole_request_and_all_input_subsets(self):
        self.assertEqual(cost('gpt-6.1-sol',272000,0,100),.545)
        self.assertEqual(cost('gpt-6.1-sol',272001,0,100),1.089504)
        usage=self.sol_usage(incoming=272001,cached=200000,short=30000,long=40000,outgoing=100)
        self.assertEqual(usage_cost('gpt-6.1-sol',usage),.599504)
        # Cached 2x and output 1.5x apply even when every input is a billed read.
        usage=self.sol_usage(incoming=272001,cached=272001,short=0,long=0,outgoing=100)
        self.assertAlmostEqual(usage_cost('gpt-6.1-sol',usage),.0559002)

    def test_sol_single_known_ttl_can_price_total_but_unknown_allocation_is_not_guessed(self):
        usage={'input_tokens':1000,'output_tokens':10,
               'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':800}}
        self.assertEqual(usage_cost('gpt-6.1-sol',usage,cache_write_ttl='30m'),.0025)
        self.assertEqual(usage_cost('gpt-6.1-sol',usage,cache_write_ttl='12h'),.0045)
        with self.assertRaises(ValueError):usage_cost('gpt-6.1-sol',usage)
        usage['input_tokens_details']['cache_write_tokens']=0
        self.assertEqual(usage_cost('gpt-6.1-sol',usage),.0021)
        del usage['input_tokens_details']['cache_write_tokens']
        with self.assertRaises(ValueError):usage_cost('gpt-6.1-sol',usage,cache_write_ttl='30m')

    def test_sol_rejects_overlapping_subsets_bad_lifetime_totals_and_missing_measurements(self):
        edits=[lambda u:u['input_tokens_details'].update(cache_write_tokens=7001),
               lambda u:u['input_tokens_details'].update(cached_tokens=6000),
               lambda u:u['input_tokens_details']['cache_write_tokens_details'].update(ttl_12h_tokens=-1),
               lambda u:u['input_tokens_details']['cache_write_tokens_details'].pop('ttl_30m_tokens'),
               lambda u:u['input_tokens_details'].pop('cached_tokens'),
               lambda u:u.pop('output_tokens'),
               lambda u:u.update(cache_write_tokens=1)]
        for edit in edits:
            usage=self.sol_usage();edit(usage)
            with self.assertRaises(ValueError):usage_cost('gpt-6.1-sol',usage)

    def test_sol_reserve_covers_12h_writes_and_byte_bound_long_context(self):
        for text in ('short','x'*300000):
            payload={'model':'gpt-6.1-sol','input':text,'max_output_tokens':100}
            identity=self.b.reserve(payload,'sol-synthetic')
            row=json.loads((self.root/'api-budget.json').read_text())['calls'][identity]
            maximum=len(json.dumps(payload,ensure_ascii=False).encode())+4096
            expected=(maximum*(10 if maximum>272000 else 5)+100*(15 if maximum>272000 else 10))/1000000
            self.assertEqual(row['reserved_upper_usd'],expected)
            self.assertEqual(row['cache_write_ttl'],'30m')

    def test_sol_finish_uses_exact_reported_components_and_does_not_reprice_old_calls(self):
        old=self.b.reserve(self.payload(),'old')
        self.b.finish(old,response={'usage':{'input_tokens':1000,'output_tokens':10,
            'input_tokens_details':{'cached_tokens':800}}})
        before=json.loads((self.root/'api-budget.json').read_text())['calls'][old]
        payload={'model':'gpt-6.1-sol','input':'x'*12000,'max_output_tokens':100}
        identity=self.b.reserve(payload,'sol');self.b.finish(identity,response={
            'id':'sol-synthetic-receipt','model':'gpt-6.1-sol','usage':self.sol_usage()})
        state=json.loads((self.root/'api-budget.json').read_text());row=state['calls'][identity]
        self.assertEqual(row['status'],'completed');self.assertEqual(row['actual_cost_usd'],.0309)
        self.assertEqual(row['guard_cost_usd'],.0309);self.assertEqual(row['total_tokens'],12100)
        self.assertEqual(state['calls'][old],before)
        self.assertEqual((state['warning_usd'],state['limit_usd']),(20.,100.))

    def test_sol_ambiguous_write_allocation_retains_exact_provider_evidence_and_reservation(self):
        payload={'model':'gpt-6.1-sol','input':[{'prompt_cache_breakpoint':{'ttl':'12h'},'text':'x'*1000}],
                 'max_output_tokens':100,'prompt_cache_options':{'ttl':'30m'}}
        identity=self.b.reserve(payload,'mixed-sol')
        before=json.loads((self.root/'api-budget.json').read_text())['calls'][identity]
        self.assertIsNone(before['cache_write_ttl'])
        usage={'input_tokens':1000,'output_tokens':10,
               'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':800}}
        self.b.finish(identity,response={'id':'synthetic-sol-unknown-allocation','model':'gpt-6.1-sol','usage':usage})
        row=json.loads((self.root/'api-budget.json').read_text())['calls'][identity]
        self.assertEqual(row['status'],'completed');self.assertNotIn('actual_cost_usd',row)
        self.assertEqual(row['cost_status'],'unavailable')
        self.assertEqual((row['input_tokens'],row['cached_input_tokens'],row['output_tokens'],row['total_tokens']),
                         (1000,0,10,1010))
        self.assertEqual(row['usage'],usage);self.assertEqual(row['guard_cost_usd'],before['guard_cost_usd'])
        self.assertEqual(row['response_id'],'synthetic-sol-unknown-allocation')

    def test_sol_missing_write_measurement_preserves_known_tokens_but_not_invented_price(self):
        payload={'model':'gpt-6.1-sol','input':'synthetic','max_output_tokens':100}
        identity=self.b.reserve(payload,'sol-missing-write')
        before=json.loads((self.root/'api-budget.json').read_text())['calls'][identity]
        usage={'input_tokens':1000,'output_tokens':10,'total_tokens':1010,
               'input_tokens_details':{'cached_tokens':500}}
        self.b.finish(identity,response={'id':'synthetic-sol-receipt','usage':usage})
        row=json.loads((self.root/'api-budget.json').read_text())['calls'][identity]
        self.assertEqual(row['status'],'completed');self.assertEqual(row['total_tokens'],1010)
        self.assertEqual(row['cached_input_tokens'],500)
        self.assertEqual(row['cost_status'],'unavailable');self.assertNotIn('actual_cost_usd',row)
        self.assertEqual(row['guard_cost_usd'],before['guard_cost_usd'])
        self.assertEqual(row['usage'],usage)

    def test_sol_missing_actual_output_or_inconsistent_total_remains_usage_unavailable(self):
        for usage in ({'input_tokens':1000,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}},
                      {'input_tokens':1000,'output_tokens':10,'total_tokens':1,
                       'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}):
            identity=self.b.reserve({'model':'gpt-6.1-sol','input':'synthetic','max_output_tokens':100},'sol-invalid')
            self.b.finish(identity,response={'usage':usage})
            row=json.loads((self.root/'api-budget.json').read_text())['calls'][identity]
            self.assertEqual(row['status'],'usage-unavailable');self.assertNotIn('actual_cost_usd',row)

    def test_legacy_rates_and_three_tuple_usage_remain_byte_for_byte_compatible(self):
        expected={'gpt-4.1-mini':.000176,'gpt-5.3-codex':.00063,
                  'text-embedding-3-large':.00013,'text-embedding-3-small':.00002}
        for model,value in expected.items():
            outgoing=0 if model.startswith('text-embedding-') else 10
            self.assertEqual(cost(model,1000,800,outgoing),value)
            usage={'prompt_tokens':1000,'completion_tokens':outgoing,'prompt_tokens_details':{'cached_tokens':800}}
            self.assertEqual(tokens(usage),(1000,800,outgoing))
            self.assertEqual(usage_cost(model,usage),value)

if __name__=='__main__':unittest.main()
