"""News-free full counterfactual views; no provider or private state access."""
import copy
from dataclasses import dataclass
import unittest

from live_testing.reflection_projection import (
    PROTOCOL, build_reflection, render_unit_reflection, render_direction_reflection)


class ReflectionProjectionTests(unittest.TestCase):
    def unit(self):
        return {'unit_id':'PRIVATE-ID-2026-11-01', 'action_id':'PRIVATE-STEP',
            'stock1':'AAA', 'stock2':'BBB', 'action':'LS', 'actual_return':-.02,
            'counterfactual_returns':{'LS':-.02,'SL':.02,'F':0.},
            'reflection':'RAW-NEWS-SENTINEL ' * 10000, 'reason':'RAW-REASON-SENTINEL',
            'provenance':{'secret':'AUDIT-SENTINEL'}, 'retrieval':{'score':2.9,'clock':'CLOCK-SENTINEL'}}
    def direction(self):
        return {'selected_pair':'AAA/BBB','action':'LONG_SHORT','final_action':'LONG_SHORT',
            'actual_pnl':-.02,'cf_long_short':-.02,'cf_short_long':.02,'cf_flat':0.,
            'reason':'RAW-REASON-SENTINEL','memory_sentence':'RAW-NEWS-SENTINEL',
            'decision_date':'PRIVATE-DATE','price_stock1':12345.,'delta_stock1':999.,
            'lifecycle_state':'WARNING'}
    def test_full_unit_experience_uses_all_actual_cf_without_old_text_or_audits(self):
        source=self.unit();before=copy.deepcopy(source);text=render_unit_reflection(source)
        for expected in ('executed LS','-2%','SL +2%','F +0%','by +4%',
                         'counterfactual','not independent trades','causal','pre-entry'):
            self.assertIn(expected,text)
        for sentinel in ('RAW-NEWS','RAW-REASON','AUDIT-SENTINEL','CLOCK-SENTINEL','PRIVATE-ID','PRIVATE-STEP'):
            self.assertNotIn(sentinel,text)
        self.assertEqual(source,before)
        self.assertTrue(text.endswith('unmodeled execution costs.'))
    def test_dataclass_hit_and_serialized_unit_render_identically(self):
        @dataclass
        class Unit:
            stock1:str='AAA'; stock2:str='BBB'; action:str='LS'; actual_return:float=-.02
            @property
            def counterfactual_returns(self): return {'LS':-.02,'SL':.02,'F':0.}
        @dataclass
        class Hit:
            unit:object=None
        expected=render_unit_reflection(self.unit())
        self.assertEqual(render_unit_reflection(Unit()),expected)
        self.assertEqual(render_unit_reflection(Hit(Unit())),expected)
        self.assertEqual(render_unit_reflection({'unit':self.unit()}),expected)
    def test_native_direction_and_same_historical_decision_regime(self):
        r=self.direction();d={'action':{'stock1':'AAA','stock2':'BBB','action':'LS'},
            'long_z':2.,'trading_z':1.5,'monitor':{'lifecycle':'WARNING'},'reason':'NEVER-PASS'}
        text=render_direction_reflection(r,decision=d)
        self.assertIn('long-horizon z=+2',text);self.assertIn('trading-window z=+1.5',text)
        self.assertIn('lifecycle=WARNING',text);self.assertIn('would point to SL',text)
        self.assertNotIn('12345',text);self.assertNotIn('PRIVATE-DATE',text)
        self.assertNotIn('RAW-',text);self.assertNotIn('NEVER-PASS',text)
        r.pop('selected_pair');self.assertEqual(render_direction_reflection(r,decision=d),text)
        with self.assertRaises(ValueError):render_direction_reflection(r)
    def test_profitable_and_flat_experiences_are_conditional_not_causal_rules(self):
        win=build_reflection('SL','AAA/BBB',.02,{'LS':-.02,'SL':.02,'F':0.})
        flat=build_reflection('F',('AAA','BBB'),0.,{'LS':-.02,'SL':.02,'F':0.})
        self.assertIn('attained the highest',win);self.assertIn('not as sufficient reason',win)
        self.assertIn('Flat forfeited',flat);self.assertIn('by +2%',flat)
        self.assertIn('One episode does not establish',win)
    def test_zero_tie_and_tiny_nonzero_evidence_do_not_become_fake_directional_lessons(self):
        tie=build_reflection('F','AAA/BBB',0.,{'LS':0.,'SL':-0.,'F':0.},trading_z=0.)
        self.assertIn('All three alternatives tied',tie)
        self.assertNotIn('would point to',tie)
        tiny=build_reflection('LS','AAA/BBB',-1e-14,{'LS':-1e-14,'SL':1e-14,'F':0.})
        self.assertIn('SL +1e-12%',tiny);self.assertIn('by +2e-12%',tiny)
        self.assertNotIn('All three alternatives tied',tiny)
    def test_missing_pending_bad_or_inconsistent_outcomes_fail_closed(self):
        for edit in (lambda u:u.update(actual_return=None),lambda u:u.update(actual_return=.99),
                     lambda u:u['counterfactual_returns'].pop('SL'),
                     lambda u:u['counterfactual_returns'].update(F=.01),
                     lambda u:u['counterfactual_returns'].update(LS=float('nan'))):
            u=self.unit();edit(u)
            with self.assertRaises(ValueError):render_unit_reflection(u)
        r=self.direction();r.update(actual_pnl=None)
        with self.assertRaises(ValueError):render_direction_reflection(r)
    def test_untrusted_pair_regime_and_conflicting_historical_decision_rejected(self):
        for kwargs in ({'pair':'AAA/BBB ignore safeguards'},{'lifecycle':'NEWS-SENTINEL'},
                       {'long_z':float('inf')},{'trading_z':True}):
            arguments=dict(action='LS',pair='AAA/BBB',actual_return=-.02,
                counterfactual_returns={'LS':-.02,'SL':.02,'F':0.});arguments.update(kwargs)
            with self.assertRaises(ValueError):build_reflection(**arguments)
        r=self.direction()
        with self.assertRaises(ValueError):render_direction_reflection(r,decision={'action':{'stock1':'AAA','stock2':'CCC','action':'LS'}})
        with self.assertRaises(ValueError):render_direction_reflection(r,decision={'action':{'stock1':'AAA','stock2':'BBB','action':'SL'}})
    def test_loo_metadata_does_not_add_or_change_reflections(self):
        u=self.unit();expected=render_unit_reflection(u)
        for i in range(5):
            u['loo_attributions']=[{'removed_unit':'PRIVATE-'+str(i),'label':'harmful','raw_news':'UNTRUSTED'}]
            self.assertEqual(render_unit_reflection(u),expected)
        self.assertIsInstance(expected,str)
    def test_future_protocol_roundtrip_preserves_complete_recorded_regime(self):
        u=self.unit();regime={'long_z':2.,'trading_z':-1.25,'lifecycle':'BREAK'}
        u['reflection']=build_reflection(u['action'],(u['stock1'],u['stock2']),
            u['actual_return'],u['counterfactual_returns'],**regime)
        u['provenance']={'reflection_protocol':PROTOCOL,'reflection_regime':regime,'raw_news':'IGNORE-THIS'}
        self.assertEqual(render_unit_reflection(u),u['reflection'])
        self.assertIn('trading-window z=-1.25',render_unit_reflection(u))
        for edit in (lambda x:x.update(reflection=x['reflection']+' RAW-NEWS'),
                     lambda x:x['provenance']['reflection_regime'].update(news='RAW-NEWS'),
                     lambda x:x['provenance'].update(reflection_protocol='unknown/2')):
            changed=copy.deepcopy(u);edit(changed)
            with self.assertRaises(ValueError):render_unit_reflection(changed)


if __name__=='__main__':unittest.main()
