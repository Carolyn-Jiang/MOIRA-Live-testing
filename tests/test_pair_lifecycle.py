import copy
from datetime import timedelta,timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from live_testing.accounting import new_book,record_decision
from live_testing import bootstrap,runner
from live_testing.calendar import DEFAULT_CALENDAR as C
from live_testing.migration import import_legacy_actions
from live_testing.pair_lifecycle import (MOIRA_MODELS,PairLifecycleViolation,
    validate_pair_lifecycle,validate_book_pair_lifecycle,validate_next_pair,validate_legacy_moira_seals)
from live_testing.registry import MODEL_IDS,TICKERS


def action(pair=('NVDA','GOOG'),label='LS'):return {'stock1':pair[0],'stock2':pair[1],'action':label}
def row(at,pair=('NVDA','GOOG'),label='LS'):return {'release_at':at.isoformat(),'action':action(pair,label)}
def source_state():
    state={'schema':'finreason.task2.phrase3-private/1','phase_id':'phrase-iii-2026-11-02','sealed':{}}
    for i,at in enumerate(C.release_nodes[:2]):
        state['sealed'][at.strftime('%Y-%m-%d_%H:%M')]={model:{'action':action(('NVDA','GOOG')if not i else('META','MSFT'))}
            for model in ('Codex','FinMem','Moira')}
    return state


class PairLifecycleTests(unittest.TestCase):
    def test_each_moira_version_allows_first_choice_and_blocks_every_other_ordered_change_before_1530(self):
        for model in MOIRA_MODELS:
            first,second=C.release_nodes[:2];rows={first.isoformat():row(first)}
            self.assertEqual(validate_pair_lifecycle(model,rows)['checked_decisions'],1)
            for pair in [('META','MSFT'),('GOOG','NVDA')]:
                for label in ('LS','SL','F'):
                    candidate={**rows,second.isoformat():row(second,pair,label)};before=copy.deepcopy(candidate)
                    with self.subTest(model=model,pair=pair,label=label),self.assertRaises(PairLifecycleViolation):
                        validate_pair_lifecycle(model,candidate)
                    self.assertEqual(candidate,before)
            same={**rows,second.isoformat():row(second,label='SL')}
            self.assertEqual(validate_pair_lifecycle(model,same)['pair_changes'],0)

    def test_change_is_allowed_only_at_a_real_new_york_1530_release(self):
        first=C.release_nodes[0];last=C.release_nodes[5];rows={first.isoformat():row(first)}
        changed={**rows,last.isoformat():row(last,('META','MSFT'))}
        report=validate_pair_lifecycle('moira',changed)
        self.assertEqual(report['change_releases'],[last.isoformat()]);self.assertEqual(report['pair_changes'],1)
        # UTC 20:30 is the same authorized November NY 15:30 boundary.
        utc=last.astimezone(timezone.utc);changed={**rows,utc.isoformat():row(utc,('META','MSFT'))}
        self.assertEqual(validate_pair_lifecycle('moira',changed)['pair_changes'],1)
        for stamp in [last+timedelta(days=5),last+timedelta(seconds=1),C.valuation_nodes[-2],
                      next(at for at in C.release_nodes if at.date().isoformat()=='2026-11-27')+timedelta(hours=6)]:
            with self.subTest(stamp=stamp),self.assertRaises(PairLifecycleViolation):
                validate_pair_lifecycle('moira',{**rows,stamp.isoformat():row(stamp,('META','MSFT'))})

    def test_chronology_and_day_boundaries_cannot_hide_an_illegal_change(self):
        first=C.release_nodes[0];last=C.release_nodes[5];next_day=C.release_nodes[6]
        rows={next_day.isoformat():row(next_day,('AAPL','AMZN')),last.isoformat():row(last,('META','MSFT')),
              first.isoformat():row(first)}
        with self.assertRaises(PairLifecycleViolation):validate_pair_lifecycle('moira',rows)
        duplicate={first.isoformat():row(first),first.astimezone(timezone.utc).isoformat():row(first)}
        with self.assertRaises(PairLifecycleViolation):validate_pair_lifecycle('moira',duplicate)
        with self.assertRaises(PairLifecycleViolation):validate_pair_lifecycle('moira',{'bad':{'action':action()}})
        self.assertEqual(validate_pair_lifecycle('moira',{})['checked_decisions'],0)

    def test_failed_legacy_import_leaves_book_and_old_source_unchanged(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory);source=source_state();(path/'state.json').write_text(json.dumps(source))
            before=(path/'state.json').read_bytes();book=new_book(MODEL_IDS,TICKERS);old=copy.deepcopy(book)
            with self.assertRaises(PairLifecycleViolation):import_legacy_actions(book,path)
            self.assertEqual(book,old);self.assertEqual((path/'state.json').read_bytes(),before)
            self.assertEqual(list(path.iterdir()),[path/'state.json'])
        with self.assertRaises(PairLifecycleViolation):validate_legacy_moira_seals(source_state())

    def test_bootstrap_rejects_old_intraday_change_before_any_native_or_seed_copy(self):
        with tempfile.TemporaryDirectory()as directory:
            parent=Path(directory);legacy=parent/'legacy';legacy.mkdir();source=legacy/'state.json'
            source.write_text(json.dumps(source_state()));before=source.read_bytes();new=parent/'new'
            with patch.dict(os.environ,{},clear=False),patch.object(bootstrap.shutil,'copytree')as copytree:
                with self.assertRaises(PairLifecycleViolation):bootstrap.prepare(new,legacy,parent/'framework',parent/'key.env')
            copytree.assert_not_called();self.assertEqual(source.read_bytes(),before)
            self.assertFalse((new/'seed').exists());self.assertFalse((new/'models').exists());self.assertFalse((new/'config.json').exists())

    def runtime(self,root):
        runtime=runner.Runtime.__new__(runner.Runtime);runtime.root=root;runtime.book=new_book(MODEL_IDS,TICKERS)
        runtime._gateway=SimpleNamespace(scoped=Mock(return_value=object()));runtime.require_variant_training=Mock()
        return runtime

    def test_runner_checks_all_existing_moira_chains_before_opening_any_gateway(self):
        for model in MOIRA_MODELS:
            with tempfile.TemporaryDirectory()as directory:
                runtime=self.runtime(Path(directory))
                for at,pair in zip(C.release_nodes[:2],[('NVDA','GOOG'),('META','MSFT')]):
                    record_decision(runtime.book,model,at,action(pair),generated_at=at+timedelta(minutes=1),input_asof=at)
                with self.assertRaises(PairLifecycleViolation):runtime.decide('codex',{'timestamp':C.release_nodes[2].isoformat()})
                runtime._gateway.scoped.assert_not_called();runtime.require_variant_training.assert_not_called()

    def test_runner_rejects_a_new_illegal_native_output_without_accepting_it_into_accounting(self):
        with tempfile.TemporaryDirectory()as directory:
            runtime=self.runtime(Path(directory));first,second=C.release_nodes[:2]
            record_decision(runtime.book,'moira',first,action(),generated_at=first+timedelta(minutes=1),input_asof=first)
            before=copy.deepcopy(runtime.book);generated={**action(('META','MSFT')),'generated_at':second.isoformat(),'model':'gpt-4.1-mini'}
            with patch('live_testing.moira_native.decide',return_value=generated)as inference:
                with self.assertRaises(PairLifecycleViolation):runtime.decide('moira',{'timestamp':second.isoformat(),'round_id':'synthetic'})
            inference.assert_called_once();self.assertEqual(runtime.book,before)

    def test_book_guard_excludes_other_models_and_next_candidate_does_not_mutate_history(self):
        book=new_book(MODEL_IDS,TICKERS);first,second=C.release_nodes[:2]
        for at,pair in zip((first,second),[('NVDA','GOOG'),('META','MSFT')]):
            record_decision(book,'codex',at,action(pair),generated_at=at+timedelta(minutes=1),input_asof=at)
        report=validate_book_pair_lifecycle(book);self.assertEqual(set(report),set(MOIRA_MODELS))
        rows={first.isoformat():row(first)};before=copy.deepcopy(rows)
        with self.assertRaises(PairLifecycleViolation):validate_next_pair('moira',rows,second,action(('META','MSFT')))
        self.assertEqual(rows,before)


if __name__=='__main__':unittest.main()
