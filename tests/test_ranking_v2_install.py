"""Synthetic transaction tests: no production files, credentials or providers."""
import copy
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from live_testing import ranking_v2_install as install
from live_testing.accounting import new_book, record_decision, record_valuation, settle_due
from live_testing.calendar import DEFAULT_CALENDAR, NY, aware
from live_testing.data import release_id, round_id
from live_testing.memory import MemoryUnit, Namespace
from live_testing.memory_v2 import CounterfactualMemoryBankV2
from live_testing.registry import MODEL_IDS, TICKERS
from live_testing.storage import digest, file_hash, read_json, write_private

NOW = datetime(2026, 11, 2, 18, tzinfo=NY).isoformat()


class Gateway:
    def __init__(self, root, scope=()): self.root, self.scope = root, scope
    def scoped(self, *scope): return Gateway(self.root, scope)


class Native:
    """A real native seam is replaced with a durable, explicitly fake provider."""
    def __init__(self, root):
        self.root = root; self.calls = []; self.seen = []; self.lock = threading.Lock()
        self.fail_after_accept = None; self.ambiguous = None
    def initialize(self, prepared, bundle, directory, training_bank, gateway, **kwargs):
        assert kwargs['scope'] == 'live' and gateway.root == self.root
        write_private(directory/'prepared.json', prepared)
        write_private(directory/'live-state.json', {'decisions': {}, 'attempts': {}, 'feedback': {},
            'model_id': kwargs['model_id'], 'seed_schema': training_bank['schema'], 'pending_break': False})
    def decide(self, obs, model, directory, bundle, gateway, *, asof, replay):
        state = read_json(directory/'live-state.json'); key = obs['round_id']; at = aware(obs['timestamp'])
        assert replay and gateway.root == self.root and gateway.scope[:2] == (model, 'replay')
        assert all(aware(f['exit_at']) <= at for f in state['feedback'].values())
        if key in state['attempts']: raise RuntimeError('An ambiguous paid request may not repeat')
        state['attempts'][key] = {'started_at': NOW, 'input_sha256': digest(obs), 'status': 'in_progress'}
        write_private(directory/'live-state.json', state)
        with self.lock:
            self.calls.append((model, key)); self.seen.append((model, at.hour, len(state['feedback'])))
            ledger = read_json(self.root/'api-budget.json')
            ledger['calls'][model+key] = {'status': 'completed', 'actual_cost_usd': .01}
            write_private(self.root/'api-budget.json', ledger)
        if self.ambiguous == (model, at.hour): raise RuntimeError('Synthetic unanswered provider request')
        pair = TICKERS[2:4] if at.hour == 15 else TICKERS[:2]
        monitor = {'pending_break': True, 'lifecycle': 'BREAK'} if state['decisions'] else None
        switched = bool(state['decisions']) and at.hour == 15
        state['pending_break'] = bool(monitor) and not switched
        action = {'stock1': pair[0], 'stock2': pair[1], 'action': 'LS', 'generated_at': NOW,
                  'used_memory_ids': ['synthetic-actually-cited']}
        state['decisions'][key] = {'timestamp': at.isoformat(), 'input_sha256': digest(obs), 'action': action,
                                   'monitor': monitor, 'pair_switched': switched}
        state['attempts'][key]['status'] = 'complete'
        write_private(directory/'live-state.json', state)
        if self.fail_after_accept == (model, at.hour):
            self.fail_after_accept = None; raise RuntimeError('Crash after durable paid acceptance')
        return action
    def apply_feedback(self, record, directory, bundle, gateway, model, *, source_cutoff):
        assert aware(record['exit_at']) <= source_cutoff
        state = read_json(directory/'live-state.json'); key = round_id(record['release_at'])
        if key in state['feedback']: assert state['feedback'][key] == record
        state['feedback'][key] = copy.deepcopy(record)
        write_private(directory/'live-state.json', state)
    def validate_state(self, state, *, model_id, directory, bundle):
        assert state['model_id'] == model_id and len(state['decisions']) == 6 and len(state['feedback']) == 4
        assert state['seed_schema'] == 'moira.counterfactual-memory/2'
        return True


class InstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()/'private'; self.root.mkdir(); root = self.root
        self.times = DEFAULT_CALENDAR.release_nodes[:6]; book = new_book(MODEL_IDS, TICKERS)
        runtime = {'published': {}, 'feedback_applied': {m: [] for m in MODEL_IDS},
                   'model_attempts': {}, 'completed_events': [], 'errors': {}}
        for index, at in enumerate(self.times):
            prices = {ticker: 100.+index*(j+1)/10 for j, ticker in enumerate(TICKERS)}
            record_valuation(book, at, prices, observed_at=at+timedelta(minutes=1), provenance={'synthetic': True})
            data = {'timestamp': at.isoformat(), 'prices': prices, 'news_rows': [], 'coverage': 'partial'}
            for folder in ('inputs', 'releases'): write_private(root/folder/(release_id(at)+'.json'), data)
            folder = 'data/'+release_id(at); hashes = {}
            for name in ('price.parquet', 'news.parquet'):
                path = root/'news/hf-staging'/folder/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'Synthetic published bytes'); hashes[name] = file_hash(path)
            runtime['published'][at.isoformat()] = {'folder': folder, 'file_sha256': hashes}
            for model in MODEL_IDS:
                record_decision(book, model, at, {'stock1': TICKERS[0], 'stock2': TICKERS[1], 'action': 'F'},
                                generated_at=at+timedelta(minutes=1), input_asof=at)
        settle_due(book, aware(NOW))
        for model in MODEL_IDS:
            write_private(root/'models'/model/'original-private.json', {'old': model})
        for name, value in (('book.json', book), ('runtime.json', runtime), ('config.json', {'synthetic': True}),
                ('company-context.json', {'dated_before_first_source': True}),
                ('seed/moira-bundle.json', {'frozen': True}),
                ('api-budget.json', {'calls': {'old': {'status': 'completed', 'actual_cost_usd': 1.}}})):
            write_private(root/name, value)
        self.training = {}
        for model in install.MODELS:
            directory = root/'training'/model; self.training[model] = directory
            text = 'Synthetic evaluated transferable prompt.'
            prepared = {'status': 'ready', 'model': 'gpt-4.1-mini', 'memory_policy_protocol': install.PROTOCOL,
                'prompt_memory': {'active_prompt': text}, 'frozen_prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}
            bank = CounterfactualMemoryBankV2(Namespace(model, 'training', 'synthetic'), embedding_id='synthetic', dimension=2)
            for index in range(29):
                at = datetime(2026, 10, 19, 9, 30, tzinfo=NY)+timedelta(minutes=index*5)
                unit = MemoryUnit(str(index), str(index), TICKERS[0], TICKERS[1], 'F', at,
                    at+timedelta(minutes=1), at+timedelta(minutes=2), at+timedelta(minutes=2), 0.,
                    {'LS': .01, 'SL': -.01, 'F': 0.}, 'Synthetic whole historical reflection.',
                    tuple(TICKERS[:2]), {'synthetic': True})
                bank.add_finalized(unit, as_of=at+timedelta(minutes=3), embedding=[1., 0.])
            report = {'objective': 'sample_sharpe_without_annualization', 'completed_cycles': 1, 'max_cycles': 8,
                'checkpoint': prepared, 'cycles': [{'baseline_metrics': {'sharpe_ratio': .1},
                    'candidate_metrics': {'sharpe_ratio': .2}, 'accepted': True}]}
            for name, value in (('prepared.json', prepared), ('retained-training-bank.json', bank.to_dict()),
                    ('training-result.json', report), ('source-proof.json', {'synthetic': True}),
                    ('parent-checkpoint.json', {'retained_old_training': True})):
                write_private(directory/name, value)
            write_private(directory/'status.json', {'status': 'ready', 'model_id': model, 'finished_at': NOW,
                'checkpoint_sha256': file_hash(directory/'prepared.json'),
                'retained_bank_sha256': file_hash(directory/'retained-training-bank.json')})
        self.native = Native(root); self.gateway = Gateway(root)
    def auth(self):
        path = self.root/'authorization.json'
        with patch.object(install, 'now', return_value=NOW):
            write_private(path, install.authorization(self.root, self.training, statement='Explicit synthetic authorization.'))
        return path
    def run_install(self, auth):
        with patch.object(install, 'now', return_value=NOW):
            return install.install(self.root, auth, self.gateway, native=self.native)
    def test_three_models_install_together_preserving_others_quotes_and_costs(self):
        before = read_json(self.root/'book.json'); trees = {m: install._tree(self.root/'models'/m) for m in MODEL_IDS}
        result = self.run_install(self.auth()); self.assertEqual(result['status'], 'installed')
        after = read_json(self.root/'book.json'); self.assertEqual(after['valuations'], before['valuations'])
        for model in MODEL_IDS:
            if model not in install.MODELS:
                self.assertEqual(after['accounts'][model], before['accounts'][model])
                self.assertEqual(after['decisions'][model], before['decisions'][model])
                self.assertEqual(install._tree(self.root/'models'/model), trees[model])
        for model in install.MODELS:
            self.assertEqual(len(after['accounts'][model]['history']), 4)
            self.assertTrue(all(r['mode']=='retrospective_point_in_time' for r in after['decisions'][model].values()))
            self.assertEqual([(hour, count) for m, hour, count in self.native.seen if m==model],
                             [(9, 0), (10, 0), (11, 1), (12, 2), (14, 3), (15, 4)])
            marker = read_json(self.root/'models'/model/'memory-policy.json')
            self.assertEqual(marker['protocol'], install.PROTOCOL); self.assertEqual(marker['status'], 'ready')
        self.assertEqual(len(self.native.calls), 18)
        self.assertEqual(read_json(self.root/'api-budget.json')['calls']['old'], {'status': 'completed', 'actual_cost_usd': 1.})
    def test_cached_acceptance_and_completed_other_variants_resume_without_paid_repeats(self):
        self.native.fail_after_accept = (install.MODELS[0], 12); auth = self.auth()
        before = file_hash(self.root/'book.json'); self.assertEqual(self.run_install(auth)['status'], 'pending')
        self.assertEqual(file_hash(self.root/'book.json'), before)
        result = self.run_install(auth); self.assertEqual(result['status'], 'installed')
        self.assertEqual(len(self.native.calls), 18); self.assertEqual(len(set(self.native.calls)), 18)
        self.assertEqual(self.run_install(auth), result); self.assertEqual(len(self.native.calls), 18)
    def test_unanswered_provider_request_stays_pending_without_flat_or_retry(self):
        self.native.ambiguous = (install.MODELS[0], 9); auth = self.auth(); before = file_hash(self.root/'book.json')
        self.assertEqual(self.run_install(auth)['status'], 'pending'); first = len(self.native.calls)
        self.assertEqual(self.run_install(auth)['status'], 'pending')
        self.assertEqual(len(self.native.calls), first); self.assertEqual(file_hash(self.root/'book.json'), before)
    def test_mid_commit_failure_restores_all_three_old_directories_and_exact_books(self):
        auth = self.auth(); before = {n: (self.root/n).read_bytes() for n in ('book.json', 'runtime.json')}
        trees = {m: install._tree(self.root/'models'/m) for m in install.MODELS}; write = install.write_private
        def fail(path, value):
            if Path(path)==self.root/'runtime.json': raise OSError('Synthetic installation failure')
            return write(path, value)
        with patch.object(install, 'write_private', side_effect=fail), self.assertRaises(OSError): self.run_install(auth)
        self.assertEqual({n: (self.root/n).read_bytes() for n in before}, before)
        self.assertEqual({m: install._tree(self.root/'models'/m) for m in install.MODELS}, trees)
        self.assertEqual(self.run_install(auth)['status'], 'installed'); self.assertEqual(len(self.native.calls), 18)
    def test_interrupted_commit_holds_dispatch_then_recovers_and_reuses_shadows(self):
        self.native.fail_after_accept = (install.MODELS[0], 15); auth = self.auth()
        self.assertEqual(self.run_install(auth)['status'], 'pending')
        record = read_json(auth); op = self.root/'repairs/ranking-v2'/digest(record)[:24]
        journal = read_json(op/'status.json'); journal['status']='committing'; write_private(op/'status.json', journal)
        (op/'live-before').mkdir()
        model = install.MODELS[1]
        (self.root/'models'/model).rename(op/'live-before'/model)
        (op/'variants'/model/'native').rename(self.root/'models'/model)
        with self.assertRaises(RuntimeError): install.assert_installation_consistent(self.root)
        self.assertEqual(self.run_install(auth)['status'], 'installed'); self.assertEqual(len(self.native.calls), 18)
        install.assert_installation_consistent(self.root)
    def test_future_genuine_decision_and_changed_input_refuse_before_any_request(self):
        auth = self.auth(); write_private(self.root/'company-context.json', {'changed': True})
        with self.assertRaises(ValueError): self.run_install(auth)
        self.assertEqual(self.native.calls, [])
        book = read_json(self.root/'book.json'); at = DEFAULT_CALENDAR.release_nodes[6]
        record_decision(book, install.MODELS[0], at, {'stock1': TICKERS[0], 'stock2': TICKERS[1], 'action': 'F'},
                        generated_at=at+timedelta(minutes=1), input_asof=at)
        write_private(self.root/'book.json', book)
        with self.assertRaises(ValueError): self.auth()
    def test_schema1_and_changed_normalized_policy_cannot_be_authorized(self):
        directory = self.training[install.MODELS[0]]
        bank = read_json(directory/'retained-training-bank.json'); bank['schema']='moira.counterfactual-memory/1'
        write_private(directory/'retained-training-bank.json', bank)
        status = read_json(directory/'status.json')
        status['retained_bank_sha256']=file_hash(directory/'retained-training-bank.json')
        write_private(directory/'status.json', status)
        with self.assertRaises(ValueError): self.auth()
        self.assertEqual(self.native.calls, [])
    def test_missing_attribution_evidence_prevents_every_live_install(self):
        auth = self.auth(); before = file_hash(self.root/'book.json')
        with patch.object(self.native, 'validate_state', side_effect=ValueError('Incomplete actual LOO evidence')):
            self.assertEqual(self.run_install(auth)['status'], 'pending')
        self.assertEqual(file_hash(self.root/'book.json'), before)
        self.assertEqual(self.run_install(auth)['status'], 'installed')
        self.assertEqual(len(self.native.calls), 18)
    def test_historical_cost_row_mutation_is_never_accepted_or_refunded(self):
        self.native.fail_after_accept = (install.MODELS[0], 15); auth = self.auth()
        self.assertEqual(self.run_install(auth)['status'], 'pending')
        ledger = read_json(self.root/'api-budget.json'); ledger['calls']['old']['actual_cost_usd']=0.
        write_private(self.root/'api-budget.json', ledger); before=file_hash(self.root/'book.json')
        with self.assertRaisesRegex(ValueError, 'Historical API costs'): self.run_install(auth)
        self.assertEqual(file_hash(self.root/'book.json'), before)
    def test_interrupted_prepaid_archive_copy_can_resume_without_new_authorization(self):
        auth = self.auth(); original = install.shutil.copytree; failed = [False]
        def fail(source, target, **kwargs):
            if Path(target).name == install.MODELS[1] and not failed[0]:
                failed[0]=True; Path(target).mkdir(parents=True, exist_ok=True)
                raise OSError('Synthetic process interruption before the paid-work journal')
            return original(source, target, **kwargs)
        with patch.object(install.shutil, 'copytree', side_effect=fail), self.assertRaises(OSError):
            self.run_install(auth)
        self.assertEqual(self.native.calls, [])
        self.assertEqual(self.run_install(auth)['status'], 'installed')
        self.assertEqual(len(self.native.calls), 18)


if __name__ == '__main__': unittest.main()
