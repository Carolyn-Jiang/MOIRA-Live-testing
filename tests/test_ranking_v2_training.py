"""Training job boundaries: never contact a provider in these tests."""
from pathlib import Path
import tempfile
import unittest

from live_testing.ranking_v2_training import train_model, _code_sources, _checked_files
from live_testing.memory_v2 import PROTOCOL_VERSION
from live_testing.storage import file_hash, write_private


class TrainingBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root/'staged'

    def test_requires_explicit_registered_variant_and_cycle_bound(self):
        for model, authorization, limit in [('moira', 'authorized', 8),
                ('moira2-top1', '', 8), ('moira2-top1', 'authorized', 9),
                ('moira2-top1', 'authorized', True)]:
            with self.subTest(model=model, authorization=authorization, limit=limit), self.assertRaises(ValueError):
                train_model(self.root, self.output, model, authorization=authorization, max_cycles=limit)

    def test_staging_cannot_overwrite_private_runtime_root(self):
        with self.assertRaises(ValueError):
            train_model(self.root, self.root, 'moira2-top1', authorization='authorized')

    def test_completed_artifacts_are_reused_only_if_hashes_match(self):
        directory=self.output/'moira2-top1'
        write_private(directory/'prepared.json', {'synthetic': 'trained checkpoint'})
        write_private(directory/'retained-training-bank.json', {'synthetic': 'bank'})
        value={'status':'ready','model_id':'moira2-top1','protocol':PROTOCOL_VERSION,
            'checkpoint_sha256':file_hash(directory/'prepared.json'),
            'retained_bank_sha256':file_hash(directory/'retained-training-bank.json')}
        write_private(directory/'status.json', value)
        self.assertEqual(train_model(self.root,self.output,'moira2-top1',authorization='authorized'),value)
        write_private(directory/'prepared.json', {'synthetic': 'tampered'})
        with self.assertRaises(RuntimeError):
            train_model(self.root,self.output,'moira2-top1',authorization='authorized')

    def test_ambiguous_incomplete_job_requires_audited_resume(self):
        write_private(self.output/'moira2-top1'/'status.json', {'status':'pending'})
        with self.assertRaises(RuntimeError):
            train_model(self.root,self.output,'moira2-top1',authorization='authorized')

    def test_code_proof_covers_clock_bank_transport_and_native_dependencies(self):
        sources=_code_sources()
        self.assertTrue({'memory_v2.py','moira_v2_native.py','moira_native.py','calendar.py',
            'memory.py','registry.py','pair_lifecycle.py','budget.py'} <= {Path(p).name for p in sources})
        _checked_files(sources)


if __name__=='__main__': unittest.main()
