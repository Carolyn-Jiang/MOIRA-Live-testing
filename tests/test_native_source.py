import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from live_testing.native_source import NATIVE_SHA,verify_native_root


class NativeSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'framework';self.root.mkdir();(self.root/'llm').mkdir()
        self.code=self.root/'llm/policy_agent.py';self.code.write_text('# Synthetic offline native source\n')
        self.lock=Path(self.temp.name)/'lock.json';self.metadata={'schema':'moira.live-testing.native-framework-lock/1',
            'repository':'https://example.invalid/native.git','commit':NATIVE_SHA,
            'files_sha256':{'llm/policy_agent.py':hashlib.sha256(self.code.read_bytes()).hexdigest()}}
        self.lock.write_text(json.dumps(self.metadata))
    def verify(self):return verify_native_root(self.root,lock_path=self.lock)
    def test_clean_exact_commit_and_file_bytes_are_verified_without_fetch(self):
        with patch('live_testing.native_source.subprocess.check_output',side_effect=[NATIVE_SHA+'\n','']) as git:
            result=self.verify()
        self.assertEqual(result['verified_files'],1);self.assertEqual(result['commit'],NATIVE_SHA)
        self.assertTrue(all('fetch' not in call.args[0] for call in git.call_args_list))
    def test_wrong_commit_dirty_checkout_or_changed_bytes_fail_closed(self):
        for responses in [('0'*40,''),(NATIVE_SHA,' M llm/policy_agent.py')]:
            with patch('live_testing.native_source.subprocess.check_output',side_effect=responses):
                with self.assertRaises(RuntimeError):self.verify()
        self.code.write_text('# Changed source\n')
        with patch('live_testing.native_source.subprocess.check_output',side_effect=[NATIVE_SHA,'']):
            with self.assertRaisesRegex(RuntimeError,'source/configuration'):self.verify()
    def test_invalid_lock_paths_cannot_read_outside_framework(self):
        self.metadata['files_sha256']={'../outside.json':'a'*64};self.lock.write_text(json.dumps(self.metadata))
        with patch('live_testing.native_source.subprocess.check_output',side_effect=[NATIVE_SHA,'']):
            with self.assertRaises(RuntimeError):self.verify()


if __name__=='__main__':unittest.main()
