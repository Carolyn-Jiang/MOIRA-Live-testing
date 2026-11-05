import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from live_testing.finmem_native import constrained_live_call
from live_testing.storage import write_private
from live_testing.registry import TICKERS


class PrivateAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.path=self.root/'controls/call.py';self.path.parent.mkdir()
        self.path.write_text('def constrained_live_call(endpoint,prompt,schema,**kwargs):\n    return endpoint(prompt,schema)\n')
        self.config={'finmem_action_adapter':{'path':str(self.path),
            'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}}
        write_private(self.root/'config.json',self.config)
        self.pair=list(TICKERS[:2])
        self.observation={'timestamp':'2026-11-02T09:30:00-05:00',
            'prices':{t:100. for t in TICKERS},'news':{t:[] for t in TICKERS}}
        write_private(self.root/'company-context.json',{'generated_at':'2026-11-01T12:00:00-05:00',
            'as_of':'2026-11-01T11:00:00-05:00',
            'companies':{t:{'recent_operations_summary':'Synthetic known prior operations.'} for t in TICKERS}})
    def tearDown(self):self.temp.cleanup()
    def call(self,endpoint):
        return constrained_live_call(endpoint,'source',{},state={'selected_pair':self.pair},observation=self.observation,
            failure_type=RuntimeError,directory=self.root,holder={'budget':SimpleNamespace(root=self.root)})
    def test_actual_endpoint_is_called_only_through_verified_external_adapter(self):
        calls=[]
        self.assertEqual(self.call(lambda *args:calls.append(args) or 'native_response'),'native_response')
        self.assertEqual(calls,[('source',{})])
    def test_tampered_or_outside_runtime_adapter_never_reaches_endpoint(self):
        def forbidden(*args):raise AssertionError('Provider must not be called')
        self.path.write_text(self.path.read_text()+'# altered\n')
        with self.assertRaises(RuntimeError):self.call(forbidden)
        self.config['finmem_action_adapter']['path']=str(self.root.parent/'elsewhere.py')
        write_private(self.root/'config.json',self.config)
        with self.assertRaises(RuntimeError):self.call(forbidden)


if __name__=='__main__':unittest.main()
