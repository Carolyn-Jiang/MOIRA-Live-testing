import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from live_testing import repair_first_native_attempt as repair
from live_testing.storage import digest,file_hash,read_json,write_private


TEST_RESERVATION='b'*32


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.folder=self.root/'models'/repair.MODEL;self.obs={'frozen_source':'exact first observation'};fingerprint=digest(self.obs)
        self.bank={'units':[{'synthetic':i} for i in range(29)]}
        self.native={'decisions':{},'feedback':{},'active_pair':None,'history':{'2026-10-13T16:00:00-04:00':{}},
            'direction_memory':[{} for _ in range(29)],'memory_bank':self.bank,
            'attempts':{repair.NATIVE_ID:{'status':'in_progress','input_sha256':fingerprint}}}
        self.runtime={'model_attempts':{repair.MODEL+':'+repair.SOURCE_AT:{'status':'ambiguous','input_sha256':fingerprint}},
            'keep_other_records':{'untouched':True}}
        write_private(self.root/'runtime.json',self.runtime);write_private(self.folder/'live-state.json',self.native)
        write_private(self.root/'book.json',{'decisions':{repair.MODEL:{}},'accounts':{repair.MODEL:{'equity':1.,'history':[]}}})
        write_private(self.folder/'initial-training-bank-promotion.json',{'promoted_snapshot_sha256':digest(self.bank)})
        write_private(self.folder/'prepared.json',{'status':'ready','synthetic':True})
        (self.folder/'initial-training-checkpoint.json').write_bytes((self.folder/'prepared.json').read_bytes())
        marker={'status':'ready','model_id':repair.MODEL,'backbone':'gpt-4.1-mini',
            'checkpoint_sha256':file_hash(self.folder/'prepared.json')}
        write_private(self.folder/'initial-training-ready-before-replay-review.json',marker)
        write_private(self.folder/'initial-training-ready.json',{**marker,'status':'operational_hold'})
        payload={'model':'gpt-4.1-mini','source':'frozen'}
        failure={'model_id':repair.MODEL,'stage':'replay','event_id':repair.NATIVE_ID,'status':'failed',
            'error_type':'RemoteProtocolError','status_code':None,'guard_cost_usd':.148488,'request_sha256':digest(payload)}
        wire={**failure,'phase':'native_pair_selection_exact_candidates','request_payload':payload}
        write_private(self.root/'api-budget.json',{'calls':{TEST_RESERVATION:failure}})
        write_private(self.root/'api-audit/http-requests'/f'{TEST_RESERVATION}.json',wire)
        write_private(self.root/'releases/2026-11-02_09-30_NY.json',{'frozen':True})

    def prepare(self):
        with patch.object(repair,'observation',return_value=self.obs):return repair.prepare(self.root,reservation_id='b'*32)

    def test_only_two_attempts_removed_and_exact_ready_bytes_restored_no_http(self):
        protected=[self.root/'book.json',self.root/'api-budget.json',self.folder/'prepared.json',
            self.root/'api-audit/http-requests'/f'{TEST_RESERVATION}.json']
        before={str(p):p.read_bytes() for p in protected};result=self.prepare()
        self.assertEqual(result['new_http_calls'],0);self.assertTrue(result['failed_reservation_retained'])
        self.assertEqual(read_json(self.folder/'live-state.json'),{**self.native,'attempts':{}})
        self.assertEqual(read_json(self.root/'runtime.json'),{**self.runtime,'model_attempts':{}})
        self.assertEqual((self.folder/'initial-training-ready.json').read_bytes(),
            (self.folder/'initial-training-ready-before-replay-review.json').read_bytes())
        self.assertEqual({str(p):p.read_bytes() for p in protected},before)
        self.assertEqual(read_json(Path(result['archive'])/'proof.json')['status'],'prepared')

    def test_accepted_decision_or_changed_training_bank_rejected(self):
        for kind in ('decision','bank'):
            state=copy.deepcopy(self.native)
            if kind=='decision':state['decisions'][repair.NATIVE_ID]={'accepted':True}
            else:state['memory_bank']['units'].append({'unexpected':True})
            write_private(self.folder/'live-state.json',state)
            with self.assertRaises(ValueError):self.prepare()

    def test_known_http_response_or_response_id_prevents_unknown_transport_repair(self):
        path=self.root/'api-budget.json';value=read_json(path);value['calls'][TEST_RESERVATION]['status_code']=400;write_private(path,value)
        with self.assertRaises(ValueError):self.prepare()
        value['calls'][TEST_RESERVATION]['status_code']=None;value['calls'][TEST_RESERVATION]['response_id']='known';write_private(path,value)
        with self.assertRaises(ValueError):self.prepare()

    def test_write_failure_restores_all_three_exact_bytes(self):
        targets=[self.root/'runtime.json',self.folder/'live-state.json',self.folder/'initial-training-ready.json']
        before={str(p):p.read_bytes() for p in targets};actual=repair.write_private
        def fail(path,value):
            if Path(path)==self.folder/'live-state.json':raise OSError('synthetic write failure')
            return actual(path,value)
        with patch.object(repair,'observation',return_value=self.obs),patch.object(repair,'write_private',side_effect=fail):
            with self.assertRaises(OSError):repair.prepare(self.root,reservation_id='b'*32)
        self.assertEqual({str(p):p.read_bytes() for p in targets},before)


if __name__=='__main__':unittest.main()
