import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from live_testing.retrain_original_sharpe import fresh_checkpoint, train


class RetrainOriginalSharpeTests(unittest.TestCase):
    def legacy(self):
        return {'model':'gpt-4.1-mini','status':'ready','pair_memory':[{'quality':'old'}],
            'prompt_memory':{'base_prompt':'Default causal direction prompt.',
                'active_prompt':'Default causal direction prompt.\nCR-selected lesson',
                'accepted_lessons':[{'lesson_text':'CR-selected lesson'}],
                'rejected_lessons':[{'lesson_text':'rejected'}],'version':12}}

    def test_fresh_training_removes_cr_selected_beliefs_and_pair_memory(self):
        old=self.legacy(); before=copy.deepcopy(old); fresh=fresh_checkpoint(old)
        self.assertEqual(old,before)
        self.assertEqual(fresh['prompt_memory']['active_prompt'],old['prompt_memory']['base_prompt'])
        self.assertEqual(fresh['pair_memory'],[])
        self.assertEqual(fresh['prompt_memory']['accepted_lessons'],[])
        self.assertEqual(fresh['prompt_memory']['rejected_lessons'],[])
        self.assertEqual(fresh['frozen_prompt_sha256'],
            hashlib.sha256(old['prompt_memory']['base_prompt'].encode()).hexdigest())

    def test_wrong_backbone_unready_or_empty_default_cannot_train(self):
        for field,value in [('model','other'),('status','pending')]:
            old=self.legacy();old[field]=value
            with self.assertRaises(ValueError):fresh_checkpoint(old)
        old=self.legacy();old['prompt_memory']['base_prompt']=' '
        with self.assertRaises(ValueError):fresh_checkpoint(old)

    def test_authorization_and_eight_cycle_cap_checked_before_any_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'staged'
            for authorization,max_cycles in [('',8),('explicit',0),('explicit',9),('explicit',True)]:
                with self.assertRaises(ValueError):
                    train(root,out,authorization=authorization,max_cycles=max_cycles)
                self.assertFalse(out.exists())

    def test_staged_training_cannot_target_live_root_or_external_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'runtime'
            for output in [root,Path(folder)/'external']:
                with self.assertRaises(ValueError):train(root,output,authorization='explicit')
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
