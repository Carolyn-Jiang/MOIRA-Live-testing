from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from live_testing.storage import append_private


class StorageTests(unittest.TestCase):
    def test_four_concurrent_appenders_preserve_every_large_json_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'logs'/'model-calls.jsonl'
            text='Synthetic untrusted evidence 文本\n'*2000
            def write(worker):
                for step in range(20):append_private(path,{'worker':worker,'step':step,'text':text})
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(write,range(4)))
            lines=path.read_text().splitlines();self.assertEqual(len(lines),80)
            records=[json.loads(line) for line in lines]
            self.assertEqual({(row['worker'],row['step']) for row in records},{(worker,step) for worker in range(4) for step in range(20)})
            self.assertTrue(all(row['text']==text for row in records));self.assertEqual(path.stat().st_mode&0o777,0o600)
            self.assertEqual(path.with_name(path.name+'.lock').stat().st_mode&0o777,0o600)

    def test_invalid_json_never_appends_a_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.jsonl';append_private(path,{'valid':True});before=path.read_bytes()
            with self.assertRaises(ValueError):append_private(path,{'invalid':float('nan')})
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
