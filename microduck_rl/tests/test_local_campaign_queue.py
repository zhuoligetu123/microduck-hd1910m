import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('local_campaign', scripts/'run_local_head_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class PredecessorTest(unittest.TestCase):
    def test_only_finished_evaluation_releases_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(campaign.predecessor_ready(root))
            for stage in ('training', 'evaluation', 'head_center', 'pilot_evaluated'):
                (root/'status.json').write_text(json.dumps(dict(stage=stage)))
                self.assertFalse(campaign.predecessor_ready(root))
            (root/'status.json').write_text(json.dumps(dict(stage='evaluated_not_hardware_qualified')))
            self.assertTrue(campaign.predecessor_ready(root))

    def test_partial_status_write_does_not_release_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'status.json').write_text('{')
            self.assertFalse(campaign.predecessor_ready(root))

    def test_failure_does_not_silently_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'failure.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                campaign.predecessor_ready(root)


if __name__ == '__main__':
    unittest.main()
