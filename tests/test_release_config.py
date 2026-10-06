import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from configure import configure


class ReleaseConfigTest(unittest.TestCase):
    def test_readonly_default_and_calibration_unchanged(self):
        path = ROOT / 'radxa/installation.json'
        before = path.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            result = configure(Path(folder), port='/dev/example-servo')
            io = json.loads((result / 'feetech.json').read_text())
            self.assertFalse(io['allow_motion'])
            self.assertTrue(io['reference_native'])
            self.assertTrue(io['scheduled_bus'])
            self.assertEqual(io['port'], '/dev/example-servo')
            self.assertEqual(Path(io['installation']), path)
            self.assertNotIn('/home/robot/workspace/huggingface', (result / 'params.toml').read_text())
        self.assertEqual(before, path.read_bytes())

    def test_unverified_example_cannot_enable(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                configure(Path(folder), motion=True)

    def test_sim_uses_no_physical_bus(self):
        with tempfile.TemporaryDirectory() as folder:
            result = configure(Path(folder), sim_port=17803)
            text = (result / 'params.toml').read_text()
            self.assertIn('sim:127.0.0.1:17803', text)
            self.assertNotIn('feetech:', text)
            self.assertTrue((result / 'policy.onnx').is_file())


if __name__ == '__main__':
    unittest.main()
