import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('campaign',
    Path(__file__).parents[1] / 'scripts/run_luwu_p6_training.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class SelectionTest(unittest.TestCase):
    def row(self, speed, free=900, code=0, samples=10):
        return dict(median_steps_s=speed, minimum_free_mib=free,
                    returncode=code, gpu_samples=samples)

    def test_selects_fastest_with_headroom(self):
        expected = self.row(200)
        self.assertEqual(campaign.select_candidate([
            self.row(100), expected, self.row(300, free=500), self.row(400, code=1)]), expected)

    def test_missing_measurements_do_not_pass(self):
        with self.assertRaises(RuntimeError):
            campaign.select_candidate([self.row(100, samples=0), self.row(0)])


if __name__ == '__main__':
    unittest.main()
