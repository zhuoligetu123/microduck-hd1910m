import importlib.util
import json
import os
from pathlib import Path
import unittest
import numpy as np
import mujoco

spec = importlib.util.spec_from_file_location('luwu_direct', Path(__file__).parents[1]/'scripts/evaluate_luwu_direct.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DirectReplayTests(unittest.TestCase):
    def test_raw_history_layout(self):
        obs = module.build_observation(np.ones(3), -np.ones(3), np.ones(14)*.4,
                                       np.ones(14)*.2, np.ones(14)*.3,
                                       np.arange(14), [.1, 0, .4])
        self.assertEqual(obs.shape, (61,))
        np.testing.assert_allclose(obs[6:20], .1)
        np.testing.assert_equal(obs[34:48], np.arange(14))
        np.testing.assert_equal(obs[51:], np.zeros(10))

    def test_swings_exclude_initial_and_incomplete_airtime(self):
        samples = [(0, .05, False), (.02, 0, True), (.04, .004, False),
                   (.06, .01, False), (.08, 0, True), (.1, .05, False)]
        result = module.summarize_swings(samples)
        self.assertEqual(result['complete_swings'], 1)
        self.assertEqual(result['peak_mm_median'], 10.)
        numeric = module.summarize_swings(np.asarray(samples))
        json.dumps(numeric, allow_nan=False)

    def test_no_swing_is_not_zero_height_gait(self):
        result = module.summarize_swings([(0, 0, True), (.02, 0, True)])
        self.assertIsNone(result['peak_mm_median'])
        self.assertEqual(result['complete_swings'], 0)

    def test_start_stop_schedule(self):
        np.testing.assert_equal(module.command_at(1.98, [.3, 0, 0], 22), [0, 0, 0])
        np.testing.assert_equal(module.command_at(2, [.3, 0, 0], 22), [.3, 0, 0])
        np.testing.assert_equal(module.command_at(18, [.3, 0, 0], 22), [0, 0, 0])

    def test_recovery_requires_continuous_dwell(self):
        self.assertIsNone(module.recovery_delay([0, .2, .4, .6], [0, 0, 20, 0], 0, 1, 15))
        self.assertEqual(module.recovery_delay([0, .2, .4, .6], [0, 0, 0, 0], 0, 1, 15), 0.)

    @unittest.skipUnless(os.environ.get('LUWU_SOURCE'), 'set LUWU_SOURCE for geometry integration')
    def test_bam_uses_torque_actuators_and_original_mass(self):
        model, data, motor = module.make_model(Path(os.environ['LUWU_SOURCE']), 5, 7.4, 4)
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1.)
        np.testing.assert_equal(model.actuator_biastype, mujoco.mjtBias.mjBIAS_NONE)
        self.assertAlmostEqual(float(model.body_mass.sum()), .8, places=6)
        self.assertEqual(len(motor.qids), 14)

    @unittest.skipUnless(os.environ.get('LUWU_RUNTIME'), 'set LUWU_RUNTIME for reference math parity')
    def test_upstream_runtime_observation_math(self):
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules):
            path = Path(os.environ['LUWU_RUNTIME'])/'python/rl_core.py'
            upstream_spec = importlib.util.spec_from_file_location('upstream_rl_core', path)
            upstream = importlib.util.module_from_spec(upstream_spec)
            sys.modules[upstream_spec.name] = upstream
            upstream_spec.loader.exec_module(upstream)
            rng = np.random.default_rng(31)
            q, dq, action = rng.normal(size=(3, 14)).astype(np.float32)
            gyro, gravity, command = rng.normal(size=(3, 3)).astype(np.float32)
            angles, speeds = np.zeros(15), np.zeros(15)
            angles[upstream.POLICY_INDEX] = np.degrees(q)
            speeds[upstream.POLICY_INDEX] = np.degrees(dq)
            state = upstream.State([0]*3, [0]*3, angles, speeds, gravity, gyro)
            expected = upstream.build_obs(state, action, command, np.zeros(4))
            actual = module.build_observation(gyro, gravity, q, dq,
                                              np.radians(upstream.HOME_DEG), action, command)
            np.testing.assert_allclose(actual, expected, atol=5e-7)


if __name__ == '__main__':
    unittest.main()
