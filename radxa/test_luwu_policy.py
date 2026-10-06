import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from luwu_policy import (LocalCalibration, LuwuPolicy, LuwuSuite, JOINTS,
                         RAD_TICK, RAD_SPEED, rotate, conjugate, BodyFeedbackFilter)


class LuwuContractTest(unittest.TestCase):
    def setUp(self):
        self.cal = LocalCalibration()

    def test_mapping_and_independent_mouth(self):
        ticks = self.cal.target_ticks(np.zeros(14), .2)
        self.assertEqual([self.cal.joints[n]['id'] for n in JOINTS],
                         [10,9,8,7,6,12,11,14,13,5,4,3,2,1])
        self.assertEqual(ticks[15], round(2048-.2/RAD_TICK))
        self.assertEqual(ticks[1], 2048)
        self.assertTrue(self.cal.unchanged())

    def test_encoder_roundtrip_nonuniform_zeros(self):
        config = self.cal.config.copy()
        config['joints'] = [dict(row, zero_ticks=1800+i*20)
                            for i, row in enumerate(reversed(config['joints']))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'installation.json'
            path.write_text(json.dumps(config))
            cal = LocalCalibration(path)
            rng = np.random.default_rng(2026)
            for _ in range(100):
                q, dq = rng.uniform(-1,1,14), rng.uniform(-3,3,14)
                ticks, speed = cal.simulate_encoders(q, dq)
                np.testing.assert_allclose(cal.positions(ticks), q, atol=RAD_TICK/2)
                np.testing.assert_allclose(cal.velocities(speed), dq, atol=RAD_SPEED/2)
            self.assertTrue(cal.unchanged())

    def test_imu_mount_once(self):
        for axis in np.eye(3):
            world_body = np.r_[math.cos(.4), axis*math.sin(.4)]
            gyro = [.2,-.3,.4]
            raw_q, raw_gyro = self.cal.simulate_imu(world_body, gyro)
            body_gyro, gravity = self.cal.raw_imu_to_body(raw_q, raw_gyro)
            np.testing.assert_allclose(body_gyro, gyro, atol=1e-14)
            np.testing.assert_allclose(gravity, rotate(conjugate(world_body), [0,0,-1]), atol=1e-14)

    def test_metadata_home_not_encoder_zero(self):
        walk, roll = LuwuPolicy('walk'), LuwuPolicy('roulade')
        self.assertAlmostEqual(walk.home[2], -.349)
        self.assertAlmostEqual(roll.home[2], -.419)
        for policy in (walk, roll):
            obs, _ = policy.observation(policy.home, np.zeros(14), [0,0,0], [0,0,-1], 0)
            np.testing.assert_allclose(obs[6:20], 0)
        self.assertTrue(self.cal.unchanged())

    def test_raw_history_and_ema(self):
        for role in ('walk','getup','pick','roulade'):
            p = LuwuPolicy(role)
            args = (p.home, np.zeros(14), [0,0,0], [0,0,-1], 0)
            obs, _ = p.observation(*args)
            raw = p.session.run(None, {p.session.get_inputs()[0].name: obs[None]})[0][0]
            target, _, _ = p.infer(*args)
            np.testing.assert_allclose(target, p.home+(1-p.alpha)*raw*p.scale, atol=1e-7)
            next_obs, _ = p.observation(*args)
            np.testing.assert_array_equal(next_obs[34:48], raw)
            p.reset()
            np.testing.assert_array_equal(p.previous, 0)

    def test_pick_phase_mouth_and_getup(self):
        p = LuwuPolicy('pick')
        cmd, head, mouth = p.command(1, [99]*3, [99]*4)
        np.testing.assert_allclose(cmd, [0,1,0], atol=1e-15)
        self.assertAlmostEqual(mouth, math.pi/6)
        self.assertEqual(p.command(1.6)[2], 0)
        np.testing.assert_array_equal(LuwuPolicy('getup').command(0, [1]*3)[0], 0)

    def test_task_switch_resets_and_unavailable_fails(self):
        suite = LuwuSuite()
        suite.select('pick', 2)
        self.assertEqual(suite.advance(5.99)[0].role, 'pick')
        self.assertEqual(suite.advance(6)[0].role, 'walk')
        suite.select('roulade', 7)
        self.assertEqual(suite.advance(8.9)[0].role, 'walk')
        with self.assertRaises(ValueError):
            suite.select('sitstand', 9)
        with self.assertRaises(ValueError):
            suite.advance(0)

    def test_invalid_inputs_never_wrap_or_rezero(self):
        with self.assertRaises(ValueError):
            self.cal.target_ticks(np.full(14, 10.))
        with self.assertRaises(ValueError):
            self.cal.target_ticks(np.full(14, float('nan')))
        p = LuwuPolicy('walk')
        with self.assertRaises(ValueError):
            p.observation(p.home, np.zeros(14), [0,0,0], [0,0,-9.81], 0)

    def test_explicit_upstream_saturation_preserves_ids_and_zeros(self):
        q = np.zeros(14)
        q[5] = -4.
        with self.assertRaises(ValueError):
            self.cal.target_ticks(q)
        clipped = self.cal.target_ticks(q, saturate=True)
        self.assertEqual(clipped[12], 0)
        self.assertEqual(clipped[1], 2048)
        self.assertTrue(self.cal.unchanged())

    def test_mount_has_analytic_nominal_axis_convention(self):
        mount = [.5,-.5,.5,-.5]
        np.testing.assert_allclose(rotate(mount, [1,2,3]), [3,-1,-2])
        np.testing.assert_allclose(rotate(conjugate([math.sqrt(.5),0,math.sqrt(.5),0]),
                                          [0,0,-1]), [1,0,0], atol=1e-15)

    def test_getup_returns_only_after_continuous_upright(self):
        s = LuwuSuite()
        s.select('getup',0)
        for i in range(40):
            self.assertFalse(s.recovery_ready([0,0,-1], i*.02))
        self.assertFalse(s.recovery_ready([1,0,0], 1))
        for i in range(49):
            self.assertFalse(s.recovery_ready([0,0,-1], 2+i*.02))
        self.assertTrue(s.recovery_ready([0,0,-1], 3))
        self.assertEqual(s.role,'walk')

    def test_filter_timing_and_gap(self):
        f = BodyFeedbackFilter()
        f.update(0,np.zeros(3),np.zeros(14))
        gyro,vel = f.update(.01,np.ones(3),np.ones(14))
        np.testing.assert_allclose(gyro,.5)
        np.testing.assert_allclose(vel,.6)
        gyro,vel = f.update(.03,np.ones(3),np.ones(14))
        np.testing.assert_allclose(gyro,.875)
        np.testing.assert_allclose(vel,.936)
        gyro,vel = f.update(1,np.ones(3)*2,np.ones(14)*2)
        np.testing.assert_allclose(gyro,2)
        with self.assertRaises(ValueError):
            f.update(.5,np.zeros(3),np.zeros(14))


if __name__ == '__main__':
    unittest.main()
