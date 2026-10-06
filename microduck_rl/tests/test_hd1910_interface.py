"""No motor: ONNX contract and the dedicated hardware CLI's dependency boundary."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'feetech_hls'))
from feetech_hls.rl_test import TestbenchPolicy, run
from test_hls_bus import ServoPeer


class PolicyTests(unittest.TestCase):
    def test_native_config_keeps_training_contract_and_remains_disabled(self):
        config = tomllib.loads((ROOT/'radxa/native_feetech.toml').read_text())['policy']
        self.assertFalse(config['enabled'])
        for key in ('action_scale','standing_action_scale','standing_gain_ratio',
                    'head_lowpass','legs_lowpass'):
            self.assertEqual(config[key], 1.0)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'contract_fixture.onnx'

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, inputs=4, outputs=1, nan=False):
        if nan:
            nodes = [helper.make_node('Constant', [], ['actions'],
                     value=helper.make_tensor('v', TensorProto.FLOAT, [1, 1], [math.nan]))]
            tensors = []
        else:
            tensors = [helper.make_tensor('indices', TensorProto.INT64, [outputs],
                       [3] if outputs == 1 else list(range(outputs)))]
            nodes = [helper.make_node('Gather', ['obs', 'indices'], ['actions'], axis=1)]
        graph = helper.make_graph(nodes, 'fixture_not_trained',
                [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, inputs])],
                [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, outputs])], tensors)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=10)
        onnx.checker.check_model(model)
        onnx.save(model, self.path)

    def test_observation_and_action_contract(self):
        self.model()
        policy = TestbenchPolicy(self.path, scale=.5, home=.1)
        state = SimpleNamespace(position_rad=.3, velocity_rad_s=-.2)
        goal, action = policy.step(state, .6)
        self.assertAlmostEqual(action, .6, places=6)
        self.assertAlmostEqual(goal, .4, places=6)
        self.assertEqual(policy.previous_action, action)
        np.testing.assert_allclose(policy.session.get_inputs()[0].shape, [1, 4])

    def test_full_body_policy_rejected_before_bus_open(self):
        self.model(61, 14)
        args = SimpleNamespace(onnx=self.path, action_scale=1., home_rad=0.)
        with self.assertRaisesRegex(ValueError, 'not a full-body policy'):
            run(args, None)

    def test_nan_output_rejected(self):
        self.model(nan=True)
        policy = TestbenchPolicy(self.path)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            policy.step(SimpleNamespace(position_rad=0., velocity_rad_s=0.), .2)

    def test_cli_help_without_training_imports(self):
        result = subprocess.run([sys.executable, str(ROOT / 'microduck_rl/scripts/testbench_hd1910.py'),
                                 '--help'], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--confirm-unloaded', result.stdout)
        self.assertIn('--watchdog-test', result.stdout)

    def test_onnx_fixture_full_rollout_over_pty(self):
        self.model()
        peer = ServoPeer()
        events = []
        def emit(event, **data):
            events.append(dict(event=event, **data))
        try:
            args = SimpleNamespace(onnx=self.path, action_scale=1., home_rad=0.,
                port=peer.path, id=1, zero_ticks=1000, direction=1, min_deg=-20,
                max_deg=20, duration=1., excursion_deg=5., motion=True,
                confirm_unloaded=True, watchdog_test=True)
            result = run(args, emit)
            self.assertEqual(result['samples'], 50)
            self.assertTrue(result['torque_off'])
            samples = [e for e in events if e['event'] == 'sample']
            self.assertEqual(len(samples[0]['observation']), 4)
            self.assertAlmostEqual(samples[1]['observation'][2], samples[0]['action'])
            self.assertTrue(any(e['event'] == 'watchdog' for e in events))
        finally:
            peer.close()


if __name__ == '__main__':
    unittest.main()
