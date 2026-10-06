"""Offline diagnostics must not change the commands being evaluated."""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from action_request_probe import ActionRequestProbe


def test_probe_keeps_outputs_and_resets_case_statistics(tmp_path):
    h = onnx.helper
    model = h.make_model(h.make_graph([
        h.make_node('Identity', ['obs'], ['action_unbounded']),
        h.make_node('Clip', ['action_unbounded', 'low', 'high'], ['action']),
    ], 'test', [h.make_tensor_value_info('obs', onnx.TensorProto.FLOAT, [None, 14])],
        [h.make_tensor_value_info('action', onnx.TensorProto.FLOAT, [None, 14])],
        [onnx.numpy_helper.from_array(np.asarray(-.1, np.float32), 'low'),
         onnx.numpy_helper.from_array(np.asarray(.1, np.float32), 'high')]),
        opset_imports=[h.make_opsetid('', 17)], ir_version=10)
    h.set_model_props(model, dict(action_delta_low=json.dumps([-1.]*14),
                                 action_delta_high=json.dumps([1.]*14)))
    path = tmp_path/'policy.onnx'
    onnx.save(model, path)
    original = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    probe = ActionRequestProbe(original, path)
    values = np.full((2, 14), .5, np.float32)
    expected = original.run(None, {'obs': values})
    actual = probe.run(None, {'obs': values})
    assert len(actual) == 1
    np.testing.assert_array_equal(actual[0], expected[0])
    metrics = probe.take_metrics()
    assert metrics['frames'] == 2
    assert metrics['request_gap_squared_sum_mean'] == pytest.approx(14*.4**2)
    assert metrics['latent_range_saturation_fraction'] == 0.
    assert metrics['applied_output_unchanged']
    assert probe.take_metrics() is None
