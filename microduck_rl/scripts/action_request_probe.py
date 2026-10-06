"""Observe latent versus applied requests without modifying the policy output."""
import json

import numpy as np
import onnx
import onnxruntime as ort


class ActionRequestProbe:
    def __init__(self, session, path):
        self.session = session
        model = onnx.load(path)
        bounded_name = model.graph.output[0].name
        latent_name = bounded_name + '_unbounded'
        names = {name for node in model.graph.node for name in node.output}
        if latent_name not in names:
            raise ValueError('request diagnostics require a bounded exported policy')
        model.graph.output.append(onnx.helper.make_tensor_value_info(
            latent_name, onnx.TensorProto.FLOAT, [None, 14]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.diagnostic = ort.InferenceSession(model.SerializeToString(), options,
                                               providers=['CPUExecutionProvider'])
        metadata = session.get_modelmeta().custom_metadata_map
        self.low = np.asarray(json.loads(metadata['action_delta_low']))
        self.high = np.asarray(json.loads(metadata['action_delta_high']))
        self.samples = []

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, output_names, inputs, *args, **kwargs):
        original = self.session.run(output_names, inputs, *args, **kwargs)
        bounded, latent = self.diagnostic.run(None, inputs)
        if not np.allclose(original[0], bounded, atol=1e-6, rtol=1e-6):
            raise ValueError('diagnostic graph changed the policy output')
        self.samples.append((latent.copy(), bounded.copy()))
        return original

    def take_metrics(self):
        if not self.samples:
            return None
        latent = np.concatenate([row[0] for row in self.samples])
        applied = np.concatenate([row[1] for row in self.samples])
        self.samples.clear()
        bounded = np.clip(latent, self.low, self.high)
        return dict(frames=len(latent),
            latent_range_saturation_fraction=float(np.mean(latent != bounded)),
            latent_abs_p95_rad=float(np.quantile(np.abs(latent), .95)),
            request_gap_squared_sum_mean=float(np.square(latent-applied).sum(axis=1).mean()),
            slew_gap_squared_sum_mean=float(np.square(bounded-applied).sum(axis=1).mean()),
            latent_step_squared_sum_mean=(float(np.square(np.diff(latent, axis=0)).sum(axis=1).mean())
                                          if len(latent) > 1 else None),
            applied_step_squared_sum_mean=(float(np.square(np.diff(applied, axis=0)).sum(axis=1).mean())
                                           if len(applied) > 1 else None),
            request_gap_rms_per_joint_rad=np.sqrt(np.square(latent-applied).mean(axis=0)).tolist(),
            applied_output_unchanged=True)
