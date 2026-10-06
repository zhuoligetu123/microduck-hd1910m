#!/usr/bin/env python3
"""Export the CPU replay physics for an ARM64 body server without Torch/mjlab."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import mujoco
from replay_hd1910 import load_replay_model, PROFILE_PATH

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--bam-reference', action='store_true')
parser.add_argument('--policy', type=Path, help='Validate and bind the ONNX to this simulation bundle')
args = parser.parse_args()
if args.bam_reference:
    from mjlab_microduck.actuator.cpu_hd1910_bam import PROFILE_PATH, KP_FW
args.output.mkdir(parents=True, exist_ok=True)
model, _, _ = load_replay_model(7.4, bam_reference=args.bam_reference)
policy_binding = {}
if args.policy:
    import onnx
    from replay_hd1910 import validate_metadata
    policy = args.policy.read_bytes()
    metadata = {p.key: p.value for p in onnx.load_model_from_string(policy).metadata_props}
    names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
    validate_metadata(metadata, names, bam_reference=args.bam_reference)
    if float(metadata.get('control_hz', 'nan')) != 50.0:
        raise ValueError('simulation policy must use 50 Hz')
    policy_binding = dict(policy_file='policy.onnx', policy_sha256=hashlib.sha256(policy).hexdigest(),
                          policy_task_id=metadata['task_id'],
                          installation_sha256=metadata.get('installation_sha256'))
    (args.output/'policy.onnx').write_bytes(policy)
path = args.output / 'hd1910.mjb'
mujoco.mj_saveModel(model, str(path))
shutil.copyfile(PROFILE_PATH, args.output/'motor_calibration.json')
(args.output / 'physics.json').write_text(json.dumps({
    'bundle_schema': 2, 'profile_file': 'motor_calibration.json', **policy_binding,
    'mujoco': mujoco.__version__, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'profile_sha256': hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
    'physics_hz': 200, 'delay_steps': 4, 'voltage': 7.4,
    **({'kp_fw': KP_FW, 'kd_fw': 20, 'kd_modelled': False} if args.bam_reference else {}),
    'actuator_backend': 'hd1910_bam_m6' if args.bam_reference else 'hd1910_reference_pd',
    'calibrated': False, 'hardware_tested': False,
}, indent=2) + '\n')
print(path)
