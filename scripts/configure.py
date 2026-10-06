#!/usr/bin/env python3
"""Create portable configs. Never changes servo EEPROM or enables torque."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'radxa/references/reference_runtime_20261005'


def configure(output, calibration=None, port='/dev/hd1910-servo', motion=False, sim_port=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((MODELS / 'manifest.json').read_text())
    for role, meta in manifest['models'].items():
        path = MODELS / f'hd1910_{role}.onnx'
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError(f'Model hash mismatch: {role}')
    text = (ROOT / 'radxa/reference_native.toml').read_text()
    text = text.replace('/home/robot/workspace/huggingface/radxa/references/reference_runtime_20261005', str(MODELS))
    old_bus = 'feetech:/home/robot/workspace/huggingface/radxa/reference_native.json'
    if sim_port is not None:
        text = text.replace(old_bus, f'sim:127.0.0.1:{sim_port}')
        shutil.copy2(MODELS / 'hd1910_walk.onnx', output / 'policy.onnx')
        text = text.replace(str(MODELS / 'hd1910_walk.onnx'), str(output / 'policy.onnx'))
    else:
        calibration = (calibration or ROOT / 'radxa/installation.json').resolve()
        if not calibration.is_file():
            raise ValueError('Installation calibration file missing')
        values = json.loads(calibration.read_text())
        if motion and not (values.get('calibration_verified') and values.get('imu_pose_verified')):
            raise ValueError('Use the actual verified device calibration; example is read-only')
        text = text.replace(old_bus, 'feetech:' + str(output / 'feetech.json'))
        config = dict(port=port, installation=str(calibration), imu_bus='/dev/i2c-4',
                      imu_address=75, allow_motion=motion, servo_gain_profile='reference_runtime',
                      scheduled_bus=True, reference_native=True)
        (output / 'feetech.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'params.toml').write_text(text)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'local')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--port', default='/dev/hd1910-servo')
    parser.add_argument('--allow-motion', action='store_true')
    args = parser.parse_args()
    print(configure(args.output, args.calibration, args.port, args.allow_motion))
