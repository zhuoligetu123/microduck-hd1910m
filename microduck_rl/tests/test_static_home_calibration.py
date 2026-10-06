import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/calibrate_static_home.py'
SPEC = importlib.util.spec_from_file_location('static_home', SCRIPT)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def capture(tmp_path, change=None):
    rows = []
    for i in range(110):
        joints = [dict(name=name, id=k+1, direction=1, zero_ticks=2048)
                  for k, name in enumerate(calibration.JOINTS)]
        states = [dict(servo_id=k+1, position_ticks=2049, valid=True, status=0,
                       torque_enabled=1, current_a=.1, voltage_v=6.6) for k in range(15)]
        row = dict(t=i*.1, policy='held', targets=[0.]*15, feedback=dict(
            sequence=i, policy_enabled=False, homed=True, control_valid=True,
            imu_valid=True, error=None, servo_gains_verified=True, servo_gain_profile='reference_runtime',
            joint_age_s=.02, imu_age_s=.01, joints=joints, states=states,
            positions=[calibration.RAD_PER_TICK]*15,
            imu=dict(gravity=[0, 0, -1], gyro=[0, 0, 0])))
        if change:
            change(row)
        rows.append(row)
    path = tmp_path/'capture.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows+[rows[-1]]))
    return path


def test_reference_deduplicates_and_does_not_apply_loaded_error(tmp_path):
    result = calibration.extract_reference(capture(tmp_path))
    assert result['frames'] == 111
    assert result['unique_snapshots'] == 110
    assert len(result['policy_joint_names']) == 14
    assert result['joints'][0]['measured_minus_target_rad'] == calibration.RAD_PER_TICK
    assert result['applied_encoder_offset_rad'] == [0.]*15
    assert result['applied_bam_parameter_changes'] == {}
    assert result['frame_median_voltage_v']['p50'] == 6.6


@pytest.mark.parametrize('fault', ['tilted', 'rl', 'stale', 'mapping', 'encoder', 'alarm', 'moving', 'restart'])
def test_invalid_calibration_rejected(tmp_path, fault):
    def change(row):
        f = row['feedback']
        if fault == 'tilted':
            f['imu']['gravity'] = [1, 0, 0]
        elif fault == 'rl':
            f['policy_enabled'] = True
        elif fault == 'stale':
            f['joint_age_s'] = .2
        elif fault == 'mapping':
            f['joints'][0]['id'] = 2
        elif fault == 'encoder':
            f['positions'][0] += .1
        elif fault == 'alarm':
            f['states'][0]['status'] = 2
        elif fault == 'moving':
            row['targets'][0] = row['t']*.1
        elif fault == 'restart' and row['t'] > 5:
            row['t'] -= 5
    with pytest.raises(ValueError):
        calibration.extract_reference(capture(tmp_path, change))
