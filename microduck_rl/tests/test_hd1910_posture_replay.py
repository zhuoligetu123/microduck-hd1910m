"""Settled posture acceptance must not confuse sitting with walking velocity."""
import importlib.util
import hashlib
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    'posture_replay', Path(__file__).parents[1] / 'scripts/replay_hd1910_task.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_straight_path_uses_segment_frame_and_unwraps_yaw():
    import math
    sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
    from replay_hd1910 import straight_path_metrics
    assert straight_path_metrics([]) is None
    result = straight_path_metrics([[0,0,math.pi/2], [0,1,math.pi/2]])
    assert result['max_cross_track_m'] < 1e-10
    assert result['final_heading_error_deg'] == 0.
    result = straight_path_metrics([[0,0,math.radians(179)], [0,0,math.radians(-179)]])
    assert result['final_heading_error_deg'] == pytest.approx(2.)


def test_velocity_pass_cannot_hide_circling_or_low_right_foot():
    sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
    from replay_hd1910 import straight_gait_check
    row = dict(steps=1000, completed=True, no_fall=True, first_head_floor_contact_s=None,
        baseline_check_passed=True, prefall_feet={'swing_peak_median_mm': [20.,20.]},
        prefall_straight_path={'max_heading_error_deg':10., 'max_cross_track_m':.1})
    assert straight_gait_check(row)
    row['prefall_feet']['swing_peak_median_mm'][1] = 10.
    assert not straight_gait_check(row)
    row['prefall_feet']['swing_peak_median_mm'][1] = 20.
    row['prefall_straight_path']['max_heading_error_deg'] = 180.
    assert not straight_gait_check(row)
    row['prefall_straight_path'] = None
    assert straight_gait_check(row) is None


def test_sit_requires_lower_height():
    standing = [(.115, 2., .001)] * 100
    assert replay.posture_phase_metrics(standing, .115)['passed']
    assert not replay.posture_phase_metrics(standing, .060)['passed']
    assert replay.posture_phase_metrics([(.060, 2., .001)] * 100, .060)['passed']


def test_fall_or_unsettled_motion_is_not_a_pass():
    assert not replay.posture_phase_metrics([(.060, 65., .001)] * 100, .060)['passed']
    assert not replay.posture_phase_metrics([(.060, 2., .1)] * 100, .060)['passed']
    assert not replay.posture_phase_metrics([], .060)['passed']


def test_transient_target_crossing_is_not_arrival():
    trace = [(.060, 0., 0.)] * 10 + [(.115, 0., 0.)] * 90
    assert not replay.posture_phase_metrics(trace, .060)['passed']


@pytest.mark.parametrize('field,bad',[
    ('task_id','Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Slew-Refine'),
    ('command_semantics','body_velocity'),('stand_flag','1'),('sit_flag','0'),
    ('posture_ramp_s','0'),('previous_action_semantics','raw_home_delta'),
    ('policy_period_s','0.01'),('default_joint_pos',','.join(['0']*14)),
])
def test_posture_requires_explicit_flag_and_history_contract(field,bad):
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from replay_hd1910 import validate_metadata,DEFAULT_POSE,PROFILE_PATH
    names=[f'joint_{i}' for i in range(14)]
    meta=dict(task_id='Mjlab-SitStand-Flat-MicroDuck-HD1910-Reference-Slew-Balanced',
              command_semantics='sit_flag_zero_zero',stand_flag='0',sit_flag='1',
              posture_ramp_s='2.0',calibration_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
              joint_names=','.join(names),action_scale='1.0',policy_period_s='0.02',
              action_semantics='bounded_slew_home_delta_v2',
              previous_action_semantics='bounded_slew_home_delta_v2',max_action_step_rad='0.1',
              default_joint_pos=','.join(f'{v:.3f}' for v in DEFAULT_POSE),
              observation_names='base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command')
    validate_metadata(meta,names,posture=True)
    with pytest.raises(ValueError):
        validate_metadata(meta,names)
    meta[field]=bad
    with pytest.raises(ValueError):
        validate_metadata(meta,names,posture=True)
