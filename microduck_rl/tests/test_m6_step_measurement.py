import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

scripts = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('step_replay', scripts/'replay_m6_step.py')
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def test_contact_jitter_is_not_stepping():
    assert step.lift_events([[.0001, -.0001], [-.0001, .0001]] * 100) == ([0, 0], 0)


def test_single_leg_motion_does_not_count_as_alternating():
    cycle = [[0, 0]] * 3 + [[.02, 0]] * 5 + [[0, 0]] * 3
    assert step.lift_events(cycle * 4) == ([4, 0], 0)


def test_both_legs_need_sustained_clearance():
    rows = [[0, 0]] * 3 + [[.02, 0]] * 5 + [[0, 0]] * 3 + [[0, .02]] * 5
    assert step.lift_events(rows) == ([1, 1], 1)
    assert step.lift_events([[0, 0], [.02, 0], [0, 0]]) == ([0, 0], 0)


def test_joint_age_replays_one_coherent_position_velocity_snapshot():
    policy = step.ReplayPolicy.__new__(step.ReplayPolicy)
    policy.data = SimpleNamespace(qpos=np.ones(14), qvel=np.full(14, 10.))
    policy.joint_qpos_indices = np.arange(14)
    policy.joint_qvel_indices = np.arange(14)
    policy.default_pose = np.zeros(14)
    policy.set_joint_observation_delay(2)
    for frame, expected in ((1, 1), (2, 1), (3, 1), (4, 2)):
        policy.data.qpos[:] = frame
        policy.data.qvel[:] = frame * 10
        np.testing.assert_allclose(policy.get_joint_pos_relative(), expected)
        np.testing.assert_allclose(policy.get_joint_vel(), expected * 10)
    policy.reset_joint_observation_history()
    policy.data.qpos[:] = 5
    policy.data.qvel[:] = 50
    np.testing.assert_allclose(policy.get_joint_pos_relative(), 5)
    np.testing.assert_allclose(policy.get_joint_vel(), 50)


def test_captured_joint_ages_keep_only_fresh_walk_frames(tmp_path):
    path = tmp_path/'capture.jsonl'
    frames = [
        dict(policy='walk', feedback=dict(control_valid=True, error=None, joint_age_s=.005)),
        dict(policy='walk', feedback=dict(control_valid=True, error=None, joint_age_s=.041)),
        dict(policy='walk', feedback=dict(control_valid=True, error=None, joint_age_s=.081)),
        dict(policy='held', feedback=dict(control_valid=True, error=None, joint_age_s=.02)),
        dict(policy='walk', feedback=dict(control_valid=False, error='fault', joint_age_s=.02)),
    ]
    path.write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
    np.testing.assert_array_equal(step.joint_age_lags_from_capture(path), [1, 3, 4])
