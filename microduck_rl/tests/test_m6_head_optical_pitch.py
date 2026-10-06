"""The head-lowering experiment must use the physical camera axis, not joint bias."""
import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks.mdp import hd_head_optical_pitch_cost


def make_env(pitch_deg, upright=True):
    half = math.radians(pitch_deg) / 2
    quat = torch.tensor([[[math.cos(half), 0., math.sin(half), 0.]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(site_quat_w=quat,
            projected_gravity_b=torch.tensor([[0., 0., -1. if upright else -.5]]),
            root_link_pos_w=torch.tensor([[0., 0., .12]])),
        find_sites=lambda name: ([0], [name]))
    return SimpleNamespace(scene={'robot': robot})


def test_optical_pitch_target_and_upright_gate():
    # The model's camera optical axis is the head_camera site's -X.
    target = hd_head_optical_pitch_cost(make_env(-12.))
    higher = hd_head_optical_pitch_cost(make_env(-9.))
    looking_up = hd_head_optical_pitch_cost(make_env(5.))
    fallen = hd_head_optical_pitch_cost(make_env(5., upright=False))
    assert target.item() < 1e-6
    assert 0 < higher.item() < looking_up.item()
    assert fallen.item() == 0.


def test_reward_axis_matches_mujoco_camera_optical_axis():
    import mujoco
    import numpy as np
    import sys
    from pathlib import Path
    from mjlab_microduck.tasks.mdp import _head_camera_pitch
    sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
    from replay_hd1910 import DEFAULT_POSE, load_replay_model

    model, data, motor = load_replay_model(7.4, bam_reference=True)
    site_id = model.site('head_camera').id
    camera_id = model.camera('head_camera').id
    joint_id = model.joint('head_pitch').qposadr[0]
    data.qpos[motor.qids] = DEFAULT_POSE
    for delta in (0., .2, -.2):
        data.qpos[joint_id] += delta
        mujoco.mj_forward(model, data)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        robot = SimpleNamespace(data=SimpleNamespace(site_quat_w=torch.tensor(quat[None, None, :])))
        env = SimpleNamespace(scene={'robot': robot}, _hd_head_camera_site_id=0)
        optical = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        expected = math.atan2(optical[2], np.linalg.norm(optical[:2]))
        assert abs(_head_camera_pitch(env).item() - expected) < 1e-6
        data.qpos[joint_id] -= delta
