from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
import sys
from pathlib import Path

import numpy as np
import torch

from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg
from mjlab_microduck.tasks.mdp import HdLowSpeedCommand, HdLowSpeedCommandCfg


def test_warp_replay_uses_recipe_and_explicit_coherent_feedback_ages():
    import pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from replay_hd1910_warp import make_replay_cfg, policy_recipe
    metadata = {'training_recipe': 'm6_repair_gait_luwu_linear_only_v22_v1',
                'command_semantics': 'twist_head_body'}
    variant = policy_recipe(metadata, True)
    assert variant == 'gait_luwu_linear_only_v22'
    cfg = make_replay_cfg(20, slew=True, bam_reference=True, repair_variant=variant,
                          joint_age_steps=3, imu_age_steps=1)
    terms = cfg.observations['actor'].terms
    assert 'joint_state' in terms and 'joint_pos' not in terms
    assert terms['joint_state'].delay_min_lag == terms['joint_state'].delay_max_lag == 3
    assert not terms['joint_state'].params['training_noise']
    assert terms['joint_state'].delay_hold_prob == 0
    assert terms['base_ang_vel'].delay_max_lag == terms['projected_gravity'].delay_max_lag == 1
    assert cfg.actions['joint_pos'].command_loss_probability_range == (0., 0.)
    for semantics in ('phase_cos_sin_zero', 'sit_flag_zero_zero', 'episodic_roll_zero_command'):
        with pytest.raises(ValueError, match='cannot execute'):
            policy_recipe({**metadata, 'command_semantics': semantics}, True)
    with pytest.raises(ValueError, match='unsupported'):
        policy_recipe({'training_recipe': 'unknown'}, True)
    with pytest.raises(ValueError, match='feedback ages'):
        make_replay_cfg(20, joint_age_steps=-1)


def test_v6_preserves_actuation_and_backward_practice():
    parent = make_xgo_bam_env_cfg(repair_variant='gait_head_dc_stride_v4')
    for name in ('gait_timing_v6', 'gait_forward_balance_v6'):
        cfg = make_xgo_bam_env_cfg(repair_variant=name)
        assert cfg.actions == parent.actions
        assert cfg.scene.entities['robot'].articulation == parent.scene.entities['robot'].articulation
        assert cfg.observations['actor'].terms['joint_state'].delay_max_lag == 2
        assert cfg.observations['actor'].terms['projected_gravity'].delay_max_lag == 1
        assert cfg.rewards['head_pose_bias'].weight == 5.
        assert 'head_pose_bias_weight' not in cfg.curriculum
    assert cfg.commands['twist'].forward_probability == .7
    assert cfg.rewards['hd_trunk_balance'].weight < parent.rewards['hd_trunk_balance'].weight
    assert cfg.rewards['foot_swing_height'] == parent.rewards['foot_swing_height']


def test_forward_buckets_keep_idle_reverse_and_turns():
    torch.manual_seed(42)
    n = 10000
    term = HdLowSpeedCommand.__new__(HdLowSpeedCommand)
    term._env = SimpleNamespace(device='cpu', num_envs=n)
    term.cfg = HdLowSpeedCommandCfg(entity_name='robot',
        ranges=HdLowSpeedCommandCfg.Ranges(lin_vel_x=(-.15,.15),
            lin_vel_y=(-.04,.04), ang_vel_z=(-.5,.5)),
        resampling_time_range=(2.,4.), forward_probability=.7)
    term.vel_command_b = torch.zeros(n,3)
    term.vel_command_w = torch.zeros(n,3)
    for name in ('is_standing_env','is_world_env','is_heading_env','is_forward_env'):
        setattr(term,name,torch.zeros(n,dtype=torch.bool))
    term._resample_command(torch.arange(n))
    q = term.vel_command_b
    straight = (q[:,1:] == 0).all(dim=1) & (q[:,0] != 0)
    assert .67 < (q[straight,0] > 0).float().mean() < .73
    assert (q[straight,0] < 0).sum() > 1000
    assert (q == 0).all(dim=1).sum() > 1500
    assert ((q[:,0] == 0) & (q[:,2] != 0)).sum() > 2000


def test_v7_keeps_long_delay_and_parent_balance_recipe():
    parent = make_xgo_bam_env_cfg(repair_variant='gait_head_dc_stride_v4')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_forward_tail_v7')
    assert cfg.actions == parent.actions
    assert cfg.observations == parent.observations
    assert cfg.events == parent.events
    assert cfg.commands['twist'].forward_probability == .65
    assert cfg.rewards['head_pose_bias'].weight == 5.
    for name in ('hd_trunk_balance', 'foot_swing_height', 'hd_slew_demand'):
        assert cfg.rewards[name] == parent.rewards[name]


def test_replay_imu_delay_changes_only_policy_observation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from replay_hd1910 import ReplayPolicy, PolicyInference
    policy = ReplayPolicy.__new__(ReplayPolicy)
    policy.data = SimpleNamespace(time=.02)
    policy.set_imu_observation_delay(10.)
    policy.imu_samples = deque((t, np.full(6, t)) for t in (0., .005, .01, .015, .02))
    base = np.arange(61,dtype=np.float32)
    with patch.object(PolicyInference, 'get_observations', return_value=base.copy()):
        actual = policy.get_observations()
    np.testing.assert_allclose(actual[:6], .01)
    np.testing.assert_array_equal(actual[6:], base[6:])
    policy.reset_joint_observation_history()
    assert not policy.imu_samples


def test_coherent_velocity_has_no_hidden_extra_cycle():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from replay_hd1910 import ReplayPolicy, PolicyInference
    policy = ReplayPolicy.__new__(ReplayPolicy)
    policy.set_joint_observation_delay(1)
    policy.coherent_joint_snapshot = True
    policy.delayed_joint_velocity = np.ones(14)
    np.testing.assert_array_equal(policy.get_joint_vel(), np.ones(14))
    policy.delayed_joint_velocity = np.full(14, 2.)
    np.testing.assert_array_equal(policy.get_joint_vel(), np.full(14, 2.))
    policy.legacy_velocity_lag = True
    policy.get_joint_vel()
    policy.delayed_joint_velocity = np.full(14, 3.)
    np.testing.assert_array_equal(policy.get_joint_vel(), np.full(14, 2.))
    policy.set_joint_observation_delay(0)
    policy.legacy_velocity_lag = False
    with patch.object(PolicyInference, 'get_joint_vel', return_value=np.full(14, 4.)):
        np.testing.assert_array_equal(policy.get_joint_vel(), np.full(14, 4.))


def test_ground_contact_replay_keeps_joints_and_inertia_but_can_detect_head_impact():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from replay_hd1910 import load_replay_model
    old, _, _ = load_replay_model(7.4, bam_reference=True)
    new, _, _ = load_replay_model(7.4, bam_reference=True, ground_contact=True)
    assert [old.joint(i).name for i in range(old.njnt)] == [new.joint(i).name for i in range(new.njnt)]
    np.testing.assert_allclose(old.body_mass, new.body_mass, atol=1e-7)
    np.testing.assert_allclose(old.body_inertia, new.body_inertia, atol=1e-7)
    np.testing.assert_allclose(old.jnt_range, new.jnt_range)
    counts = []
    for model in (old, new):
        floor, head = model.geom('floor').id, model.body('jaw_soft').id
        counts.append(sum(model.geom_bodyid[i] == head and bool(
            (model.geom_contype[i] & model.geom_conaffinity[floor])
            or (model.geom_contype[floor] & model.geom_conaffinity[i])) for i in range(model.ngeom)))
    assert counts[0] == 0 and counts[1] > 0
