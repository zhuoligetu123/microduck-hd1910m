"""The upstream baseline must not silently inherit local action semantics."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from replay_hd1910 import ReferenceReplayPolicy, ReplayPolicy, reference_policy_pose
from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg, configure_head_bias_course
from mjlab_microduck.tasks.mdp import HdLowSpeedCommand, standing_envs_curriculum


@pytest.mark.parametrize('height', [12., 15., 20., 25.])
def test_unified_course_has_one_sole_target_without_changing_execution(height):
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import (configure_clearance_course,
        configure_airtime_height_gate, configure_bilateral_clearance_bonus)
    from mjlab_microduck.tasks.mdp import hd_sole_swing_height
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    configure_airtime_height_gate(cfg, 'gentle')
    configure_bilateral_clearance_bonus(cfg, 1.)
    before = deepcopy(cfg)
    configure_clearance_course(cfg, height)
    assert 'foot_clearance' not in cfg.rewards
    for name, term in cfg.rewards.items():
        assert term.weight == before.rewards[name].weight
        if 'target_height' in term.params:
            assert term.func is hd_sole_swing_height
            assert term.params['target_height'] == height/1000
            assert term.params['shortfall_only'] and term.params['tolerance'] == 0.
    for field in ('commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(before, field) == getattr(cfg, field)
    for invalid in (0., 13., float('nan')):
        with pytest.raises(ValueError):
            configure_clearance_course(cfg, invalid)


def test_raw_history_and_filtered_output_are_distinct(monkeypatch):
    def infer(policy):
        policy.last_action = np.ones(14, dtype=np.float32)
        return policy.last_action.copy()
    monkeypatch.setattr(ReplayPolicy, 'infer', infer)
    policy = ReferenceReplayPolicy.__new__(ReferenceReplayPolicy)
    policy.action_alpha = .45
    policy.reset_joint_observation_history()
    np.testing.assert_allclose(policy.infer(), .55)
    np.testing.assert_allclose(policy.last_action, 1.)
    np.testing.assert_allclose(policy.infer(), .7975)
    policy.reset_joint_observation_history()
    np.testing.assert_allclose(policy.infer(), .55)


def test_external_contract_rejects_local_bounded_history():
    names = [str(i) for i in range(14)]
    meta = dict(joint_names=','.join(names), action_scale='1.0',
        observation_names='base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command',
        default_joint_pos=','.join(['.349']*14))
    np.testing.assert_allclose(reference_policy_pose(meta, names), .349)
    meta['action_semantics'] = 'bounded_slew_home_delta_v2'
    with pytest.raises(ValueError):
        reference_policy_pose(meta, names)


def test_scheduled_head_command_reaches_observation_slots():
    policy = ReplayPolicy.__new__(ReplayPolicy)
    policy.new_cmd_obs = True
    policy.behavior_mode = None
    policy.current_policy = 'walking'
    policy.vel_cmd = np.array([.1, 0, 0])
    policy.body_cmd = np.zeros(6)
    policy.head_offset = np.zeros(4)
    policy.set_head_command(0.)
    np.testing.assert_allclose(policy.command[3:7], 0.)
    policy.set_head_command([.2, -.3, 0, 0])
    np.testing.assert_allclose(policy.command[3:7], [.2, -.3, 0, 0])
    np.testing.assert_allclose(policy.command[:3], [.1, 0, 0])


def test_reward_reset_preserves_physics_and_drops_added_penalties():
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_reference_recipe_v18')
    base = make_xgo_bam_env_cfg()
    payload = make_xgo_bam_env_cfg(repair_variant='gait_payload_v8')
    assert cfg.rewards == base.rewards
    assert cfg.actions == payload.actions
    assert cfg.observations == payload.observations
    assert cfg.events == payload.events
    assert 'hd_slew_demand' not in cfg.rewards
    for term in cfg.curriculum.values():
        if 'reward_name' in term.params:
            assert term.params['reward_name'] in cfg.rewards


def test_tracking_tolerance_scales_with_command_range_not_weight():
    base = make_xgo_bam_env_cfg(repair_variant='gait_reference_recipe_v18')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_reference_scaled_v19')
    for name, ratio in [('track_linear_velocity', .15/.4),
                        ('track_angular_velocity', .5/1.)]:
        assert cfg.rewards[name].weight == base.rewards[name].weight
        assert cfg.rewards[name].params['std'] == pytest.approx(base.rewards[name].params['std'] * ratio)
    std = cfg.rewards['track_linear_velocity'].params['std']
    assert np.exp(-.1**2/std**2) < .5
    assert cfg.actions == base.actions


def test_idle_ablation_changes_only_task_fraction_and_its_curriculum():
    from copy import deepcopy
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_recipe_v18')
    b = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_v20')
    assert b.commands['twist'].rel_standing_envs == .02
    assert a.commands['twist'].rel_standing_envs == .20
    adjusted = deepcopy(b.commands)
    adjusted['twist'].rel_standing_envs = .20
    assert adjusted == a.commands
    assert b.rewards == a.rewards
    assert b.actions == a.actions
    assert b.events == a.events
    assert b.observations == a.observations
    b.curriculum.pop('standing_envs')
    assert b.curriculum == a.curriculum


@pytest.mark.parametrize('idle_fraction', [0., .02, .2, .25, 1.])
def test_sampled_idle_fraction_matches_live_config(idle_fraction):
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_reference_recipe_v18').commands['twist']
    cfg.rel_standing_envs = idle_fraction
    count = 50000
    command = SimpleNamespace(cfg=cfg, device='cpu',
        vel_command_b=torch.zeros(count, 3), vel_command_w=torch.zeros(count, 3))
    for name in ('is_standing_env', 'is_world_env', 'is_heading_env', 'is_forward_env'):
        setattr(command, name, torch.zeros(count, dtype=torch.bool))
    torch.manual_seed(2026)
    HdLowSpeedCommand._resample_command(command, torch.arange(count))
    idle = command.is_standing_env
    assert abs(idle.float().mean().item() - idle_fraction) < .008
    assert (command.vel_command_b[idle] == 0).all()
    moving = command.vel_command_b[~idle]
    if len(moving):
        straight = (moving[:, 1:] == 0).all(dim=1)
        turn = (moving[:, :2] == 0).all(dim=1)
        assert abs(straight.float().mean().item() - .5) < .015
        assert abs(turn.float().mean().item() - .375) < .015
        assert (moving[straight, 0].abs() >= .075).all()
        assert (moving[turn, 2].abs() >= .25).all()


def test_curriculum_updates_live_command_not_a_config_copy():
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_v20')
    term = SimpleNamespace(cfg=cfg.commands['twist'])
    class Manager:
        def get_term(self, name):
            assert name == 'twist'
            return term
    env = SimpleNamespace(command_manager=Manager(), common_step_counter=0)
    params = cfg.curriculum['standing_envs'].params
    for iteration, expected in [(0, .02), (501, .05), (1001, .15), (2001, .25)]:
        env.common_step_counter = iteration * 24
        standing_envs_curriculum(env, None, **params)
        assert term.cfg.rel_standing_envs == expected


def test_curriculum_scaled_ablation_only_changes_tracking_tolerance():
    from copy import deepcopy
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_v20')
    b = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    rewards = deepcopy(b.rewards)
    for name, ratio in [('track_linear_velocity', .15/.4),
                        ('track_angular_velocity', .5/1.)]:
        assert b.rewards[name].params['std'] == pytest.approx(a.rewards[name].params['std'] * ratio)
        rewards[name].params['std'] = a.rewards[name].params['std']
    assert rewards == a.rewards
    for field in ('commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)
    assert b.commands['twist'].rel_standing_envs == .02


@pytest.mark.parametrize('play', [False, True])
def test_linear_only_ablation_changes_only_yaw_tolerance(play):
    from copy import deepcopy
    a = make_xgo_bam_env_cfg(play=play, repair_variant='gait_reference_curriculum_scaled_v21')
    b = make_xgo_bam_env_cfg(play=play, repair_variant='gait_reference_linear_only_v22')
    original = make_xgo_bam_env_cfg(play=play, repair_variant='gait_reference_curriculum_v20')
    assert b.rewards['track_angular_velocity'] == original.rewards['track_angular_velocity']
    assert b.rewards['track_linear_velocity'] == a.rewards['track_linear_velocity']
    rewards = deepcopy(b.rewards)
    rewards['track_angular_velocity'].params['std'] = a.rewards['track_angular_velocity'].params['std']
    assert rewards == a.rewards
    for field in ('commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


@pytest.mark.parametrize('variant', ['gait_reference_curriculum_scaled_v21', 'gait_reference_linear_only_v22'])
@pytest.mark.parametrize('play', [False, True])
def test_head_bias_ablation_leaves_every_other_term_unchanged(variant, play):
    from copy import deepcopy
    a = make_xgo_bam_env_cfg(play=play, repair_variant=variant)
    b = deepcopy(a)
    configure_head_bias_course(a, 'scheduled')
    configure_head_bias_course(b, 'frozen')
    assert a.rewards == b.rewards
    if 'head_pose_bias_weight' in b.curriculum:
        assert all(s['weight'] == 0 for s in b.curriculum['head_pose_bias_weight'].params['weight_stages'])
        b.curriculum['head_pose_bias_weight'] = deepcopy(a.curriculum['head_pose_bias_weight'])
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


def test_head_bias_boundary_changes_live_reward_only():
    from mjlab_microduck.tasks.mdp import reward_weight
    from copy import deepcopy
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    b = deepcopy(a)
    configure_head_bias_course(b, 'frozen')
    for cfg, expected in ((a, 1.), (b, 0.)):
        class Rewards:
            def get_term_cfg(self, name):
                return cfg.rewards[name]
        env = SimpleNamespace(reward_manager=Rewards(), common_step_counter=601*24)
        reward_weight(env, None, **cfg.curriculum['head_pose_bias_weight'].params)
        assert cfg.rewards['head_pose_bias'].weight == expected
        assert cfg.curriculum['action_rate_weight'] == a.curriculum['action_rate_weight']
        assert cfg.commands == a.commands


@pytest.mark.parametrize('play', [False, True])
@pytest.mark.parametrize('weight', [-.2, -.1])
def test_action_rate_ablation_changes_only_penalty_and_schedule(play, weight):
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_action_rate_weight
    a = make_xgo_bam_env_cfg(play=play, repair_variant='gait_reference_curriculum_scaled_v21')
    b = deepcopy(a)
    configure_action_rate_weight(b, weight)
    assert b.rewards['action_rate_l2'].weight == weight
    assert 'action_rate_weight' not in b.curriculum
    b.rewards['action_rate_l2'].weight = a.rewards['action_rate_l2'].weight
    if 'action_rate_weight' in a.curriculum:
        b.curriculum['action_rate_weight'] = a.curriculum['action_rate_weight']
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), .1])
def test_action_rate_ablation_rejects_invalid_penalty(weight):
    from mjlab_microduck.tasks.hd1910_bam import configure_action_rate_weight
    with pytest.raises(ValueError, match='nonpositive'):
        configure_action_rate_weight(SimpleNamespace(), weight)


@pytest.mark.parametrize('kind', ['linear', 'angular'])
def test_axis_tolerance_preserves_upstream_when_equal(kind):
    from mjlab.tasks.velocity.mdp import rewards
    from mjlab_microduck.tasks import mdp
    generator = torch.Generator().manual_seed(7)
    actual = torch.randn(30, 3, generator=generator)
    command = torch.randn(30, 3, generator=generator)
    class Commands:
        def get_command(self, name):
            return command
    env = SimpleNamespace(command_manager=Commands(), scene={'robot': SimpleNamespace(data=
        SimpleNamespace(root_link_lin_vel_b=actual, root_link_ang_vel_b=actual))})
    ref = getattr(rewards, 'track_' + kind + '_velocity')(env, std=.3, command_name='twist')
    test = getattr(mdp, 'hd_track_' + kind + '_velocity_axes')(
        env, std=.3, stability_std=.3, command_name='twist')
    torch.testing.assert_close(test, ref)


def test_axis_tolerance_changes_only_uncommanded_axis_penalty():
    from mjlab_microduck.tasks.mdp import hd_track_linear_velocity_axes, hd_track_angular_velocity_axes
    class Commands:
        def get_command(self, name):
            return torch.tensor([[.1, 0., .4]])
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[.1, 0., .1]]),
                           root_link_ang_vel_b=torch.tensor([[.5, 0., .4]]))
    env = SimpleNamespace(command_manager=Commands(), scene={'robot': SimpleNamespace(data=data)})
    for func, std, stability in ((hd_track_linear_velocity_axes, .1185854, .3162278),
                                 (hd_track_angular_velocity_axes, .3535534, .7071068)):
        assert func(env, std, stability, 'twist') > func(env, std, std, 'twist')
    data.root_link_lin_vel_b[:, 2] = 0.
    data.root_link_ang_vel_b[:, :2] = 0.
    for func in (hd_track_linear_velocity_axes, hd_track_angular_velocity_axes):
        torch.testing.assert_close(func(env, .1, .7, 'twist'), torch.ones(1))


def test_axis_tolerance_ablation_keeps_all_other_terms():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_tracking_axes
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    b = deepcopy(a)
    configure_tracking_axes(a, 'coupled')
    configure_tracking_axes(b, 'separate')
    for name in ('track_linear_velocity', 'track_angular_velocity'):
        assert a.rewards[name].weight == b.rewards[name].weight
        assert a.rewards[name].params['std'] == b.rewards[name].params['std']
        assert b.rewards[name].params['stability_std'] > a.rewards[name].params['std']
        b.rewards[name] = a.rewards[name]
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


def test_swing_reference_keeps_target_weight_and_other_terms():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_swing_reference
    from mjlab_microduck.tasks.mdp import hd_sole_swing_height
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    b = deepcopy(a)
    configure_swing_reference(a, 'ray')
    configure_swing_reference(b, 'collision')
    term = b.rewards['foot_swing_height']
    assert term.func is hd_sole_swing_height
    assert term.weight == a.rewards['foot_swing_height'].weight
    assert term.params['target_height'] == a.rewards['foot_swing_height'].params['target_height']
    assert term.params.pop('tolerance') == 0.
    term.func = a.rewards['foot_swing_height'].func
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


def test_standing_fraction_freeze_preserves_all_other_terms():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_standing_fraction
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    b = deepcopy(a)
    configure_standing_fraction(b, .05)
    assert b.commands['twist'].rel_standing_envs == .05
    assert 'standing_envs' not in b.curriculum
    b.commands['twist'].rel_standing_envs = a.commands['twist'].rel_standing_envs
    b.curriculum['standing_envs'] = a.curriculum['standing_envs']
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


def test_mirror_loss_never_augments_unmirrored_privileged_critic():
    from mjlab_microduck.tasks.hd1910_bam import mirror_loss_config
    assert mirror_loss_config(.1)['mirror_loss_coeff'] == .1
    assert mirror_loss_config(.1)['use_mirror_loss']
    assert not mirror_loss_config(.1)['use_data_augmentation']
    assert not mirror_loss_config(0.)['use_mirror_loss']
    for value in [-1., float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            mirror_loss_config(value)


@pytest.mark.parametrize('kind', ['linear', 'angular'])
def test_tracking_mean_resets_at_episode_and_command_boundaries(kind):
    from mjlab_microduck.tasks.mdp import hd_cycle_velocity_tracking
    command = torch.zeros(2, 3)
    velocity = torch.zeros(2, 3)
    class Commands:
        def get_command(self, name):
            return command
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02,
        episode_length_buf=torch.tensor([2, 2]), command_manager=Commands(),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity,
                                                           root_link_ang_vel_b=velocity))})
    params = dict(kind=kind, mean_seconds=.2, command_name='twist', std=.2)
    term = hd_cycle_velocity_tracking(SimpleNamespace(params=params), env)
    torch.testing.assert_close(term(env, **params), torch.ones(2))
    velocity[:, :] = .1
    term(env, **params)
    assert (term.mean_velocity < velocity).all()
    term.reset(torch.tensor([0]))
    term(env, **params)
    torch.testing.assert_close(term.mean_velocity[0], velocity[0])
    assert (term.mean_velocity[1] < velocity[1]).all()
    command[1, 0] = .1
    term(env, **params)
    torch.testing.assert_close(term.mean_velocity[1], velocity[1])
    velocity[0] = .2
    env.episode_length_buf[0] = 1
    term(env, **params)
    torch.testing.assert_close(term.mean_velocity[0], velocity[0])


def test_tracking_mean_does_not_hide_vertical_instability_or_steady_error():
    from mjlab_microduck.tasks.mdp import hd_cycle_velocity_tracking
    command = torch.zeros(1, 3)
    velocity = torch.tensor([[0., 0., 1.]])
    class Commands:
        def get_command(self, name):
            return command
    env = SimpleNamespace(num_envs=1, device='cpu', step_dt=.02,
        episode_length_buf=torch.tensor([2]), command_manager=Commands(),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))})
    params = dict(kind='linear', mean_seconds=.2, command_name='twist', std=.2)
    term = hd_cycle_velocity_tracking(SimpleNamespace(params=params), env)
    assert term(env, **params).item() < 1e-5
    velocity[:] = torch.tensor([[.2, 0., 0.]])
    for _ in range(200):
        value = term(env, **params)
    assert value.item() == pytest.approx(np.exp(-1), rel=1e-5)


def test_tracking_mean_changes_reward_timing_not_policy_contract():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_tracking_mean
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_linear_only_v22')
    b = deepcopy(a)
    configure_tracking_mean(a, 0.)
    configure_tracking_mean(b, .2)
    for name in ('track_linear_velocity', 'track_angular_velocity'):
        term = b.rewards[name]
        assert term.weight == a.rewards[name].weight
        assert term.params.pop('mean_seconds') == .2
        assert term.params.pop('kind') in ('linear', 'angular')
        term.func = a.rewards[name].func
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


def test_straight_yaw_changes_only_one_reward_and_rejects_conflicting_mean():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import (
        configure_straight_yaw, configure_tracking_axes, configure_tracking_mean)
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    configure_tracking_axes(a, 'separate_yaw')
    b = deepcopy(a)
    configure_straight_yaw(b, .1)
    assert b.rewards['track_angular_velocity'].weight == a.rewards['track_angular_velocity'].weight
    assert b.rewards['track_angular_velocity'].params['mean_seconds'] == .4
    assert b.rewards['track_angular_velocity'].params['straight_std'] == .1
    b.rewards['track_angular_velocity'] = deepcopy(a.rewards['track_angular_velocity'])
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)
    for value in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_straight_yaw(b, value)
    configure_tracking_mean(b, .2)
    with pytest.raises(ValueError):
        configure_straight_yaw(b, .1)


def test_straight_yaw_preserves_turn_idle_and_prices_persistent_drift():
    from mjlab_microduck.tasks.mdp import hd_cycle_velocity_tracking, hd_track_angular_velocity_axes
    command = torch.tensor([[.1, 0., 0.], [-.1, 0., 0.], [0., 0., .4], [0., 0., 0.]])
    velocity = torch.zeros_like(command)
    class Commands:
        def get_command(self, name):
            return command
    env = SimpleNamespace(num_envs=4, device='cpu', step_dt=.02,
        episode_length_buf=torch.full((4,), 2), command_manager=Commands(),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=velocity))})
    common = dict(std=.3535534, stability_std=.7071068, command_name='twist', stability_weight=0.)
    params = dict(**common, kind='angular', mean_seconds=.4, straight_std=.1)
    term = hd_cycle_velocity_tracking(SimpleNamespace(params=params), env)
    periodic, drifting = [], []
    for step in range(300):
        velocity[:, 2] = .2 * np.sin(2 * np.pi * step / 20)
        velocity[1, 2] += .1
        value = term(env, **params)
        old = hd_track_angular_velocity_axes(env, **common)
        torch.testing.assert_close(value[2:], old[2:])
        if step >= 200:
            periodic.append(value[0].item())
            drifting.append(value[1].item())
    assert np.mean(periodic) > .93
    assert np.mean(drifting) < .45
    command[0, 2] = .4
    torch.testing.assert_close(term(env, **params)[0], hd_track_angular_velocity_axes(env, **common)[0])
    torch.testing.assert_close(term.mean_velocity[0], velocity[0])


def test_airtime_quality_does_not_add_a_new_reward_or_change_weight():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_airtime_height_gate
    from mjlab_microduck.tasks.mdp import hd_sole_swing_height
    a = make_xgo_bam_env_cfg(repair_variant='gait_reference_linear_only_v22')
    b = deepcopy(a)
    configure_airtime_height_gate(a, 'off')
    configure_airtime_height_gate(b, 'on')
    assert a.rewards.keys() == b.rewards.keys()
    assert b.rewards['air_time'].weight == a.rewards['air_time'].weight
    assert b.rewards['air_time'].func is hd_sole_swing_height
    assert b.rewards['air_time'].params['target_height'] == .020
    for name, value in a.rewards['air_time'].params.items():
        assert b.rewards['air_time'].params[name] == value
    b.rewards['air_time'] = a.rewards['air_time']
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(a, field) == getattr(b, field)


def test_feedback_age_course_only_changes_training_snapshot_lag():
    from copy import deepcopy
    import pytest
    from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg, configure_feedback_age
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_linear_only_v22')
    candidate = deepcopy(parent)
    configure_feedback_age(candidate, 1)
    term = candidate.observations['actor'].terms['joint_state']
    assert term.delay_min_lag == 0 and term.delay_max_lag == 1
    assert candidate.actions == parent.actions and candidate.events == parent.events
    assert candidate.rewards == parent.rewards and candidate.commands == parent.commands
    assert candidate.curriculum == parent.curriculum
    term.delay_max_lag = 4
    assert candidate.observations == parent.observations
    for value in (-1, 9, .5, True):
        with pytest.raises(ValueError):
            configure_feedback_age(candidate, value)


def test_bilateral_bonus_keeps_all_existing_rewards_and_interfaces():
    from copy import deepcopy
    import pytest
    from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg, configure_bilateral_clearance_bonus
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_bilateral_clearance_bonus(candidate, 1.)
    bonus = candidate.rewards.pop('hd_bilateral_clearance_quality')
    assert bonus.weight == 1. and bonus.params['target_height'] == .025
    assert bonus.params['reward_bilateral_quality'] and bonus.params['reward_bilateral']
    assert candidate.rewards == parent.rewards
    assert candidate.actions == parent.actions and candidate.observations == parent.observations
    assert candidate.events == parent.events and candidate.curriculum == parent.curriculum
    assert candidate.commands == parent.commands
    for value in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_bilateral_clearance_bonus(candidate, value)


def test_resume_exploration_cap_preserves_actor_mean_and_other_optimizer_state():
    from types import SimpleNamespace
    import pytest
    import torch
    from mjlab_microduck.tasks.hd1910_bam import cap_resume_exploration
    mean = torch.nn.Parameter(torch.tensor([.2, -.1]))
    std = torch.nn.Parameter(torch.tensor([.5, .1]))
    optimizer = torch.optim.Adam([mean, std], lr=.001)
    (mean.sum() + std.sum()).backward()
    optimizer.step()
    before_mean = mean.detach().clone()
    mean_momentum = optimizer.state[mean]['exp_avg'].clone()
    alg = SimpleNamespace(actor=SimpleNamespace(distribution=SimpleNamespace(
        std_type='scalar', std_param=std)), optimizer=optimizer)
    change = cap_resume_exploration(alg, .15)
    assert torch.equal(mean, before_mean)
    assert torch.equal(optimizer.state[mean]['exp_avg'], mean_momentum)
    assert std not in optimizer.state
    assert torch.allclose(std.detach(), torch.tensor([.15, .099]))
    assert change['before'][0] > .49 and change['after'][0] < .151
    for cap in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            cap_resume_exploration(alg, cap)


def test_hip_roll_tolerance_does_not_change_standing_or_other_terms():
    from copy import deepcopy
    import pytest
    from mjlab_microduck.tasks.hd1910_bam import configure_walking_hip_roll_std
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_walking_hip_roll_std(candidate, .15)
    for regime in ('std_walking', 'std_running'):
        assert candidate.rewards['pose'].params[regime]['.*hip_roll.*'] == .15
    for regime in ('std_walking', 'std_running'):
        candidate.rewards['pose'].params[regime]['.*hip_roll.*'] = .05
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)
    for value in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_walking_hip_roll_std(candidate, value)


def test_airtime_shift_retains_interval_width_weights_and_interfaces():
    from copy import deepcopy
    import pytest
    from mjlab_microduck.tasks.hd1910_bam import configure_airtime_window_shift
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_airtime_window_shift(candidate, .06)
    params = candidate.rewards['air_time'].params
    assert params['threshold_min'] == pytest.approx(.185)
    assert params['threshold_max'] == pytest.approx(.360)
    assert params['threshold_max'] - params['threshold_min'] == pytest.approx(.175)
    for name in ('threshold_min', 'threshold_max'):
        params[name] = parent.rewards['air_time'].params[name]
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)
    for value in (-1., .21, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_airtime_window_shift(candidate, value)


def test_low_obstacle_course_retains_robot_rewards_and_observation_contract():
    from copy import deepcopy
    import pytest
    from mjlab_microduck.tasks.hd1910_bam import configure_terrain_course, configure_bilateral_clearance_bonus
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    flat, obstacle = deepcopy(parent), deepcopy(parent)
    configure_terrain_course(flat, 'flat')
    configure_terrain_course(obstacle, 'microblocks')
    for candidate in (flat, obstacle):
        for field in ('rewards', 'commands', 'actions', 'observations', 'terminations', 'sim'):
            assert getattr(candidate, field) == getattr(parent, field)
        robot, original = candidate.scene.entities['robot'], parent.scene.entities['robot']
        for field in ('articulation', 'init_state', 'collisions'):
            assert getattr(robot, field) == getattr(original, field)
        assert robot.spec_fn.func is original.spec_fn.func
        assert robot.spec_fn.args == original.spec_fn.args
        assert robot.spec_fn.keywords == original.spec_fn.keywords
        assert candidate.events['reset_base'].params['pose_range']['x'] == (-.1, .1)
        assert candidate.scene.terrain.max_init_terrain_level == 0
    assert flat.events == obstacle.events and flat.curriculum == obstacle.curriculum
    box = obstacle.scene.terrain.terrain_generator.sub_terrains['microblocks']
    assert box.box_height_range == (.006, .006)
    assert box.platform_width == .6
    configure_bilateral_clearance_bonus(parent, 1.)
    with pytest.raises(ValueError, match='plane-only'):
        configure_terrain_course(parent, 'microblocks')


def test_twelve_mm_obstacles_are_above_current_sole_clearance_not_half_height():
    import numpy as np
    import mujoco
    from mjlab.terrains.terrain_generator import TerrainGenerator
    from mjlab_microduck.tasks.hd1910_bam import configure_terrain_course
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    configure_terrain_course(cfg, 'microblocks12')
    generator = TerrainGenerator(cfg.scene.terrain.terrain_generator)
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    tops = model.geom_pos[:, 2] + model.geom_size[:, 2]
    assert len(tops[tops > 1e-8]) > 0
    assert np.allclose(tops[tops > 1e-8], .012)


def test_slew_demand_ablation_keeps_actuation_and_existing_rewards():
    from copy import deepcopy
    import pytest
    from mjlab_microduck.tasks.hd1910_bam import configure_slew_demand_weight
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_slew_demand_weight(candidate, -.2)
    assert candidate.rewards.pop('hd_slew_demand').weight == -.2
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)
    for weight in (.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_slew_demand_weight(candidate, weight)


def test_slew_demand_distinguishes_hidden_requests_with_identical_applied_step():
    from types import SimpleNamespace
    import torch
    from mjlab_microduck.tasks.mdp import hd_slew_demand_cost, hd_applied_action_rate_cost
    term = SimpleNamespace(range_delta=torch.tensor([[.2], [1.]]),
                           raw_action=torch.tensor([[.1], [.1]]),
                           previous_delta=torch.zeros(2, 1))

    class Actions:
        def get_term(self, name):
            assert name == 'joint_pos'
            return term

    env = SimpleNamespace(action_manager=Actions())
    assert torch.allclose(hd_applied_action_rate_cost(env), torch.tensor([.01, .01]))
    assert torch.allclose(hd_slew_demand_cost(env), torch.tensor([.01, .81]))


def test_forward_curriculum_only_changes_straight_direction_sampling():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_forward_probability
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_forward_probability(candidate, .85)
    for field in ('rewards', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)
    candidate.commands['twist'].rel_standing_envs = .05
    count = 100000
    term = SimpleNamespace(cfg=candidate.commands['twist'], device='cpu',
        vel_command_b=torch.zeros(count, 3), vel_command_w=torch.zeros(count, 3))
    for name in ('is_standing_env', 'is_world_env', 'is_heading_env', 'is_forward_env'):
        setattr(term, name, torch.zeros(count, dtype=torch.bool))
    torch.manual_seed(2026)
    HdLowSpeedCommand._resample_command(term, torch.arange(count))
    command = term.vel_command_b
    forward = (command[:, 0] > 0) & (command[:, 1:] == 0).all(dim=1)
    backward = (command[:, 0] < 0) & (command[:, 1:] == 0).all(dim=1)
    turn = (command[:, :2] == 0).all(dim=1) & (command[:, 2] != 0)
    assert forward.float().mean().item() == pytest.approx(.95*.5*.85, abs=.005)
    assert backward.float().mean().item() == pytest.approx(.95*.5*.15, abs=.005)
    assert turn.float().mean().item() == pytest.approx(.95*.375, abs=.005)
    assert term.is_standing_env.float().mean().item() == pytest.approx(.05, abs=.005)
    for value in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            configure_forward_probability(candidate, value)


def test_flexion_tolerance_changes_only_moving_sagittal_pose_prior_once():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_walking_flexion_scale
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_walking_flexion_scale(candidate, 2.)
    for regime in ('std_walking', 'std_running'):
        for joint in ('.*hip_pitch.*', '.*knee.*', '.*ankle.*'):
            assert candidate.rewards['pose'].params[regime][joint] == 2*parent.rewards['pose'].params[regime][joint]
        assert candidate.rewards['pose'].params[regime]['.*hip_roll.*'] == .05
        assert candidate.rewards['pose'].params[regime]['.*hip_yaw.*'] == .3
        candidate.rewards['pose'].params[regime] = deepcopy(parent.rewards['pose'].params[regime])
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)
    for scale in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_walking_flexion_scale(candidate, scale)


def test_yaw_decoupling_preserves_independent_balance_costs_and_command_scale():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_tracking_axes, configure_tracking_mean
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_tracking_axes(parent, 'separate')
    configure_tracking_axes(candidate, 'separate_yaw')
    assert candidate.rewards['track_angular_velocity'].params.pop('stability_weight') == 0.
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)
    candidate.rewards['track_angular_velocity'].params['stability_weight'] = 0.
    configure_tracking_mean(candidate, .2)
    assert candidate.rewards['track_angular_velocity'].params['stability_weight'] == 0.


def test_yaw_reward_does_not_disappear_during_roll_with_explicit_decoupling():
    from mjlab_microduck.tasks.mdp import hd_track_angular_velocity_axes
    velocity = torch.tensor([[0., 0., .3], [2., 0., .3]])
    command = torch.tensor([[0., 0., .4], [0., 0., .4]])

    class Commands:
        def get_command(self, name):
            return command

    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=velocity))},
                          command_manager=Commands())
    shared = dict(std=.35, stability_std=.707, command_name='twist')
    old = hd_track_angular_velocity_axes(env, **shared)
    new = hd_track_angular_velocity_axes(env, **shared, stability_weight=0.)
    assert old[1] < .001
    assert new[0] == new[1]
    assert new[0] == old[0]


def test_fixed_exploration_survives_optimizer_updates_without_changing_actor_mean():
    from rsl_rl.modules.distribution import GaussianDistribution
    from mjlab_microduck.tasks.hd1910_bam import freeze_exploration
    distribution = GaussianDistribution(14, init_std=.5)
    mean = torch.nn.Parameter(torch.linspace(-.1, .1, 14))
    optimizer = torch.optim.Adam([mean, distribution.std_param], lr=.01)
    algorithm = SimpleNamespace(actor=SimpleNamespace(distribution=distribution), optimizer=optimizer)
    distribution.update(mean)
    (distribution.entropy.sum() + mean.square().sum()).backward()
    optimizer.step()
    before = mean.detach().clone()
    freeze_exploration(algorithm, .15)
    assert torch.equal(before, mean.detach())
    assert distribution.std_param not in optimizer.state
    for _ in range(5):
        optimizer.zero_grad()
        distribution.update(mean)
        (-distribution.log_prob(torch.ones_like(mean)).sum() - distribution.entropy.sum()).backward()
        optimizer.step()
    assert not torch.equal(before, mean.detach())
    assert torch.all(distribution.std_param == .15)
    assert distribution.std_param.grad is None
    for invalid in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            freeze_exploration(algorithm, invalid)


def test_latent_action_rate_changes_only_reward_measurement_not_action_contract():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_action_rate_domain
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_action_rate_domain(candidate, 'latent')
    term = SimpleNamespace(raw_action=torch.tensor([[.1], [.1]]),
                           previous_delta=torch.zeros(2, 1))

    class Actions:
        action = torch.tensor([[.2], [1.]])
        prev_action = torch.tensor([[-.2], [-1.]])

        def get_term(self, name):
            return term

    env = SimpleNamespace(action_manager=Actions())
    assert torch.allclose(parent.rewards['action_rate_l2'].func(env), torch.tensor([.01, .01]))
    assert torch.allclose(candidate.rewards['action_rate_l2'].func(env), torch.tensor([.16, 4.]))
    configure_action_rate_domain(candidate, 'applied')
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)


def test_clearance_course_changes_only_bonus_target_without_relaxing_final_acceptance():
    from copy import deepcopy
    from mjlab_microduck.tasks.hd1910_bam import configure_bilateral_clearance_bonus
    parent = make_xgo_bam_env_cfg(repair_variant='gait_reference_curriculum_scaled_v21')
    candidate = deepcopy(parent)
    configure_bilateral_clearance_bonus(parent, 1.)
    configure_bilateral_clearance_bonus(candidate, 1., .012)
    assert candidate.rewards['hd_bilateral_clearance_quality'].params['target_height'] == .012
    assert candidate.rewards['foot_swing_height'].params['target_height'] == .020
    candidate.rewards['hd_bilateral_clearance_quality'].params['target_height'] = .025
    for field in ('rewards', 'commands', 'curriculum', 'events', 'actions', 'observations', 'terminations'):
        assert getattr(candidate, field) == getattr(parent, field)
    for invalid in (0., -.1, .1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_bilateral_clearance_bonus(candidate, 1., invalid)
