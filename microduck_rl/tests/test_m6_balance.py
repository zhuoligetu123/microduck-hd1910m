from types import SimpleNamespace
import sys
from pathlib import Path
import torch
import pytest
from mjlab_microduck.tasks.mdp import hd_trunk_balance_cost
from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from replay_hd1910 import posture_metrics, load_replay_model
from replay_m6_recovery import recovery_success


def test_balance_cost_cannot_reward_flopping_or_crouching():
    data = SimpleNamespace(projected_gravity_b=torch.tensor([[0.,0.,-1.], [.5,0.,-.866],
                       [-.5,0.,-.866], [0.,0.,1.], [0.,0.,-1.]]),
                       root_link_pos_w=torch.tensor([[0.,0.,.12]]*4+[[0.,0.,.06]]))
    cost = hd_trunk_balance_cost(SimpleNamespace(scene={'robot': SimpleNamespace(data=data)}))
    assert cost[0] == 0
    assert (cost[1:] > 0).all()
    assert cost[1] == cost[2]


def test_balance_lift_changes_rewards_only():
    base = make_xgo_bam_env_cfg(repair_variant='control')
    for variant, target in [('balance', .02), ('balance_lift', .025)]:
        cfg = make_xgo_bam_env_cfg(repair_variant=variant)
        assert cfg.actions == base.actions
        assert cfg.observations == base.observations
        assert cfg.events == base.events
        assert cfg.commands == base.commands
        assert cfg.terminations == base.terminations
        assert cfg.scene.entities['robot'].articulation.actuators == base.scene.entities['robot'].articulation.actuators
        assert cfg.rewards['hd_trunk_balance'].weight < 0
        for name in ('foot_clearance', 'foot_swing_height'):
            assert cfg.rewards[name].params['target_height'] == target
            assert cfg.rewards[name].weight < 0


def test_posture_metrics_keep_signed_pitch_and_per_foot_clearance():
    r = posture_metrics([[-25,.12,.02,.03], [25,.07,.01,.02]])
    assert r['posture_metrics_valid']
    assert r['mean_pitch_deg'] == 0
    assert r['pitch_over_20deg_fraction'] == 1
    assert r['low_trunk_fraction'] == .5
    assert r['foot_clearance_max_m'] == [.02,.03]
    assert not posture_metrics([])['posture_metrics_valid']


def test_low_voltage_stress_requires_explicit_simulation_opt_in():
    with pytest.raises(ValueError, match='explicit simulation'):
        load_replay_model(6.4, bam_reference=True)
    _, _, motor = load_replay_model(6.4, bam_reference=True, voltage_extrapolation=True)
    assert motor.controller.model.actuator.vin == 6.4


def test_focus_and_robust_keep_actions_and_physics_profile():
    base = make_xgo_bam_env_cfg(repair_variant='balance_lift')
    for variant in ('lift_focus', 'lift_robust'):
        cfg = make_xgo_bam_env_cfg(repair_variant=variant)
        assert cfg.actions == base.actions
        assert cfg.observations == base.observations
        assert cfg.rewards['foot_swing_height'].params['target_height'] == .025
        assert cfg.rewards['foot_swing_height'].weight == -2.5
        actuator = cfg.scene.entities['robot'].articulation.actuators[0]
        assert actuator.json_path == base.scene.entities['robot'].articulation.actuators[0].json_path
        if variant == 'lift_robust':
            assert actuator.vin_range == (7.,8.)
            assert cfg.events['reset_base'].params['pose_range']['pitch'][0] < 0
            assert cfg.events['reset_base'].params['pose_range']['roll'][1] > 0
        else:
            assert cfg.events == base.events


def test_pitch_robust_keeps_25mm_and_both_push_directions():
    cfg = make_xgo_bam_env_cfg(repair_variant='pitch_robust')
    assert cfg.rewards['foot_swing_height'].params['target_height'] == .025
    assert cfg.rewards['foot_swing_height'].weight == make_xgo_bam_env_cfg(repair_variant='balance_lift').rewards['foot_swing_height'].weight
    assert cfg.events['push_robot'].params['velocity_range']['pitch'] == (-1.2,1.2)
    assert cfg.events['reset_base'].params['pose_range']['pitch'][0] < 0
    body = make_xgo_bam_env_cfg(repair_variant='pitch_body')
    assert body.actions == cfg.actions and body.observations == cfg.observations
    assert body.rewards['foot_swing_height'].params['target_height'] == .025
    assert body.events['push_robot'].func.__name__ == 'hd_sagittal_push'


def test_recovery_keeps_policy_contract_but_not_fall_termination():
    walk = make_xgo_bam_env_cfg(repair_variant='balance_lift')
    cfg = make_xgo_bam_env_cfg(repair_variant='recovery')
    from dataclasses import replace
    assert cfg.actions['joint_pos'].reset_from_joint_state
    assert replace(cfg.actions['joint_pos'], reset_from_joint_state=False) == walk.actions['joint_pos']
    assert cfg.observations == walk.observations
    assert 'fell_over' not in cfg.terminations
    assert 'nan_state' in cfg.terminations
    assert not cfg.curriculum
    assert cfg.commands['twist'].ranges.lin_vel_x == (0.,0.)
    reset = cfg.events['recovery_reset'].params
    assert reset['prone_prob'] == .4 and reset['face_down_prob'] == .5
    assert cfg.scene.entities['robot'].articulation.actuators == walk.scene.entities['robot'].articulation.actuators
    models = [c.scene.entities['robot'].build().spec.compile() for c in (cfg,walk)]
    assert sum(models[0].geom_contype == 1) > sum(models[1].geom_contype == 1)


def test_recovery_requires_sustained_feet_support_not_height_alone():
    assert recovery_success([[1,5,.115,2,0,0]]*100)
    assert not recovery_success([[1,5,.115,2,0,0]]*99)
    assert not recovery_success([[1,40,.115,2,0,0]]*100)
    assert not recovery_success([[1,5,.115,0,0,0]]*100)
    assert not recovery_success([[1,5,.115,2,1,0]]*100)


def test_recovery_support_only_changes_endpoint_reward_and_sensor(monkeypatch):
    from mjlab_microduck.tasks import mdp
    base = make_xgo_bam_env_cfg(repair_variant='recovery')
    cfg = make_xgo_bam_env_cfg(repair_variant='recovery_support')
    assert cfg.actions == base.actions and cfg.observations == base.observations
    assert cfg.events == base.events and cfg.commands == base.commands
    assert len(cfg.scene.sensors) == len(base.scene.sensors) + 1
    monkeypatch.setattr(mdp, 'standing_composite_score', lambda env, **kwargs: torch.ones(3))
    scene = {'feet_ground_contact': SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1,1],[1,1],[1,0]]))),
             'recovery_nonfeet_contact': SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0],[1],[0]])))}
    assert mdp.hd_recovery_standing_score(SimpleNamespace(scene=scene)).tolist() == [1.,0.,0.]


def test_sagittal_push_rotates_with_robot_heading():
    import math
    from mjlab_microduck.tasks.mdp import hd_sagittal_push
    class Robot:
        def __init__(self):
            s = math.sqrt(.5)
            self.data = SimpleNamespace(root_link_quat_w=torch.tensor([[s,0.,0.,s]]),
                                        root_link_vel_w=torch.zeros(1,6))
        def write_root_link_velocity_to_sim(self, velocity, env_ids):
            self.velocity = velocity
    robot = Robot()
    env = SimpleNamespace(scene={'robot':robot}, num_envs=1, device='cpu')
    hd_sagittal_push(env, None, linear_range=(.25,.25), angular_range=(1.2,1.2))
    v = robot.velocity[0]
    assert abs(v[0]) < 1e-6 and abs(v[2]) < 1e-6 and abs(v[4]) < 1e-6
    assert torch.isclose(abs(v[1]),torch.tensor(.25))
    assert torch.isclose(abs(v[3]),torch.tensor(1.2))
    assert v[1]*v[3] < 0


def test_lateral_push_uses_body_y_and_roll_without_changing_policy_contract():
    from mjlab_microduck.tasks.mdp import hd_sagittal_push

    class Robot:
        def __init__(self, count=1):
            self.data = SimpleNamespace(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
                                        root_link_vel_w=torch.zeros(count, 6))

        def write_root_link_velocity_to_sim(self, velocity, env_ids):
            self.velocity = velocity

    robot = Robot()
    env = SimpleNamespace(scene={'robot': robot}, num_envs=1, device='cpu')
    hd_sagittal_push(env, None, linear_range=(.25, .25), angular_range=(1.2, 1.2),
                     lateral_probability=1.)
    v = robot.velocity[0]
    assert v[0] == v[2] == v[4] == v[5] == 0
    assert torch.isclose(abs(v[1]), torch.tensor(.25))
    assert torch.isclose(abs(v[3]), torch.tensor(1.2))
    assert v[1] * v[3] < 0
    mixed_robot = Robot(128)
    mixed_env = SimpleNamespace(scene={'robot': mixed_robot}, num_envs=128, device='cpu')
    torch.manual_seed(7)
    hd_sagittal_push(mixed_env, None, linear_range=(.25, .25), angular_range=(1.2, 1.2),
                     lateral_probability=.5)
    mixed = mixed_robot.velocity
    lateral = mixed[:, 1] != 0
    assert lateral.any() and (~lateral).any()
    assert torch.all(mixed[lateral, 1] * mixed[lateral, 3] < 0)
    assert torch.all(mixed[~lateral, 0] * mixed[~lateral, 4] > 0)
    for variant, probability in (('head_lateral', 1.), ('head_omni', .5)):
        cfg = make_xgo_bam_env_cfg(repair_variant=variant)
        base = make_xgo_bam_env_cfg(repair_variant='head_quiet')
        assert cfg.actions == base.actions and cfg.observations == base.observations
        assert cfg.events['push_robot'].params['lateral_probability'] == probability
        assert cfg.events['reset_base'].params['pose_range']['roll'][0] < 0
        assert cfg.rewards['hd_head_motion'].weight == -.1
