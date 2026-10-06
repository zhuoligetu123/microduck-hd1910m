import math
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_step_env_cfg import make_microduck_step_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def test_alternating_lift_and_double_support():
    phase = torch.tensor([0., .25, .5, .75]) * 2 * math.pi
    command = torch.stack((phase.cos(), phase.sin(), torch.zeros(4)), dim=1)
    heights = mdp.step_height_targets(command)
    assert torch.allclose(heights, torch.tensor([[0.,0.],[.02,0.],[0.,0.],[0.,.02]]), atol=1e-6)
    assert (heights.min(dim=1).values == 0).all()


def test_contact_phase_uses_same_lift_as_height_target():
    command = torch.tensor([[0., .18, 0.]])
    env = SimpleNamespace(command_manager=Mock(), scene={
        'robot': SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([[0.,0.,-1.]]))),
        'feet_ground_contact': SimpleNamespace(data=SimpleNamespace(current_air_time=torch.tensor([[.1,0.]])))})
    env.command_manager.get_command.return_value = command
    assert mdp.step_contacts(env, lift=.025).item() == 1.
    assert mdp.step_contacts(env, lift=.02).item() == .5


def test_no_walk_or_turn_objective_and_preserved_contract():
    c = make_microduck_step_env_cfg()
    assert math.isclose(c.decimation * c.sim.mujoco.timestep, .02)
    assert list(c.observations['actor'].terms) == ['base_ang_vel','projected_gravity',
        'joint_pos','joint_vel','actions','command','head_command','body_command']
    assert c.commands['twist'].ranges.lin_vel_x == (0.,0.)
    assert c.commands['twist'].ranges.lin_vel_y == (0.,0.)
    assert c.commands['twist'].ranges.ang_vel_z == (0.,0.)
    assert 'track_linear_velocity' not in c.rewards
    assert 'standing_envs' not in c.curriculum
    assert 'push_robot' not in c.events
    assert c.rewards['step_drift'].weight < 0
    assert c.rewards['step_heights'].weight > 0


def test_stock_xl330_actuation_and_protections_preserved():
    base = make_microduck_velocity_env_cfg()
    step = make_microduck_step_env_cfg()
    assert step.actions['joint_pos'].scale == base.actions['joint_pos'].scale == 1.0
    assert step.actions['joint_pos'].clip is base.actions['joint_pos'].clip is None
    assert step.terminations['fell_over'].params == base.terminations['fell_over'].params
    assert step.rewards['upright'].weight == base.rewards['upright'].weight
    for name in ('dof_pos_limits', 'action_rate_l2', 'self_collisions'):
        assert step.rewards[name].weight == base.rewards[name].weight
    assert step.curriculum['action_rate_weight'].params == base.curriculum['action_rate_weight'].params
    old_stages = base.curriculum['head_pose_bias_weight'].params['weight_stages']
    new_stages = step.curriculum['head_pose_bias_weight'].params['weight_stages']
    assert [s['step'] for s in new_stages] == [s['step'] for s in old_stages]
    assert [s['weight'] for s in new_stages] == [3*s['weight'] for s in old_stages]
    assert step.rewards['head_pose_tracking'].params == base.rewards['head_pose_tracking'].params
    assert math.isclose(step.commands['head_pose'].ranges[0][0], math.radians(10.))
    assert all(lo == hi for lo,hi in step.commands['head_pose'].ranges)
    assert 'step_action' not in step.rewards
    assert 'step_target_limits' not in step.rewards
    actuators = step.scene.entities['robot'].articulation.actuators
    assert actuators == base.scene.entities['robot'].articulation.actuators
