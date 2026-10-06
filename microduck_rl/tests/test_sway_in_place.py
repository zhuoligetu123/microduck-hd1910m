import math
from types import SimpleNamespace

import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_sway_env_cfg import make_microduck_sway_env_cfg, make_microduck_sway_head_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


class Commands:
    def __init__(self, phase):
        self.command = torch.stack((phase.cos(), phase.sin(), phase*0), dim=1)

    def get_command(self, name):
        return self.command


def test_roll_tracking_sign_and_phase():
    phase = torch.tensor([0., .25, .5, .75])*2*math.pi
    roll = math.radians(8)*phase.sin()
    gravity = torch.stack((roll*0, -roll.sin(), -roll.cos()), dim=1)
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))},
                          command_manager=Commands(phase))
    reward = mdp.sway_roll_tracking(env, math.radians(8), math.radians(4))
    assert torch.allclose(reward, roll.cos().square(), atol=1e-6)
    gravity[:, 1] *= -1
    wrong = mdp.sway_roll_tracking(env, math.radians(8), math.radians(4))
    assert (wrong[[1,3]] < .001).all()


def test_gaze_uses_world_camera_forward_axis():
    pitch = math.radians(10)
    # Negative rotation about world Y tilts the camera's +X axis upward.
    angles = torch.tensor([-pitch, math.radians(70)])
    q = torch.stack(((angles/2).cos(), angles*0, (angles/2).sin(), angles*0), dim=1)
    cfg = SimpleNamespace(name='robot', site_ids=[0])
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(site_quat_w=q[:,None,:]))})
    error = mdp.head_gaze_error(env, pitch, cfg)
    assert error[0] < 1e-8
    assert error[1] > 1.


def test_independent_task_retains_native_actuation_and_contract():
    base = make_microduck_velocity_env_cfg()
    cfg = make_microduck_sway_env_cfg()
    assert cfg.commands['twist'].period == 4.
    assert math.isclose(cfg.decimation*cfg.sim.mujoco.timestep, .02)
    assert 'step_heights' not in cfg.rewards and 'step_contacts' not in cfg.rewards
    assert cfg.rewards['head_gaze_error'].weight < 0
    assert cfg.actions == base.actions
    assert list(cfg.observations['actor'].terms) == list(base.observations['actor'].terms)
    assert cfg.scene.entities['robot'].articulation.actuators == base.scene.entities['robot'].articulation.actuators
    assert cfg.terminations['fell_over'].params == base.terminations['fell_over'].params


def test_height_accounts_for_environment_origin():
    env = SimpleNamespace(scene=Scene())
    result = mdp.sway_trunk_height(env, .117, .008)
    assert torch.allclose(result, torch.tensor([1., math.exp(-4)]), atol=1e-6)


def test_head_sway_is_relative_to_body_and_phase_conditioned():
    phase = torch.tensor([0., .25, .5, .75])*2*math.pi
    yaw = math.radians(15)*phase.sin()
    heading = .8
    world = yaw + heading
    camera = torch.stack(((world/2).cos(), world*0, world*0, (world/2).sin()), dim=1)
    body = torch.tensor([math.cos(heading/2), 0., 0., math.sin(heading/2)]).repeat(4,1)
    data = SimpleNamespace(site_quat_w=camera[:,None,:], root_link_quat_w=body,
                           projected_gravity_b=torch.tensor([0.,0.,-1.]).repeat(4,1))
    env = SimpleNamespace(scene={'robot':SimpleNamespace(data=data)},command_manager=Commands(phase))
    cfg = SimpleNamespace(name='robot',site_ids=[0])
    assert torch.allclose(mdp.sway_head_yaw_tracking(env, math.radians(15), math.radians(6), cfg), torch.ones(4), atol=1e-6)
    env.command_manager = Commands(-phase)
    wrong = mdp.sway_head_yaw_tracking(env, math.radians(15), math.radians(6), cfg)
    assert (wrong[[1,3]] < .001).all()


def test_wide_sway_preserves_old_task_and_interface():
    cfg = make_microduck_sway_head_env_cfg()
    old = make_microduck_sway_env_cfg()
    assert math.isclose(cfg.rewards['sway_roll'].params['amplitude'], math.radians(12))
    assert math.isclose(old.rewards['sway_roll'].params['amplitude'], math.radians(8))
    assert math.isclose(cfg.rewards['sway_head_yaw'].params['amplitude'], math.radians(15))
    assert cfg.actions == old.actions
    assert cfg.commands['twist'].period == 2.5
    assert old.commands['twist'].period == 4.
    assert cfg.commands['head_pose'] == old.commands['head_pose']
    assert cfg.rewards['trunk_height'] == old.rewards['trunk_height']
    assert cfg.terminations == old.terminations


class Scene(dict):
    def __init__(self):
        super().__init__(robot=SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0., 0., .117], [0., 0., 1.101]]))))
        self.env_origins = torch.tensor([[0., 0., 0.], [0., 0., 1.]])
