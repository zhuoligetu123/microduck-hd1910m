"""Low-speed command distribution and repair-recipe invariants (CPU only)."""
from types import SimpleNamespace

import torch
import pytest
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import (
    make_discovery_hd1910_velocity_env_cfg, make_refined_hd1910_velocity_env_cfg,
    refine_hd1910_posture,
)
from mjlab_microduck.tasks.hd1910_suite import adapt_task
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg


def test_low_speed_buckets_never_inherit_upstream_03_floor():
    torch.manual_seed(7)
    count = 20000
    term = object.__new__(mdp.HdLowSpeedCommand)
    term._env = SimpleNamespace(device='cpu', num_envs=count)
    term.cfg = make_refined_hd1910_velocity_env_cfg().commands['twist']
    term.vel_command_b = torch.zeros(count, 3)
    term.vel_command_w = torch.zeros(count, 3)
    for name in ('is_standing_env', 'is_world_env', 'is_heading_env', 'is_forward_env'):
        setattr(term, name, torch.zeros(count, dtype=torch.bool))
    term._resample_command(torch.arange(count))
    cmd = term.command
    assert torch.all(cmd.abs() <= torch.tensor([.15,.04,.5]))
    idle = (cmd == 0).all(dim=1)
    straight = (cmd[:,0].abs() > 0) & (cmd[:,1:] == 0).all(dim=1)
    turn = (cmd[:,:2] == 0).all(dim=1) & (cmd[:,2].abs() > 0)
    for mask, fraction in ((idle,.2),(straight,.4),(turn,.3)):
        assert abs(mask.float().mean().item() - fraction) < .015
    assert .45 < (cmd[straight,0] > 0).float().mean() < .55
    assert .45 < (cmd[turn,2] > 0).float().mean() < .55
    torch.testing.assert_close(idle,term.is_standing_env)
    before = cmd.clone()
    term._update_command()
    torch.testing.assert_close(cmd,before)
    term._resample_command(torch.tensor([],dtype=torch.long))


def reward_env(command, linear, angular):
    class Commands:
        def get_command(self, name):
            return torch.tensor(command,dtype=torch.float32)
    return SimpleNamespace(command_manager=Commands(), scene={'robot':SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_b=torch.tensor(linear,dtype=torch.float32),
                             root_link_ang_vel_b=torch.tensor(angular,dtype=torch.float32)))})


def test_tracking_sign_and_error_cost_cannot_reward_stationary_or_reverse_motion():
    env = reward_env([[.1,0,.4]]*3, [[.1,0,.03],[0,0,0],[-.1,0,0]],
                     [[.5,.5,.4],[0,0,0],[0,0,-.4]])
    yaw = mdp.hd_yaw_velocity_tracking(env)
    lin = mdp.hd_planar_velocity_tracking(env)
    cost = mdp.hd_velocity_error_cost(env)
    assert yaw[0] == lin[0] == 1
    assert yaw[0] > yaw[1] > yaw[2]
    assert lin[0] > lin[1] > lin[2]
    assert cost[0] == 0 and cost[2] > cost[1] > 0
    assert torch.all(-cost <= 0)


def test_yaw_square_cost_penalizes_oscillation_not_roll_or_correct_turn():
    env = reward_env([[0,0,.4]]*4, [[0,0,0]]*4,
                     [[.5,.5,.4],[0,0,.8],[0,0,1.2],[0,0,-.4]])
    original = mdp.hd_velocity_error_cost(env)
    new = mdp.hd_velocity_error_cost(env,yaw_square_weight=.5)
    torch.testing.assert_close(new-original,torch.tensor([0.,.5,2.,2.]))
    assert new[0] == 0 and torch.all(new >= original)


def test_held_delay_covers_constant_latency_without_initial_zero_delay():
    from mjlab.utils.buffers.delay_buffer import DelayBuffer
    torch.manual_seed(42)
    buffer = DelayBuffer(min_lag=3,max_lag=6,batch_size=1024,update_period=400,per_env_phase=False)
    first = None
    for step in range(401):
        buffer.append(torch.full((1024,1),float(step)))
        delayed = buffer.compute()
        assert torch.all((buffer.current_lags >= 3) & (buffer.current_lags <= 6))
        if step == 0:
            first = buffer.current_lags.clone()
            assert set(first.tolist()) == {3,4,5,6}
        elif step < 400:
            torch.testing.assert_close(first,buffer.current_lags)
        if step > 6:
            torch.testing.assert_close(delayed[:,0],step-buffer.current_lags.float())
    buffer.reset([0,1])
    buffer.append(torch.full((1024,1),401.))
    buffer.compute()
    assert torch.all(buffer.current_lags[:2] >= 3)


def test_posture_progress_is_signed_symmetric_and_cannot_pay_for_rest():
    class Commands:
        def get_command(self,name):
            return torch.tensor([[1.,0,0],[1.,0,0],[0.,0,0],[0.,0,0],[1.,0,0],[1.,0,0]])
    data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.,0,.115],[0,0,.115],
        [0,0,.060],[0,0,.060],[0,0,.115],[0,0,.060]]),
        root_link_lin_vel_w=torch.tensor([[0.,0,-.02],[0,0,.02],[0,0,.02],
            [0,0,-.02],[0,0,0],[0,0,.02]]))
    class Scene(dict):
        terrain=SimpleNamespace(env_origins=torch.zeros(6,3))
    env=SimpleNamespace(command_manager=Commands(),scene=Scene(robot=SimpleNamespace(data=data)))
    r=mdp.hd_posture_goal_progress(env,'twist',.060,.115)
    torch.testing.assert_close(r,torch.tensor([r[0],-r[0],r[0],-r[0],0.,0.]))
    assert r[0]>0


def test_transition_resets_preserve_endpoint_probability_and_interpolate_only():
    count = 2000
    home = torch.zeros(count,14)
    class Asset:
        data = SimpleNamespace(default_joint_pos=home)
        def find_joints(self, pattern):
            return list(range(14)), []
    env = SimpleNamespace(device='cpu', scene={'robot':Asset()},
                          sim=SimpleNamespace(data=SimpleNamespace(
                              qpos=torch.zeros(count,21),qvel=torch.zeros(count,20))))
    torch.manual_seed(123)
    mdp.set_random_ground_state(env,torch.arange(count),face_down_prob=0,face_up_prob=0,
        sitting_prob=.5,standing_prob=.5,sitting_joint_overrides={2:1.},
        sitting_z_min=.06,sitting_z_max=.06,standing_z_min=.115,standing_z_max=.115,
        transition_prob=.5)
    q = env.sim.data.qpos[:,9]
    middle = (q>0)&(q<1)
    assert .45 < middle.float().mean() < .55
    assert .20 < (q==0).float().mean() < .30
    assert .20 < (q==1).float().mean() < .30
    torch.testing.assert_close(env.sim.data.qpos[:,2],.115+q*(.06-.115))
    assert torch.isfinite(env.sim.data.qpos).all()


def test_warp_stress_uses_requested_physics_without_changing_contract():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from replay_hd1910_warp import make_replay_cfg
    cfg = make_replay_cfg(20,slew=True,voltage=6.5,delay_steps=6)
    motor = cfg.scene.entities['robot'].articulation.actuators[0]
    assert motor.voltage_range == (6.5,6.5)
    assert motor.delay_min_lag == motor.delay_max_lag == 6
    assert cfg.actions['joint_pos'].max_step_rad == .10
    with pytest.raises(ValueError):
        make_replay_cfg(20,voltage=9.)


def test_mid_crouch_clearance_does_not_start_inside_floor():
    import math
    import mujoco
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from replay_hd1910 import load_replay_model, DEFAULT_POSE
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES
    model, data, motor = load_replay_model(7.4,posture=True)
    sit = DEFAULT_POSE.copy()
    for index, value in SITTING_TARGET_OVERRIDES.items():
        sit[index] = value
    for blend in np.linspace(0,1,21):
        for roll in np.radians([-8,0,8]):
            for pitch in np.radians([-8,0,8]):
                mujoco.mj_resetData(model,data)
                data.qpos[:7] = [0,0,.11+blend*(.06-.11)+.03,
                    math.cos(roll/2)*math.cos(pitch/2),math.sin(roll/2)*math.cos(pitch/2),
                    math.cos(roll/2)*math.sin(pitch/2),-math.sin(roll/2)*math.sin(pitch/2)]
                data.qpos[motor.qids] = DEFAULT_POSE + blend*(sit-DEFAULT_POSE)
                mujoco.mj_forward(model,data)
                for i in range(data.ncon):
                    contact = data.contact[i]
                    if 0 in model.geom_bodyid[contact.geom]:
                        assert contact.dist >= -1e-5


def test_old_yaw_objective_prefers_stillness_to_correct_yaw_with_sway():
    import math
    from mjlab.tasks.velocity.mdp.rewards import track_angular_velocity
    env = reward_env([[0,0,.4]]*2, [[0,0,0]]*2, [[.5,.5,.4],[0,0,0]])
    old = track_angular_velocity(env,std=math.sqrt(.08),command_name='twist')
    new = mdp.hd_yaw_velocity_tracking(env)
    assert old[0] < old[1]
    assert new[0] > new[1]


def test_refine_preserves_physics_observations_limits_and_old_recipes():
    from copy import deepcopy
    old = make_discovery_hd1910_velocity_env_cfg()
    new = make_refined_hd1910_velocity_env_cfg()
    assert old.observations == new.observations
    assert old.actions == new.actions
    expected_events = deepcopy(old.events)
    expected_events['push_robot'].params['velocity_range']['yaw'] = (0., 0.)
    assert expected_events == new.events
    assert old.terminations == new.terminations
    assert old.scene.entities == new.scene.entities
    assert old.commands['twist'].rel_forward_envs == .2
    assert new.commands['twist'].rel_forward_envs == 0
    assert old.rewards['track_angular_velocity'].func != new.rewards['track_angular_velocity'].func
    assert new.rewards['hd_velocity_error'].weight < 0


def test_posture_repair_keeps_flags_height_and_physics():
    from copy import deepcopy
    old = adapt_task(make_microduck_sitstand_env_cfg(), slew=True)
    new = refine_hd1910_posture(deepcopy(old))
    assert new.commands['twist'] == old.commands['twist']
    assert new.observations == old.observations
    assert new.actions == old.actions
    assert new.events == old.events
    assert old.rewards['posture_pose_legs'].weight == 4
    assert new.rewards['posture_stillness'].params['tilt_zero_deg'] == 25
    assert new.rewards['posture_pose_l1'].weight > 0  # self-negating term
    assert new.rewards['action_rate_l2'].weight < 0


def test_balanced_posture_preserves_seated_learning_mass(monkeypatch):
    from mjlab_microduck.tasks.microduck_hd1910_env_cfg import balance_hd1910_posture
    def blend(env, command):
        return torch.tensor([0.,.5,1.])
    def match(*args):
        return torch.ones(3)
    def cost(*args):
        return -torch.ones(3)
    monkeypatch.setattr(mdp,'_posture_blend',blend)
    monkeypatch.setattr(mdp,'posture_pose_match',match)
    monkeypatch.setattr(mdp,'posture_pose_l1',cost)
    cfg = balance_hd1910_posture(adapt_task(make_microduck_sitstand_env_cfg(),slew=True))
    score = mdp.hd_posture_pose_match(None,'twist',{},[]) * cfg.rewards['posture_pose_legs'].weight
    penalty = mdp.hd_posture_pose_l1(None,'twist',{},[]) * cfg.rewards['posture_pose_l1'].weight
    torch.testing.assert_close(score,torch.tensor([1.5,2.75,4.]))
    torch.testing.assert_close(penalty,torch.tensor([-.5,-.75,-1.]))


@pytest.mark.parametrize('kind,weight',[('velocity','40'),('posture','nan'),
                                        ('posture','inf'),('posture','0'),('posture','-1')])
def test_height_ablation_rejects_invalid_cli_without_starting_training(tmp_path,kind,weight):
    import subprocess
    import sys
    from pathlib import Path
    script=Path(__file__).parents[1]/'scripts/iterate_hd1910.py'
    output=tmp_path/'must_not_start'
    result=subprocess.run([sys.executable,str(script),'--kind',kind,
                           '--posture-height-weight',weight,'--output',str(output)],
                          capture_output=True,text=True,timeout=10)
    assert result.returncode == 2
    assert 'posture-height-weight' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('kind,steps',[('posture','6'),('velocity','2'),('velocity','7')])
def test_delay_curriculum_cannot_escape_the_existing_profile(tmp_path,kind,steps):
    import subprocess
    import sys
    from pathlib import Path
    output=tmp_path/'must_not_start'
    result=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/iterate_hd1910.py'),
                           '--kind',kind,'--motor-delay-min-steps',steps,'--output',str(output)],
                          capture_output=True,text=True,timeout=10)
    assert result.returncode == 2
    assert 'motor-delay-min-steps' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('kind,flag,value',[
    ('posture','--yaw-square-weight','1'),('velocity','--yaw-square-weight','nan'),
    ('velocity','--yaw-square-weight','-1'),('velocity','--posture-transition-prob','.5'),
    ('posture','--posture-transition-prob','1.1'),('posture','--posture-transition-prob','nan'),
    ('posture','--posture-transition-clearance','.03'),
    ('velocity','--action-rate-cost','nan'),('velocity','--action-rate-cost','-1'),
    ('posture','--action-rate-cost','.2'),
])
def test_new_training_controls_fail_before_starting(tmp_path,kind,flag,value):
    import subprocess
    import sys
    from pathlib import Path
    output = tmp_path/'must_not_start'
    result = subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/iterate_hd1910.py'),
        '--kind',kind,flag,value,'--output',str(output)],capture_output=True,text=True,timeout=10)
    assert result.returncode == 2
    assert not output.exists()


def test_intermediate_screen_rejects_other_task_before_opening_gpu(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    (tmp_path/'manifest.json').write_text(json.dumps({'kind':'posture','recipe':'refine'}))
    result = subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/screen_hd1910_checkpoint.py'),
        '--run',str(tmp_path),'--iteration','400'],capture_output=True,text=True,timeout=10)
    assert result.returncode == 2
    assert not (tmp_path/'intermediate').exists()
