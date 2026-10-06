"""Reference-model integration tests; no calibration or physical robot access."""
from pathlib import Path
import hashlib
import sys
import numpy as np
import mujoco
import pytest
import torch
from types import SimpleNamespace
from mjlab.actuator.dc_actuator import DcMotorActuator
from mjlab_microduck.actuator.reference_hd1910 import (
    make_hd1910_spec, load_cpu_model, clip_torque_numpy, performance_at_voltage,
)
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import make_reference_hd1910_velocity_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'src/mjlab_microduck/robot/microduck/scene_walk.xml'


def test_mass_delta_is_45g_not_315g_and_geometry_unchanged():
    original=mujoco.MjModel.from_xml_path(str(SCENE))
    updated=make_hd1910_spec(SCENE,clear_actuators=False).compile()
    assert updated.body_mass.sum()-original.body_mass.sum() == pytest.approx(.045)
    np.testing.assert_allclose(updated.geom_pos,original.geom_pos)
    np.testing.assert_allclose(updated.jnt_axis,original.jnt_axis)
    np.testing.assert_allclose(updated.jnt_range,original.jnt_range)
    assert np.all(updated.body_inertia[1:]>0)


def test_reference_factory_preserves_policy_contract():
    cfg=make_reference_hd1910_velocity_env_cfg()
    stock=make_microduck_velocity_env_cfg()
    assert cfg.observations==stock.observations and cfg.actions==stock.actions
    assert cfg.rewards==stock.rewards
    assert cfg.decimation*cfg.sim.mujoco.timestep == pytest.approx(.02)
    assert len(cfg.scene.entities['robot'].build().spec.actuators)==14
    assert 'randomize_motor_gains' not in cfg.events


def test_cpu_delay_and_voltage_cap_are_explicit():
    model,data,controller=load_cpu_model(SCENE)
    controller.q_target[:]=.1
    controller.update()
    np.testing.assert_allclose(data.ctrl,.055)
    controller.q_target[:]=.2
    for _ in range(4):
        controller.update()
        np.testing.assert_allclose(data.ctrl,.055)
    controller.update()
    np.testing.assert_allclose(data.ctrl,.11)
    np.testing.assert_allclose(performance_at_voltage(8.4),performance_at_voltage(7.4))
    assert abs(clip_torque_numpy(100.,0.,7.4)) == pytest.approx(.36284605)
    assert model.nu==14


@pytest.mark.parametrize('lag',[0,1,4,6])
def test_cpu_command_history_matches_training_delay_buffer(lag):
    from mjlab.utils.buffers import DelayBuffer
    _,data,controller=load_cpu_model(SCENE)
    controller.delay=lag
    buffer=DelayBuffer(min_lag=lag,max_lag=lag,batch_size=1,device='cpu')
    for _ in range(2):
        controller.reset(data.qpos)
        buffer.reset()
        for target in (.1,.2,-.1,.3,0.,.1,.2,-.2,.1):
            controller.q_target[:]=target
            controller.update()
            buffer.append(torch.full((1,14),target))
            expected=buffer.compute().numpy()[0]*controller.stiffness
            np.testing.assert_allclose(data.ctrl,expected,rtol=1e-6)


def test_m3_parameters_are_preserved_not_silently_reinterpreted():
    sys.path.insert(0,str(ROOT/'scripts'))
    from audit_hd1910_m3 import audit
    result=audit(ROOT/'config/hd1910_m3_external.json')
    assert result['q_offset_deg']==pytest.approx(-2.620995,abs=1e-5)
    assert result['command_delay_ms']==pytest.approx(1.849)
    assert result['hardware_changes'] is False
    assert len(result['missing'])==3


def test_cpu_torque_envelope_matches_training_including_overspeed():
    velocity=np.linspace(-25,25,51)
    effort=np.linspace(-2,2,51)
    for voltage in (4.8,6.5,7.4,8.4):
        stall,rated,speed=performance_at_voltage(voltage)
        state=SimpleNamespace(
            saturation_effort=torch.as_tensor(stall),
            force_limit=torch.as_tensor(rated),
            velocity_limit_motor=torch.as_tensor(speed),
            _vel_at_effort_lim=torch.as_tensor(speed*(1+rated/stall)),
            _joint_vel_clipped=torch.as_tensor(velocity),
        )
        actual=DcMotorActuator._clip_effort(state,torch.as_tensor(effort)).numpy()
        np.testing.assert_allclose(actual,clip_torque_numpy(effort,velocity,voltage))


def test_replay_uses_training_solver_and_foot_contacts():
    sys.path.insert(0,str(ROOT/'scripts'))
    from replay_hd1910 import load_replay_model
    model,_,_=load_replay_model(7.4)
    assert model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.iterations==10
    assert model.geom('left_foot_collision').priority==1
    assert model.geom('right_foot_collision').priority==1
    assert model.geom('floor').contype==1


def test_viewer_waits_for_render_shutdown(monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts'))
    import replay_hd1910

    class Handle:
        closed=False
        checks=0

        def close(self):
            self.closed=True

        def _sim(self):
            assert self.closed
            self.checks+=1
            return self if self.checks<3 else None

    handle=Handle()

    def launch(*args):
        return handle

    monkeypatch.setattr(replay_hd1910.mujoco.viewer,'launch_passive',launch)
    with replay_hd1910.replay_viewer(None,None) as actual:
        assert actual is handle and not handle.closed
    assert handle.closed and handle.checks==3


@pytest.mark.parametrize('field,bad_value',[
    ('task_id','Mjlab-Velocity-Flat-MicroDuck'),
    ('calibration_sha256','wrong-profile'),
    ('joint_names','reversed-joints'),
    ('observation_names','different-observation-order'),
    ('action_scale','0.5'),
    ('default_joint_pos',','.join(['0']*14)),
])
def test_replay_rejects_wrong_policy_contract(field,bad_value):
    sys.path.insert(0,str(ROOT/'scripts'))
    from replay_hd1910 import validate_metadata,DEFAULT_POSE,PROFILE_PATH
    names=[f'joint_{i}' for i in range(14)]
    meta=dict(task_id='Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference',
              calibration_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
              joint_names=','.join(names),action_scale='1.0',
              default_joint_pos=','.join(f'{p:.3f}' for p in DEFAULT_POSE),
              observation_names='base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command')
    assert validate_metadata(meta,names)==meta['calibration_sha256']
    meta[field]=bad_value
    with pytest.raises(ValueError):
        validate_metadata(meta,names)


def test_gait_screen_rejects_fall_stationary_walking_and_short_replay():
    sys.path.insert(0,str(ROOT/'scripts'))
    from replay_hd1910 import baseline_check
    row=dict(case='forward',command=(.1,0.,0.),completed=True,no_fall=True,max_tilt_deg=15,
             mean_body_velocity_after_1s=(.1,0.,0.),rms_vx_error_after_1s=.04,
             rms_yaw_error_after_1s=.1)
    assert baseline_check(row,20)
    assert not baseline_check(row,10)
    assert not baseline_check(dict(row,no_fall=False),20)
    assert not baseline_check(dict(row,mean_body_velocity_after_1s=(0.,0.,0.)),20)
    assert not baseline_check(dict(row,max_tilt_deg=50),20)


def test_replay_observations_follow_post_integration_pose():
    sys.path.insert(0,str(ROOT/'scripts'))
    from replay_hd1910 import load_replay_model,step_control_period
    model,data,motor=load_replay_model(7.4)
    data.qpos[2]=1.
    data.qvel[3]=2.
    step_control_period(model,data,motor)
    np.testing.assert_allclose(data.xquat[model.body('trunk_base').id],data.qpos[3:7],atol=1e-12)
    assert data.time==pytest.approx(.02)


def test_extended_replay_covers_opposite_commands():
    sys.path.insert(0,str(ROOT/'scripts'))
    from replay_hd1910 import replay_cases
    original=replay_cases()
    extended=replay_cases(True)
    assert extended[:3]==original
    commands=dict(extended)
    np.testing.assert_array_equal(commands['backward'],-np.asarray(commands['forward']))
    np.testing.assert_array_equal(commands['turn_right'],-np.asarray(commands['turn']))


def test_warp_replay_uses_fixed_reference_conditions():
    sys.path.insert(0,str(ROOT/'scripts'))
    from replay_hd1910_warp import make_replay_cfg
    cfg=make_replay_cfg(30)
    assert cfg.events=={} and cfg.curriculum=={}
    assert cfg.episode_length_s==31
    assert cfg.decimation*cfg.sim.mujoco.timestep==pytest.approx(.02)
    motor=cfg.scene.entities['robot'].articulation.actuators[0]
    assert motor.voltage_range==(7.4,7.4)
    assert motor.delay_min_lag==motor.delay_max_lag==4
    assert cfg.observations['actor'].terms['base_ang_vel'].delay_max_lag==0
    assert cfg.observations['actor'].terms['projected_gravity'].delay_max_lag==0
    assert cfg.observations['actor'].terms['joint_vel'].delay_min_lag==1
