"""Synthetic parameter fixtures only; no physical gain identification claim."""
from pathlib import Path
from dataclasses import asdict
import importlib.util
import json
import math
import sys
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
from mjlab.actuator.dc_actuator import DcMotorActuator
from mjlab_microduck.robot.hd1910 import (
    MotorFit, ratings, actuator_cfg, adapt_env, load_cpu_model, KT_OUTPUT_NM_PER_A,
)
from mjlab_microduck.tasks.microduck_sway_env_cfg import make_microduck_sway_env_cfg

ROOT = Path(__file__).resolve().parents[1]
FIT = MotorFit(1.,.05,.00001,.01,.001,2,.005,'synthetic test fixture, NOT physical calibration')


def test_spec_units_are_output_shaft_not_multiplied_by_320():
    spec = ratings()
    assert spec['stall_torque_nm'] == pytest.approx(1.4709975)
    assert spec['rated_torque_nm'] == pytest.approx(.36284605)
    assert spec['no_load_rad_s'] == pytest.approx(113*2*math.pi/60)
    assert KT_OUTPUT_NM_PER_A == pytest.approx(.73549875)
    assert spec['stall_current_a'] == 2.
    assert spec['rated_current_a'] == .9
    for v in (0.,8.4,math.nan):
        with pytest.raises(ValueError): ratings(v)


def test_missing_identification_not_silently_replaced_by_xl330():
    with pytest.raises(ValueError):
        MotorFit.load(ROOT/'config/hd1910_identification.json')
    cfg=actuator_cfg(FIT)
    assert cfg.effort_limit == cfg.saturation_effort == pytest.approx(1.4709975)
    assert cfg.stiffness == FIT.stiffness_nm_per_rad
    assert cfg.delay_min_lag == cfg.delay_max_lag == 2


def test_training_adapter_preserves_task_obs_and_original_cfg():
    original=make_microduck_sway_env_cfg()
    cfg=adapt_env(original,FIT)
    assert cfg.scene.entities['robot'].articulation.actuators == (actuator_cfg(FIT),)
    assert original.scene.entities['robot'].articulation.actuators != cfg.scene.entities['robot'].articulation.actuators
    assert cfg.observations == original.observations
    assert cfg.rewards == original.rewards
    assert cfg.decimation == original.decimation
    assert 'randomize_joint_friction' not in cfg.events
    assert 'expand_bam_friction_fields' in original.events


def test_cpu_torque_matches_mjlab_speed_envelope():
    model,data,ctrl,names=load_cpu_model(
        ROOT/'src/mjlab_microduck/robot/microduck/scene_walk.xml',FIT)
    assert len(names)==14
    ctrl.reset(data.qpos)
    spec=ratings()
    dc=DcMotorActuator.__new__(DcMotorActuator)
    dc.saturation_effort=torch.full((14,),spec['stall_torque_nm'],dtype=torch.float64)
    dc.force_limit=dc.saturation_effort.clone()
    dc.velocity_limit_motor=torch.full((14,),spec['no_load_rad_s'],dtype=torch.float64)
    dc._vel_at_effort_lim=2*dc.velocity_limit_motor
    for speed in (-30.,-12.,0.,12.,30.):
        dq=np.full(14,speed)
        data.qvel[ctrl.vids]=dq
        ctrl.q_target[:]=data.qpos[ctrl.qids]+.3
        dc._joint_vel_clipped=torch.from_numpy(dq)
        expected=dc._clip_effort(torch.from_numpy(.3-dq*FIT.damping_nm_s_per_rad)).numpy()
        ctrl.update()
        np.testing.assert_allclose(data.ctrl,expected,atol=1e-12)
    mujoco.mj_resetData(model,data)
    ctrl.reset(data.qpos)
    for _ in range(200):
        ctrl.update()
        mujoco.mj_step(model,data)
    assert np.isfinite(data.qpos).all()


def test_onnx_viewer_adapter_executes_with_synthetic_fit(tmp_path):
    policy=ROOT/'policies/sway_in_place/sway_height_1500.onnx'
    if not policy.exists():
        pytest.skip('local candidate export not present')
    fit_path=tmp_path/'synthetic.json'
    fit_path.write_text(json.dumps(dict(model='HD-1910-C001',mode=4,fit=asdict(FIT))))
    sys.path.insert(0,str(ROOT/'scripts'))
    try:
        spec=importlib.util.spec_from_file_location('hd1910_test_eval',ROOT/'scripts/evaluate_step.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args=SimpleNamespace(policy=policy,hd1910_calibration=fit_path,seconds=.2,episodes=1,
                             hz=50,seed=42,viewer=False,video=None,report=None,trace=None,
                             trajectory=None,mouth_preview=False,blend_seconds=0.,actuator_lag=4)
        result=module.run(args)
        assert result['motor_profile']=='hd1910_mode4_pd_dc_approximation'
        assert result['motor_calibration_sha256']
    finally:
        sys.path.pop(0)
