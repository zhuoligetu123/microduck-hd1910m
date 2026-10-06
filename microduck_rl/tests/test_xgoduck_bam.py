"""External references remain isolated from physical servo configuration."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import mjlab
import mujoco
from mjlab_microduck.actuator.cpu_xgoduck_bam import PROFILE_PATH, XgoBamCpuController
from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg, XgoBamActuator
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_hd1910 import load_replay_model, validate_metadata


def test_luwu_runtime_p6_selection_in_fresh_process():
    import os
    import subprocess
    subprocess.run([sys.executable, '-c', '''
import mjlab
from mjlab_microduck.actuator.cpu_xgoduck_bam import KP_FW, TASK_ID
from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg
assert KP_FW == 6 and TASK_ID.endswith('-P6-Slew')
assert make_xgo_bam_env_cfg().scene.entities['robot'].articulation.actuators[0].kp_fw == 6
'''], env={**os.environ, 'MICRODUCK_BAM_KP':'6'}, check=True)


def test_reference_and_contract():
    p = json.loads(PROFILE_PATH.read_text())
    assert p['model'] == 'm6' and p['actuator'] == 'sts3215'
    assert p['R'] == pytest.approx(4.910191564179625)
    cfg = make_xgo_bam_env_cfg()
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(.02)
    assert cfg.actions['joint_pos'].max_step_rad == .1
    assert 'expand_bam_friction_fields' in cfg.events
    assert cfg.scene.entities['robot'].articulation.actuators[0].kp_fw == 5
    assert cfg.scene.entities['robot'].init_state.joint_pos['.*left_hip_pitch.*'] == -.4579


def test_motion_refine_does_not_change_physics_or_baseline():
    base = make_xgo_bam_env_cfg()
    cfg = make_xgo_bam_env_cfg(motion_refine=True)
    assert 'action_rate_weight' in base.curriculum
    assert 'hd_slew_demand' not in base.rewards
    assert 'action_rate_weight' not in cfg.curriculum
    assert cfg.rewards['hd_slew_demand'].weight == -20
    assert cfg.rewards['action_rate_l2'].weight == -5
    assert cfg.scene.entities['robot'].articulation.actuators == base.scene.entities['robot'].articulation.actuators
    assert cfg.actions['joint_pos'].max_step_rad == base.actions['joint_pos'].max_step_rad


def test_locomotion_recipe_preserves_m6_and_action_contract():
    base = make_xgo_bam_env_cfg()
    cfg = make_xgo_bam_env_cfg(locomotion_refine=True)
    assert 'expand_bam_friction_fields' in cfg.events
    assert cfg.scene.entities['robot'].articulation.actuators == base.scene.entities['robot'].articulation.actuators
    assert cfg.actions['joint_pos'].max_step_rad == base.actions['joint_pos'].max_step_rad
    assert cfg.rewards['track_linear_velocity'].weight == 4.
    assert cfg.rewards['hd_velocity_error'].weight == -1.
    assert cfg.rewards['head_pose_tracking'].params == base.rewards['head_pose_tracking'].params


def test_transfer_recipe_preserves_physics_and_ramps_regularizers():
    base = make_xgo_bam_env_cfg(locomotion_refine=True)
    cfg = make_xgo_bam_env_cfg(transfer_refine=True)
    assert cfg.scene.entities['robot'].articulation.actuators == base.scene.entities['robot'].articulation.actuators
    assert cfg.actions == base.actions
    assert cfg.observations == base.observations
    assert 'expand_bam_friction_fields' in cfg.events
    assert cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] == .25
    assert cfg.rewards['head_pose_bias'].params['axis_weights'] == (1., 1., 2., 3.)
    assert cfg.rewards['hd_slew_demand'].weight == 0.
    assert cfg.curriculum['hd_slew_demand_weight'].params['weight_stages'][-1] == {'step':12000, 'weight':-10.}
    assert cfg.curriculum['head_pose_bias_weight'].params['weight_stages'][0]['weight'] > 0.


def test_cpu_delay_and_episode_reset():
    model, data, c = load_replay_model(7.4, bam_reference=True)
    assert model.nu == 14 and model.nq == 21
    c.q_target[:] = .1
    c.update()
    expected = .1 * 5 * .166 * 1.321738701681577 * 7.4 * .6237611235393989 / 4.910191564179625
    np.testing.assert_allclose(data.ctrl, expected)
    c.q_target[:] = .2
    for _ in range(4):
        c.update()
        np.testing.assert_allclose(c.controller.q_target, .1)
    c.reset(data.qpos)
    np.testing.assert_array_equal(c.controller.model.actuator.q_target_smooth, data.qpos[c.qids])
    assert not c.history
    with pytest.raises(ValueError):
        XgoBamCpuController(model, data, 8.4)


def test_warp_reset_preserves_other_environments(monkeypatch):
    captured = []
    def compute(self, cmd):
        captured.append(self._bam_model.actuator.q_target_smooth.clone())
        return torch.zeros_like(cmd.pos)
    monkeypatch.setattr(FrictionDRBamActuator, 'compute', compute)
    actuator = XgoBamActuator.__new__(XgoBamActuator)
    actuator._bam_model = SimpleNamespace(actuator=SimpleNamespace(q_target_smooth=torch.full((2,14), 9.)))
    actuator._target_unset = torch.tensor([True, False])
    actuator.compute(SimpleNamespace(pos=torch.full((2,14), .2)))
    torch.testing.assert_close(captured[0][0], torch.full((14,), .2))
    torch.testing.assert_close(captured[0][1], torch.full((14,), 9.))
    assert not actuator._target_unset.any()


def test_reject_pd_policy_in_bam_replay():
    with pytest.raises(ValueError, match='explicitly tagged'):
        validate_metadata({'calibration_sha256': hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()}, [], bam_reference=True)


def test_cpu_removes_own_friction_using_dof_not_joint_ids(monkeypatch):
    m, d, c = load_replay_model(7.4, bam_reference=True)
    c.q_target[:] = .1
    for _ in range(8):
        c.update()
        mujoco.mj_step(m, d)
    mask = d.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    assert mask.any()
    friction = np.bincount(d.efc_id[mask], weights=d.efc_force[mask], minlength=m.nv)
    expected = (-d.qfrc_bias + d.qfrc_constraint - friction)[c.vids]
    actual = []
    original = c.controller.model.compute_frictions
    def record(torque, external, velocity):
        actual.append(external.copy())
        return original(torque, external, velocity)
    monkeypatch.setattr(c.controller.model, 'compute_frictions', record)
    c.update()
    np.testing.assert_allclose(actual[0], expected)
