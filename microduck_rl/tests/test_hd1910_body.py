"""Simulation transport must preserve the CPU replay's dynamics and joint order."""
import hashlib
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest
import mjlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_hd1910 import load_replay_model, step_control_period, PROFILE_PATH
from mjlab_microduck.sim.hd1910_body import HdWorld, HdBody, HdHandler, validate_launch
from mjlab_microduck.sim.body_server import HOME_TRUNK_Z


@pytest.fixture
def pair(tmp_path):
    model, data, motor = load_replay_model(7.4)
    path = tmp_path/'hd1910.mjb'
    mujoco.mj_saveModel(model, str(path))
    (tmp_path/'physics.json').write_text(json.dumps(dict(mujoco=mujoco.__version__,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
        voltage=7.4, delay_steps=4)))
    world = HdWorld(tmp_path)
    body = HdBody(world, 0)
    body.place(None, HOME_TRUNK_Z, 0)
    world.bodies.append(body)
    return world, body, model, data, motor


def test_cpu_dynamics_match_replay(pair):
    world, body, model, data, motor = pair
    data.qpos[:] = world.data.qpos
    mujoco.mj_forward(model, data)
    motor.reset(data.qpos)
    body.set_torque(True)
    target = np.zeros(15)
    target[body.to_wire] = motor.q_target + .02
    body.set_targets(target)
    motor.q_target[:] = target[body.to_wire]
    for _ in range(10):
        world.step(4)
        step_control_period(model, data, motor)
    np.testing.assert_allclose(world.data.qpos, data.qpos, atol=1e-10)
    np.testing.assert_allclose(world.data.qvel, data.qvel, atol=1e-10)
    frame = body.sensors()
    assert len(frame['positions']) == 15
    assert frame['positions'][9] == 0  # The gait model excludes the mouth.


def test_start_support_is_explicit_not_automatic_recovery(pair):
    world, body, *_ = pair
    body.supported_start = True
    body.set_torque(True)
    assert not body.released
    world.step(4)
    assert body.sensors()['trunk_z'] == HOME_TRUNK_Z
    handler = object.__new__(HdHandler)
    assert handler.dispatch(body, {'op':'release_start_support'}) == {}
    assert body.released


def test_state_trace_records_actual_full_qpos(pair):
    import io
    world, body, *_ = pair
    world.state_log = io.StringIO()
    world.step(4)
    row = json.loads(world.state_log.getvalue())
    assert row['sim_time'] == world.data.time
    np.testing.assert_array_equal(row['qpos'], world.data.qpos)
    assert row['enabled'] == body.torque_on


def test_simulation_accepts_native_recovery_gain_ramp(pair):
    world, body, *_ = pair
    for gain in (0, 50, 75, 125, 199, 200):
        body.set_gain(gain)
        assert body.kp == gain
    for gain in (-1, 201, float('nan')):
        with pytest.raises(ValueError):
            body.set_gain(gain)


def test_invalid_target_does_not_replace_last_target(pair):
    world, body, *_ = pair
    before = world.motor.q_target.copy()
    for target in ([0]*14, [float('nan')]*15):
        with pytest.raises(ValueError):
            body.set_targets(target)
    np.testing.assert_array_equal(world.motor.q_target, before)


@pytest.fixture
def bound_m6(tmp_path):
    from mjlab_microduck.actuator.cpu_xgoduck_bam import PROFILE_PATH as reference
    model, _, _ = load_replay_model(7.4, bam_reference=True)
    path = tmp_path/'hd1910.mjb'
    mujoco.mj_saveModel(model, str(path))
    profile = json.loads(reference.read_text())
    profile['q_offset'] = 0.
    profile['R'] = 3. # Deliberately different: controller must consume the bundle.
    profile_path = tmp_path/'motor_calibration.json'
    profile_path.write_text(json.dumps(profile))
    (tmp_path/'policy.onnx').write_bytes(b'test-policy-binding-only')
    meta = dict(bundle_schema=2, mujoco=mujoco.__version__,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        profile_file='motor_calibration.json', voltage=7.4, delay_steps=4, physics_hz=200,
        actuator_backend='xgoduck_bam_m6', policy_file='policy.onnx',
        policy_sha256=hashlib.sha256((tmp_path/'policy.onnx').read_bytes()).hexdigest())
    (tmp_path/'physics.json').write_text(json.dumps(meta))
    (tmp_path/'params.toml').write_text('[bus]\nport="sim:127.0.0.1:17801"\n'
        '[control]\nhz=50\n[policy]\nwalk="'+str(tmp_path/'policy.onnx')+'"\n')
    return tmp_path


def test_m6_bundle_consumed_without_environment_override(bound_m6):
    world = HdWorld(bound_m6)
    world.motor.q_target[:] = .1
    world.motor.update()
    np.testing.assert_allclose(world.data.ctrl, .1*5*.166*1.321738701681577*7.4*.6237611235393989/3)
    validate_launch(bound_m6, 17801)


def test_m6_p6_bundle_uses_manifest_gain_not_host_default(bound_m6):
    path = bound_m6/'physics.json'
    meta = json.loads(path.read_text())
    meta['kp_fw'] = 6.
    path.write_text(json.dumps(meta))
    world = HdWorld(bound_m6)
    world.motor.set_gain(200)
    world.motor.q_target[:] = .1
    world.motor.update()
    np.testing.assert_allclose(world.data.ctrl, .1*6*.166*1.321738701681577*7.4*.6237611235393989/3)


@pytest.mark.parametrize('name,message', [('policy.onnx','policy checksum'),
    ('motor_calibration.json','actuator profile'), ('hd1910.mjb','MJB checksum')])
def test_changed_bundle_files_rejected(bound_m6, name, message):
    with (bound_m6/name).open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match=message):
        HdWorld(bound_m6)


@pytest.mark.parametrize('old,new,message', [('hz=50','hz=25','frequency'),
    ('sim:127.0.0.1:17801','feetech:/tmp/test','simulation endpoint'),
    ('policy.onnx','other.onnx','policy path')])
def test_runtime_binding_rejected(bound_m6, old, new, message):
    path = bound_m6/'params.toml'
    path.write_text(path.read_text().replace(old,new))
    with pytest.raises(ValueError, match=message):
        validate_launch(bound_m6,17801)
