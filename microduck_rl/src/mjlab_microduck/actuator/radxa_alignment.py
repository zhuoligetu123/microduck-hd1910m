"""Freeze the native encoder contract, not an actuator-identification offset.

The firmware midpoint and model joint zero are different concepts. This module
checks the saved transform, but cannot verify mechanical alignment without a
measured reference pose. It never reads a serial port or changes calibration.
"""
import hashlib
import json
import math
from pathlib import Path

RAD_PER_TICK = math.tau / 4096
JOINTS = (
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll', 'mouth',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
)


def alignment_contract(path):
    raw = Path(path).read_bytes()
    installation = json.loads(raw)
    joints = installation['joints']
    if ([j['name'] for j in joints] != list(JOINTS)
            or {j['id'] for j in joints} != set(range(1, 16))
            or any(type(j['id']) is not int or type(j['direction']) is not int
                   or j['direction'] not in (-1, 1) or type(j['zero_ticks']) is not int
                   or not 0 <= j['zero_ticks'] < 4096 for j in joints)):
        raise ValueError('invalid native installation mapping')
    result = []
    for joint in joints:
        sign, zero = joint['direction'], joint['zero_ticks']
        # q_model = direction * (ticks - 2048) * RAD_PER_TICK + offset.
        offset = sign * (2048 - zero) * RAD_PER_TICK
        errors = []
        for ticks in (0, zero, 2048, 4095):
            q = sign * (ticks - 2048) * RAD_PER_TICK + offset
            errors.append(abs(round(zero + sign * q / RAD_PER_TICK) - ticks))
        if max(errors):
            raise ValueError('encoder round-trip mismatch')
        result.append(dict(**joint, model_offset_from_midpoint_rad=offset))
    return dict(schema=1, source_sha256=hashlib.sha256(raw).hexdigest(),
        formula='q_model = direction * (ticks - zero_ticks) * 2*pi/4096',
        inverse='ticks = round(zero_ticks + direction * q_model * 4096/(2*pi))',
        joints=result, roundtrip_max_error_ticks=0,
        calibration_verified=installation.get('calibration_verified') is True,
        physical_alignment_tested=False, bam_q_offset_rad=0.0,
        bam_offset_reason='Native I/O handles installation zero; no testbench offset added to model coordinates',
        policy_joint_names=[name for name in JOINTS if name != 'mouth'])


def geometry_contract(cfg):
    import mujoco
    xml = Path(__file__).parents[1] / 'robot/microduck/robot_walk.xml'
    original = mujoco.MjModel.from_xml_path(str(xml))
    adapted = cfg.scene.entities['robot'].spec_fn().compile()
    motor = cfg.scene.entities['robot'].articulation.actuators[0]
    return dict(xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        geometry='original MicroDuck; joint axes and link lengths unchanged',
        source_total_mass_kg=float(original.body_mass.sum()),
        adapted_total_mass_kg=float(adapted.body_mass.sum()),
        mass_method='15 housings with +3g each at original CAD centroids; parallel-axis inertia correction',
        housing_inertia_identified=False, physical_mass_measured=False,
        body_mass_kg=adapted.body_mass.tolist(), body_com_m=adapted.body_ipos.tolist(),
        body_inertia_kg_m2=adapted.body_inertia.tolist(),
        voltage_range_v=list(motor.vin_range), voltage_min_v=motor.vin_min,
        kp_fw=motor.kp_fw, physics_dt_s=cfg.sim.mujoco.timestep,
        delay_steps=[motor.delay_min_lag, motor.delay_max_lag],
        encoder_bias_range_rad=list(cfg.events['encoder_bias'].params['bias_range']))
