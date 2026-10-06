# SPDX-License-Identifier: Apache-2.0
# Adapted from qiaosiyi/microduck_rl_hd1910, commit 77286a243c849e04483c7cc8ff4c5f8bae4eea16.
# Local changes: explicit profile selection; original XL330 tasks stay unchanged.
"""Provisional HD1910 model: datasheet limits, explicitly uncalibrated dynamics.

No XL330 firmware model is used. This is an output-side PD approximation,
not a claim about the vendor's internal mode-4 transfer function.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import math
import warnings
import numpy as np
import mujoco
import torch
from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg

from .cpu_hd1910 import (
    PROFILE_PATH, PROFILE, SIM, PERF, performance_at_voltage,
    clip_torque_numpy, Hd1910CpuController,
)


def announce():
    warnings.warn("HD1910 PROVISIONAL: datasheet torque/speed limits and approximate "
                  "15-motor mass correction active. PD gains, friction, armature and "
                  "delay are UNCALIBRATED legacy references. Above 7.4 V performance "
                  "is capped at 7.4 V. Rated torque is the hard output cap.", stacklevel=2)


@dataclass(kw_only=True)
class Hd1910ActuatorCfg(DcMotorActuatorCfg):
    stiffness: float = SIM["stiffness_nm_per_rad"]
    damping: float = SIM["damping_nm_s_per_rad"]
    effort_limit: float = PERF[-1]["rated_nm"]
    saturation_effort: float = PERF[-1]["stall_nm"]
    velocity_limit: float = PERF[-1]["no_load_rpm"] * 2 * math.pi / 60
    armature: float = SIM["armature_kg_m2"]
    frictionloss: float = SIM["coulomb_nm"]
    viscous_damping: float = SIM["viscous_nm_s_per_rad"]
    delay_min_lag: int = SIM["delay_physics_steps"][0]
    delay_max_lag: int = SIM["delay_physics_steps"][1]
    voltage_range: tuple[float, float] = tuple(SIM["voltage_range_v"])
    calibration_status: str = PROFILE["status"]

    def __post_init__(self):
        super().__post_init__()
        lo, hi = self.voltage_range
        performance_at_voltage([lo, hi])
        if lo > hi:
            raise ValueError("voltage_range must be ordered")

    def build(self, entity, target_ids, target_names):
        return Hd1910Actuator(self, entity, target_ids, target_names)


class Hd1910Actuator(DcMotorActuator):
    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        announce()
        name_to_local = {n: i for i, n in enumerate(self.entity.joint_names)}
        ids = [name_to_local.get(f"passive_{n}_backlash") for n in self._target_names]
        self._backlash_ids = torch.tensor([i or 0 for i in ids], device=device)
        self._backlash_mask = torch.tensor([i is not None for i in ids], device=device)
        self._backlash_velocity = torch.zeros((data.nworld, len(ids)), device=device)
        lo, hi = self.cfg.voltage_range
        self.supply_voltage = torch.empty((data.nworld, 1), device=device).uniform_(lo, hi)
        # Startup-only samples: reset cannot accumulate voltage or scale factors.
        stall, rated, speed = performance_at_voltage(self.supply_voltage.cpu().numpy())
        self.saturation_effort[:] = torch.as_tensor(stall, device=device)
        self.velocity_limit_motor[:] = torch.as_tensor(speed, device=device)
        self.force_limit[:] = torch.as_tensor(rated, device=device).clamp(max=self.cfg.effort_limit)
        self.default_force_limit[:] = self.force_limit
        self._vel_at_effort_lim[:] = self.velocity_limit_motor * (1 + self.force_limit / self.saturation_effort)


    def get_command(self, data):
        cmd = super().get_command(data)
        # Preserve the existing output-side encoder assumption on backlash
        # variants. HD1910 encoder placement still needs hardware confirmation.
        offset = data.joint_pos[:, self._backlash_ids] * self._backlash_mask
        self._backlash_velocity[:] = data.joint_vel[:, self._backlash_ids] * self._backlash_mask
        return replace(cmd, pos=cmd.pos + offset)

    def compute(self, cmd):
        # PD uses link/encoder velocity; the torque-speed envelope uses the
        # motor joint velocity. Passive wheels never enter either selector.
        self._joint_vel_clipped[:] = cmd.vel
        return IdealPdActuator.compute(self, replace(cmd, vel=cmd.vel + self._backlash_velocity))


def finalize_hd1910_env(cfg):
    """Remove XL330-specific hooks from every public environment factory.

    Keep other task randomization intact. HD friction and gain uncertainty
    cannot be calibrated from the datasheet; do not relabel BAM DR as HD DR.
    """
    for key in ("expand_bam_friction_fields", "randomize_joint_friction",
                "randomize_joint_damping", "randomize_motor_gains"):
        cfg.events.pop(key, None)
    return cfg


def _rot(quat):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, quat)
    return out.reshape(3, 3)


def make_hd1910_spec(xml_path=None, clear_actuators=True):
    if xml_path is None:
        xml_path = Path(__file__).parents[1] / "robot/microduck/robot_walk.xml"
    return adapt_hd1910_spec(mujoco.MjSpec.from_file(str(xml_path)), clear_actuators)


def adapt_hd1910_spec(spec, clear_actuators=True):
    """Retain geometry, apply documented approximate mass delta to all 15 motors.

    Additional mass is a point at the compiled mesh centroid. This does NOT
    identify the HD1910 housing inertia or its rotor inertia.
    """
    model = spec.compile()
    mesh = model.mesh("xl330").id
    # Some CAD variants have coincident visual AND collision copies of each
    # motor mesh. Count the physical housing once, not once per geom role.
    housing_geoms = {}
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[i] == mesh:
            key = (int(model.geom_bodyid[i]), *np.round(model.geom_pos[i], 10),
                   *np.round(model.geom_quat[i], 10))
            housing_geoms.setdefault(key, i)
    ids = list(housing_geoms.values())
    if len(ids) != PROFILE["hardware"]["motor_count"]:
        raise ValueError(f"Expected 15 motor geoms, found {len(ids)}; inspect CAD before applying mass correction")
    delta = PROFILE["hardware"]["mass_kg"] - SIM["old_servo_nominal_mass_kg"]
    for body_id in sorted({int(model.geom_bodyid[g]) for g in ids}):
        positions = [model.geom_pos[g] for g in ids if model.geom_bodyid[g] == body_id]
        old_mass = model.body_mass[body_id]
        old_com = model.body_ipos[body_id]
        mass = old_mass + delta * len(positions)
        com = (old_mass * old_com + delta * np.sum(positions, axis=0)) / mass
        rot = _rot(model.body_iquat[body_id])
        inertia = rot @ np.diag(model.body_inertia[body_id]) @ rot.T
        def parallel(m, r):
            return m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        inertia += parallel(old_mass, old_com - com)
        for pos in positions:
            inertia += parallel(delta, pos - com)
        body = spec.body(model.body(body_id).name)
        body.mass = float(mass)
        body.ipos = com
        body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                            inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    # Datasheet specifies <=0.5 degrees TOTAL backlash. Use that upper bound
    # in backlash variants, not the old XML's +/-1 degree reference.
    half_play = math.radians(PROFILE["hardware"]["backlash_max_deg"] / 2)
    for joint in spec.joints:
        if joint.name.startswith("passive_") and joint.name.endswith("_backlash"):
            joint.range = [-half_play, half_play]
    if clear_actuators:
        for actuator in list(spec.actuators):
            spec.delete(actuator)
    return spec




def load_cpu_model(xml_path, voltage=7.4):
    spec = make_hd1910_spec(xml_path, clear_actuators=False)
    for a in spec.actuators:
        a.set_to_motor()
        a.gear = [1, 0, 0, 0, 0, 0]
        a.forcelimited = True
        a.forcerange = [-PERF[-1]["rated_nm"], PERF[-1]["rated_nm"]]
        # Position-control ctrlrange is no longer appropriate for torque input.
        a.ctrllimited = False
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE and not j.name.startswith("passive_"):
            j.armature = SIM["armature_kg_m2"]
            j.frictionloss = SIM["coulomb_nm"]
            j.damping = np.array([[SIM["viscous_nm_s_per_rad"]], [0.0], [0.0]])
    spec.option.timestep = 0.005
    model = spec.compile()
    data = mujoco.MjData(model)
    announce()
    return model, data, Hd1910CpuController(model, data, voltage)
