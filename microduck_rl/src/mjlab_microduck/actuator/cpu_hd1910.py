# SPDX-License-Identifier: Apache-2.0
"""NumPy-only HD1910 CPU dynamics shared by replay and the Radxa simulator."""
from pathlib import Path
import json
import math
import os
import mujoco
import numpy as np

PROFILE_PATH = Path(os.environ.get("MICRODUCK_HD1910_REFERENCE", Path(__file__).with_name("reference_hd1910_profile.json")))
PROFILE = json.loads(PROFILE_PATH.read_text())
SIM = PROFILE["simulation"]
PERF = PROFILE["performance"]


def performance_at_voltage(voltage):
    """SI limits; linear interpolation within measured knots, clamp above 7.4V.

    4.0-4.8V is allowed electrically but not performance-characterized: reject.
    """
    v = np.asarray(voltage, dtype=float)
    if not np.isfinite(v).all() or np.any(v < 4.8) or np.any(v > 8.4):
        raise ValueError("HD1910 performance model supports 4.8-8.4 V only")
    knots = [p["voltage_v"] for p in PERF]
    values = tuple(np.interp(v, knots, [p[key] for p in PERF])
                   for key in ("stall_nm", "rated_nm", "no_load_rpm"))
    return values[0], values[1], values[2] * (2 * math.pi / 60)


def clip_torque_numpy(request, velocity, voltage):
    stall, rated, speed = performance_at_voltage(voltage)
    v = np.clip(velocity, -speed * (1 + rated / stall), speed * (1 + rated / stall))
    lower = np.maximum(stall * (-1 - v / speed), -rated)
    upper = np.minimum(stall * (1 - v / speed), rated)
    return np.clip(request, lower, upper)


class Hd1910CpuController:
    """CPU MuJoCo equivalent of the training actuator, fixed supply and delay."""
    def __init__(self, model, data, voltage=7.4, delay=4):
        performance_at_voltage(voltage)
        self.model, self.data, self.voltage = model, data, voltage
        self.stiffness = SIM["stiffness_nm_per_rad"]
        self.damping = SIM["damping_nm_s_per_rad"]
        self.joint_ids = model.actuator_trnid[:, 0]
        self.qids = model.jnt_qposadr[self.joint_ids]
        self.vids = model.jnt_dofadr[self.joint_ids]
        self.backlash_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                             f"passive_{model.joint(int(j)).name}_backlash") for j in self.joint_ids]
        self.backlash_mask = np.array(self.backlash_ids) >= 0
        self.backlash_qids = model.jnt_qposadr[np.maximum(self.backlash_ids, 0)]
        self.backlash_vids = model.jnt_dofadr[np.maximum(self.backlash_ids, 0)]
        self.delay = delay
        self.q_target = np.zeros(model.nu)
        self.reset(data.qpos)

    def reset(self, qpos):
        self.q_target[:] = qpos[self.qids] + qpos[self.backlash_qids] * self.backlash_mask
        self.history = []

    def update(self):
        self.history.append(self.q_target.copy())
        if len(self.history) > self.delay + 1:
            self.history.pop(0)
        # Match mjlab DelayBuffer: missing history repeats the first command,
        # not the reset joint position. Later commands retain the full delay.
        target = self.history[0]
        q, v = self.data.qpos[self.qids], self.data.qvel[self.vids]
        encoder_q = q + self.data.qpos[self.backlash_qids] * self.backlash_mask
        encoder_v = v + self.data.qvel[self.backlash_vids] * self.backlash_mask
        request = self.stiffness * (target - encoder_q) - self.damping * encoder_v
        self.data.ctrl[:] = clip_torque_numpy(request, v, self.voltage)
