# SPDX-License-Identifier: Apache-2.0
"""External HLS1910 M6 reference, not an identified HD1910M calibration.

Parameters: ReferenceDynamics/hd1910_rl, commit
326d77a1122870bdefa2c36403937502c958e69c, robot/hd1910/params/1910_m6.json.
Reuse BAM's control/friction equations, not a second implementation. The fit's
q_offset is a testbench offset; command_delay is not consumed by this controller.
Neither is a robot installation calibration. Explicit lag is in physics steps.
"""
from pathlib import Path
import os
import numpy as np
from bam.model import load_model
from bam.mujoco import MujocoController

PROFILE_PATH = Path(os.environ.get('MICRODUCK_BAM_PROFILE',
    str(Path(__file__).with_name('hd1910_1910_m6.json')))).resolve()
KP_FW = float(os.environ.get('MICRODUCK_BAM_KP', '5'))
if KP_FW not in (5., 6.):
    raise ValueError('supported M6 gains: upstream training P5 or Reference runtime P6')
TASK_ID = 'Mjlab-Velocity-Flat-MicroDuck-HD1910-XgoBam' + ('-P6-Slew' if KP_FW == 6 else '-Slew')


class XgoBamCpuController:
    def __init__(self, model, data, voltage=7.4, delay=4, profile_path=None, kp_fw=KP_FW,
                 voltage_extrapolation=False):
        minimum = 6.0 if voltage_extrapolation else 7.0
        if not np.isfinite(voltage) or not minimum <= voltage <= 8.0:
            raise ValueError('external BAM reference: 7..8 V; explicit simulation extrapolation: 6..8 V')
        fit = load_model(str(PROFILE_PATH if profile_path is None else profile_path))
        if kp_fw not in (5., 6.):
            raise ValueError('unsupported M6 firmware P')
        self.kp_fw = kp_fw
        fit.actuator.kp = kp_fw
        fit.actuator.vin = voltage
        self.controller = MujocoController(fit, [model.actuator(i).name for i in range(model.nu)], model, data)
        # BAM 62bd8ce uses this field only for its friction constraint selector
        # after construction. efc_id for FRICTION_DOF is a DOF, not a joint id;
        # they differ on a floating-base robot (freejoint has six DOFs).
        self.controller.joint_indexes = self.controller.dof_indexes.copy()
        self.model, self.data = model, data
        self.joint_ids = model.actuator_trnid[:, 0]
        self.qids = model.jnt_qposadr[self.joint_ids]
        self.vids = model.jnt_dofadr[self.joint_ids]
        self.delay = delay
        self.reset(data.qpos)

    def reset(self, qpos):
        self.controller.reset(qpos)
        self.q_target = np.array(qpos[self.qids], copy=True)
        self.controller.model.actuator.q_target_smooth = self.q_target.copy()
        self.controller.last_ts = self.data.time - self.model.opt.timestep
        self.history = []

    def set_gain(self, kp):
        # Runtime's 200 means nominal. Do not write XL330 gain 200 to Feetech.
        self.controller.model.actuator.kp = self.kp_fw * kp / 200

    def update(self):
        if self.delay not in range(3, 11):
            raise ValueError('simulation delay requires 3..10 physics steps')
        self.history.append(self.q_target.copy())
        if len(self.history) > self.delay + 1:
            self.history.pop(0)
        self.controller.q_target = self.history[0]
        self.controller.update()
