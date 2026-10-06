"""HD-1910-C001 mode-4 simulation adapter, separate from stock XL330 BAM.

Datasheet A/0 (2026-09-07), pp.2-4: output-shaft ratings, not rotor ratings.
The linear torque-speed envelope is an approximation, not identified dynamics.
Physical PD gains/friction/inertia/delay MUST be supplied from identification;
firmware register values (e.g. 32/40) are not Nm/rad or Nm*s/rad.
This module never accesses a physical servo or changes current/PID registers.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

KGCM_TO_NM = .0980665
MASS_KG = .021
GEAR_RATIO = 320
BACKLASH_MAX_RAD = math.radians(.5)
KT_OUTPUT_NM_PER_A = 7.5 * KGCM_TO_NM
# Voltage, no-load RPM, stall kg.cm, rated kg.cm, stall A, rated A.
RATINGS = ((4.8,73.,9.,2.2,1.2,.5), (6.,92.,12.,3.,1.6,.69),
           (7.4,113.,15.,3.7,2.,.9))


def ratings(voltage=7.4):
    if not math.isfinite(voltage) or not 4.8 <= voltage <= 7.4:
        raise ValueError('datasheet interpolation only covers 4.8..7.4 V; no extrapolation')
    low, high = (RATINGS[0],RATINGS[1]) if voltage <= 6 else (RATINGS[1],RATINGS[2])
    u = (voltage-low[0])/(high[0]-low[0])
    rpm, stall, rated, stall_a, rated_a = [a+u*(b-a) for a,b in zip(low[1:],high[1:])]
    return dict(voltage_v=voltage, no_load_rad_s=rpm*2*math.pi/60,
                stall_torque_nm=stall*KGCM_TO_NM, rated_torque_nm=rated*KGCM_TO_NM,
                stall_current_a=stall_a, rated_current_a=rated_a)


@dataclass(frozen=True)
class MotorFit:
    stiffness_nm_per_rad: float
    damping_nm_s_per_rad: float
    armature_kg_m2: float
    frictionloss_nm: float
    viscous_damping_nm_s_per_rad: float
    delay_steps: int
    physics_dt_s: float
    evidence: str

    def __post_init__(self):
        for name in ('stiffness_nm_per_rad','damping_nm_s_per_rad','armature_kg_m2',
                     'frictionloss_nm','viscous_damping_nm_s_per_rad','physics_dt_s'):
            value = getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} requires an identified nonnegative physical value')
        if self.stiffness_nm_per_rad == 0 or self.physics_dt_s == 0:
            raise ValueError('stiffness and physics timestep must be positive')
        if type(self.delay_steps) is not int or self.delay_steps < 0 or not self.evidence:
            raise ValueError('supply measured delay and identification evidence')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['model'] != 'HD-1910-C001' or data['mode'] != 4:
            raise ValueError('expected HD-1910-C001 mode 4 calibration')
        return cls(**data['fit'])


def actuator_cfg(fit, voltage=7.4):
    from mjlab.actuator.dc_actuator import DcMotorActuatorCfg
    spec = ratings(voltage)
    return DcMotorActuatorCfg(
        target_names_expr=(r'^(?!passive_).*',),
        stiffness=fit.stiffness_nm_per_rad, damping=fit.damping_nm_s_per_rad,
        saturation_effort=spec['stall_torque_nm'], velocity_limit=spec['no_load_rad_s'],
        # Peak physical capability, not a new .9-A software current limiter.
        # Continuous/thermal suitability requires separate measured validation.
        effort_limit=spec['stall_torque_nm'],
        armature=fit.armature_kg_m2, frictionloss=fit.frictionloss_nm,
        viscous_damping=fit.viscous_damping_nm_s_per_rad,
        delay_min_lag=fit.delay_steps, delay_max_lag=fit.delay_steps)


def adapt_env(cfg, fit, voltage=7.4):
    """Adapt a copied step/sway cfg; never mutate the original XL330 recipe."""
    from copy import deepcopy
    from functools import partial
    result = deepcopy(cfg)
    if not math.isclose(result.sim.mujoco.timestep,fit.physics_dt_s):
        raise ValueError('delay calibration physics timestep differs from environment')
    result.scene.entities['robot'].articulation.actuators = (actuator_cfg(fit,voltage),)
    robot = result.scene.entities['robot']
    robot.spec_fn = partial(without_xml_actuators, robot.spec_fn)
    # These callbacks target BamActuator and silently do nothing for DC motors.
    # Do not pretend XL330 friction/voltage randomization applies to HD1910.
    for name in ('expand_bam_friction_fields','randomize_motor_gains','randomize_joint_friction'):
        result.events.pop(name,None)
    return result


def without_xml_actuators(factory):
    # BAM edits existing XML actuators; mjlab DC creates them. Remove the old
    # actuator definitions from this fresh spec, not from the shared MJCF file.
    spec = factory()
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


class CpuController:
    """Same PD and torque-speed envelope as mjlab DcMotorActuator (CPU viewer)."""
    def __init__(self, model, data, fit, voltage):
        import numpy as np
        self.model, self.data, self.fit = model,data,fit
        self.spec = ratings(voltage)
        self.qids = model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.vids = model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.q_target = np.zeros(model.nu)

    def reset(self, qpos):
        self.q_target[:] = qpos[self.qids]

    def update(self):
        import numpy as np
        q, dq = self.data.qpos[self.qids], self.data.qvel[self.vids]
        effort = self.fit.stiffness_nm_per_rad*(self.q_target-q)-self.fit.damping_nm_s_per_rad*dq
        stall, speed = self.spec['stall_torque_nm'], self.spec['no_load_rad_s']
        v = np.clip(dq,-2*speed,2*speed)
        self.data.ctrl[:] = np.clip(effort,np.maximum(stall*(-1-v/speed),-stall),
                                   np.minimum(stall*(1-v/speed),stall))


def load_cpu_model(xml_path, fit, voltage=7.4):
    import mujoco
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.option.timestep = fit.physics_dt_s
    stall = ratings(voltage)['stall_torque_nm']
    for actuator in spec.actuators:
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.gainprm[:] = 0
        actuator.gainprm[0] = 1
        actuator.biasprm[:] = 0
        actuator.forcelimited = True
        actuator.forcerange[:] = [-stall,stall]
        actuator.ctrllimited = False
        joint = spec.joint(actuator.target)
        joint.armature = fit.armature_kg_m2
        joint.frictionloss = fit.frictionloss_nm
        joint.damping[:] = fit.viscous_damping_nm_s_per_rad
    model = spec.compile()
    data = mujoco.MjData(model)
    names = [model.actuator(i).name for i in range(model.nu)]
    return model,data,CpuController(model,data,fit,voltage),names
