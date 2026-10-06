from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
import os


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def load(self, path, *args, **kwargs):
        load_cfg = kwargs.get('load_cfg', args[0] if args else None)
        if len(args) < 3 and kwargs.get('map_location') is None:
            kwargs['map_location'] = self.device
        result = super().load(path, *args, **kwargs)
        if os.environ.get('MICRODUCK_HD1910_WARM_START') == '1':
            self.env.unwrapped.common_step_counter = 0
            self.current_learning_iteration = 0
            # Loading optimizer state restores the parent's param-group LR.
            # An explicit warm start must honor this experiment's configured LR.
            for group in self.alg.optimizer.param_groups:
                group['lr'] = self.alg.learning_rate
            print('[HD1910] warm start: weights/normalizer retained, curriculum counters reset')
        elif load_cfg is None or load_cfg.get('iteration', False):
            # rsl_rl saves the completed iteration, but does not save env counters.
            self.current_learning_iteration += 1
            self.env.unwrapped.common_step_counter = (
                self.current_learning_iteration * self.cfg['num_steps_per_env'])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
            print(f'[MicroDuck] resume: next iteration={self.current_learning_iteration}, '
                  f'curriculum step={self.env.unwrapped.common_step_counter}')
        return result

    def export_policy_to_onnx(self, path, filename='policy.onnx', verbose=False):
        super().export_policy_to_onnx(path, filename, verbose)
        from pathlib import Path
        from mjlab_microduck.actuator.bounded_position import bound_export
        bound_export(Path(path)/filename, self.env.unwrapped.action_manager.get_term('joint_pos'))
        motor = self.env.unwrapped.cfg.scene.entities['robot'].articulation.actuators[0]
        if type(motor).__name__ == 'Hd1910ActuatorCfg':
            import hashlib
            import onnx
            from mjlab_microduck.actuator.reference_hd1910 import PROFILE_PATH
            profile = PROFILE_PATH.read_bytes()
            model = onnx.load(Path(path)/filename)
            metadata = {p.key:p.value for p in model.metadata_props}
            metadata.update(hardware_profile='HD1910M-mode4', deployment_ready='false',
                            calibration_status='external_reference_unvalidated',
                            calibration_sha256=hashlib.sha256(profile).hexdigest())
            from .mdp import SitStandCommandCfg, GroundPickPhaseCommandCfg
            command = self.env.unwrapped.cfg.commands.get('twist')
            if isinstance(command, SitStandCommandCfg):
                metadata.update(command_semantics='sit_flag_zero_zero',
                                stand_flag='0', sit_flag='1', posture_ramp_s=str(command.ramp_s))
            elif isinstance(command, GroundPickPhaseCommandCfg):
                metadata.update(command_semantics='phase_cos_sin_zero',
                                phase_period_s=str(command.period))
            metadata['control_hz'] = str(1.0 / self.env.unwrapped.step_dt)
            onnx.helper.set_model_props(model, metadata)
            onnx.save(model, Path(path)/filename)
            (Path(path)/'motor_calibration.json').write_bytes(profile)

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant
from .microduck_step_env_cfg import make_microduck_step_env_cfg, MicroduckStepRlCfg
from .microduck_sway_env_cfg import (
    make_microduck_sway_env_cfg, MicroduckSwayRlCfg,
    make_microduck_sway_head_env_cfg, MicroduckSwayHeadRlCfg,
)

register_mjlab_task(
    task_id="Mjlab-SwayHead-Flat-MicroDuck",
    env_cfg=make_microduck_sway_head_env_cfg(),
    play_env_cfg=make_microduck_sway_head_env_cfg(play=True),
    rl_cfg=MicroduckSwayHeadRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SwayInPlace-Flat-MicroDuck",
    env_cfg=make_microduck_sway_env_cfg(),
    play_env_cfg=make_microduck_sway_env_cfg(play=True),
    rl_cfg=MicroduckSwayRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StepInPlace-Flat-MicroDuck",
    env_cfg=make_microduck_step_env_cfg(),
    play_env_cfg=make_microduck_step_env_cfg(play=True),
    rl_cfg=MicroduckStepRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

# Opt-in: missing identification must not break unrelated stock task imports.
if os.environ.get("MICRODUCK_HD1910_REFERENCE"):
    from copy import deepcopy
    from .microduck_hd1910_env_cfg import make_reference_hd1910_velocity_env_cfg
    _ref_agent = deepcopy(MicroduckRlCfg)
    _ref_agent.experiment_name = "microduck_hd1910_reference"
    register_mjlab_task(
        task_id="Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference",
        env_cfg=make_reference_hd1910_velocity_env_cfg(),
        play_env_cfg=make_reference_hd1910_velocity_env_cfg(play=True),
        rl_cfg=_ref_agent,
        runner_cls=MicroduckOnPolicyRunner,
    )
    from .microduck_hd1910_env_cfg import make_bounded_hd1910_velocity_env_cfg
    _bounded_agent = deepcopy(_ref_agent)
    _bounded_agent.experiment_name = 'microduck_hd1910_bounded'
    register_mjlab_task(
        task_id='Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Bounded',
        env_cfg=make_bounded_hd1910_velocity_env_cfg(),
        play_env_cfg=make_bounded_hd1910_velocity_env_cfg(play=True),
        rl_cfg=_bounded_agent, runner_cls=MicroduckOnPolicyRunner,
    )
    from .microduck_hd1910_env_cfg import make_slew_hd1910_velocity_env_cfg
    _slew_agent = deepcopy(_bounded_agent)
    _slew_agent.experiment_name = 'microduck_hd1910_slew'
    register_mjlab_task(
        task_id='Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Slew',
        env_cfg=make_slew_hd1910_velocity_env_cfg(),
        play_env_cfg=make_slew_hd1910_velocity_env_cfg(play=True),
        rl_cfg=_slew_agent, runner_cls=MicroduckOnPolicyRunner,
    )
    from .microduck_hd1910_env_cfg import make_discovery_hd1910_velocity_env_cfg
    _discovery_agent = deepcopy(_slew_agent)
    _discovery_agent.experiment_name = 'microduck_hd1910_discovery'
    register_mjlab_task(
        task_id='Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Slew-Discovery',
        env_cfg=make_discovery_hd1910_velocity_env_cfg(),
        play_env_cfg=make_discovery_hd1910_velocity_env_cfg(play=True),
        rl_cfg=_discovery_agent, runner_cls=MicroduckOnPolicyRunner,
    )
    from .microduck_hd1910_env_cfg import make_refined_hd1910_velocity_env_cfg
    _refined_agent = deepcopy(_slew_agent)
    _refined_agent.experiment_name = 'microduck_hd1910_refine'
    register_mjlab_task(
        task_id='Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Slew-Refine',
        env_cfg=make_refined_hd1910_velocity_env_cfg(),
        play_env_cfg=make_refined_hd1910_velocity_env_cfg(play=True),
        rl_cfg=_refined_agent, runner_cls=MicroduckOnPolicyRunner,
    )
    if os.environ.get('MICRODUCK_HD1910_SUITE') == '1':
        from .hd1910_suite import register_suite
        register_suite(MicroduckOnPolicyRunner)

if _hd_fit := os.environ.get("MICRODUCK_HD1910_FIT"):
    from copy import deepcopy
    from .microduck_hd1910_env_cfg import make_hd1910_velocity_env_cfg
    _hd_agent = deepcopy(MicroduckRlCfg)
    _hd_agent.experiment_name = "microduck_hd1910_velocity"
    register_mjlab_task(
        task_id="Mjlab-Velocity-Flat-MicroDuck-HD1910",
        env_cfg=make_hd1910_velocity_env_cfg(_hd_fit),
        play_env_cfg=make_hd1910_velocity_env_cfg(_hd_fit, play=True),
        rl_cfg=_hd_agent,
        runner_cls=MicroduckOnPolicyRunner,
    )
