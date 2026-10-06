# SPDX-License-Identifier: Apache-2.0
"""Opt-in M6 hypothesis on MicroDuck geometry and its existing action contract."""
from dataclasses import dataclass
from functools import partial
import hashlib
from pathlib import Path
import torch
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator, FrictionDRBamActuatorCfg
from mjlab_microduck.actuator.cpu_xgoduck_bam import PROFILE_PATH, KP_FW, TASK_ID

SITSTAND_TASK_ID = 'Mjlab-SitStand-Flat-MicroDuck-XgoBam-P6'
ROULADE_TASK_ID = 'Mjlab-Roulade-Flat-MicroDuck-XgoBam-P6'
STEP_TASK_ID = 'Mjlab-Step-Flat-MicroDuck-XgoBam-P6'


def configure_head_bias_course(cfg, mode):
    """Single-term curriculum ablation; no observation or actuator changes."""
    if mode not in ('scheduled', 'frozen'):
        raise ValueError('unknown head bias course')
    if mode == 'frozen':
        cfg.rewards['head_pose_bias'].weight = 0.
        term = cfg.curriculum.get('head_pose_bias_weight')
        if term is not None:
            for stage in term.params['weight_stages']:
                stage['weight'] = 0.


def configure_action_rate_weight(cfg, weight):
    """Freeze this one reward and remove only its automatic schedule."""
    import math
    if not math.isfinite(weight) or weight > 0:
        raise ValueError('action rate penalty must be finite and nonpositive')
    cfg.rewards['action_rate_l2'].weight = weight
    cfg.curriculum.pop('action_rate_weight', None)


def configure_action_rate_domain(cfg, domain):
    from mjlab.envs.mdp import action_rate_l2
    from .mdp import hd_applied_action_rate_cost
    if domain not in ('applied', 'latent'):
        raise ValueError('unknown action rate domain')
    term = cfg.rewards['action_rate_l2']
    term.func = action_rate_l2 if domain == 'latent' else hd_applied_action_rate_cost
    term.params = {}


def configure_tracking_axes(cfg, mode):
    """Keep upstream stability tolerances while scaling command-axis tracking."""
    if mode not in ('coupled', 'separate', 'separate_yaw'):
        raise ValueError('unknown tracking axes mode')
    if mode == 'coupled':
        return
    from . import mdp
    from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
    upstream = make_microduck_velocity_env_cfg()
    for name, func in (('track_linear_velocity', mdp.hd_track_linear_velocity_axes),
                       ('track_angular_velocity', mdp.hd_track_angular_velocity_axes)):
        cfg.rewards[name].func = func
        cfg.rewards[name].params['stability_std'] = upstream.rewards[name].params['std']
    if mode == 'separate_yaw':
        cfg.rewards['track_angular_velocity'].params['stability_weight'] = 0.


def configure_swing_reference(cfg, mode):
    """Change landing-height measurement only, preserving its scale and weight."""
    if mode not in ('ray', 'collision'):
        raise ValueError('unknown swing reference')
    if mode == 'collision':
        from .mdp import hd_sole_swing_height
        cfg.rewards['foot_swing_height'].func = hd_sole_swing_height
        cfg.rewards['foot_swing_height'].params['tolerance'] = 0.


def configure_standing_fraction(cfg, fraction):
    if not 0. <= fraction <= 1.:
        raise ValueError('standing fraction must be within 0..1')
    cfg.commands['twist'].rel_standing_envs = fraction
    cfg.curriculum.pop('standing_envs', None)


def configure_slew_demand_weight(cfg, weight):
    import math
    from mjlab.managers import RewardTermCfg
    from .mdp import hd_slew_demand_cost
    if not math.isfinite(weight) or weight > 0:
        raise ValueError('slew demand penalty must be finite and nonpositive')
    cfg.curriculum.pop('hd_slew_demand_weight', None)
    cfg.rewards['hd_slew_demand'] = RewardTermCfg(func=hd_slew_demand_cost, weight=weight)


def configure_forward_probability(cfg, probability):
    from .mdp import HdLowSpeedCommandCfg
    if not 0. <= probability <= 1.:
        raise ValueError('forward probability must be within 0..1')
    if not isinstance(cfg.commands['twist'], HdLowSpeedCommandCfg):
        raise ValueError('forward curriculum requires disjoint low-speed command buckets')
    cfg.commands['twist'].forward_probability = probability


def configure_walking_flexion_scale(cfg, scale):
    import math
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('walking flexion scale must be finite and positive')
    params = cfg.rewards['pose'].params
    # Walking/running can share the same dictionary in the upstream recipe.
    for regime in ('std_walking', 'std_running'):
        params[regime] = dict(params[regime])
    for regime in ('std_walking', 'std_running'):
        for joint in ('.*hip_pitch.*', '.*knee.*', '.*ankle.*'):
            params[regime][joint] *= scale


def configure_walking_hip_roll_std(cfg, std):
    """Allow lateral weight transfer without relaxing the standing posture."""
    import math
    if not math.isfinite(std) or std <= 0.:
        raise ValueError('walking hip roll std must be finite and positive')
    for regime in ('std_walking', 'std_running'):
        cfg.rewards['pose'].params[regime]['.*hip_roll.*'] = std


def configure_airtime_window_shift(cfg, seconds):
    """Shift the rewarded swing interval without changing its width or weight."""
    import math
    if not math.isfinite(seconds) or not 0. <= seconds <= .2:
        raise ValueError('airtime window shift must be within 0..0.2 seconds')
    for name in ('threshold_min', 'threshold_max'):
        cfg.rewards['air_time'].params[name] += seconds


def configure_terrain_course(cfg, mode):
    """Paired generated flat/low obstacles; actor never receives a terrain scan."""
    import mjlab.terrains as terrain_gen
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    from mjlab.managers import CurriculumTermCfg
    from mjlab.tasks.velocity.mdp import terrain_levels_vel
    from .mdp import hd_sole_swing_height
    if mode not in ('flat', 'microblocks', 'microblocks12'):
        raise ValueError('unknown terrain course')
    if any(term.func is hd_sole_swing_height for term in cfg.rewards.values()):
        raise ValueError('plane-only sole rewards cannot measure obstacle clearance')
    terrains = {'flat': terrain_gen.BoxFlatTerrainCfg(proportion=.5)}
    if mode != 'flat':
        height = .012 if mode == 'microblocks12' else .006
        terrains['microblocks'] = terrain_gen.BoxRandomSpreadTerrainCfg(
            proportion=.5, num_boxes=40, box_width_range=(.06, .12),
            box_length_range=(.12, .24), box_height_range=(height, height),
            box_yaw_range=(0., 0.), platform_width=.6, add_floor=True)
    cfg.scene.terrain.terrain_type = 'generator'
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=2026, size=(2., 2.), num_rows=4, num_cols=len(terrains),
        curriculum=True, border_width=2.,
        difficulty_range=(1., 1.) if mode == 'microblocks12' else (.375, 1.),
        sub_terrains=terrains, add_lights=False)
    cfg.scene.terrain.max_init_terrain_level = 0
    for axis in ('x', 'y'):
        cfg.events['reset_base'].params['pose_range'][axis] = (-.1, .1)
    cfg.curriculum['low_terrain_levels'] = CurriculumTermCfg(
        func=terrain_levels_vel, params={'command_name': 'twist'})


def mirror_loss_config(weight):
    import math
    from copy import deepcopy
    from .symmetry import SYMMETRY_CFG
    if not math.isfinite(weight) or weight < 0.:
        raise ValueError('mirror loss weight must be finite and nonnegative')
    cfg = deepcopy(SYMMETRY_CFG)
    cfg['mirror_loss_coeff'] = weight
    cfg['use_mirror_loss'] = weight > 0.
    return cfg


def configure_tracking_mean(cfg, seconds):
    import math
    if not math.isfinite(seconds) or seconds < 0.:
        raise ValueError('tracking mean must be finite and nonnegative')
    if seconds == 0.:
        return
    from .mdp import hd_cycle_velocity_tracking
    for name, kind in (('track_linear_velocity', 'linear'), ('track_angular_velocity', 'angular')):
        cfg.rewards[name].func = hd_cycle_velocity_tracking
        cfg.rewards[name].params.update(kind=kind, mean_seconds=seconds)


def configure_straight_yaw(cfg, std):
    """Single reward ablation: EMA yaw for translation, old reward for turns/idle."""
    import math
    from .mdp import hd_cycle_velocity_tracking, hd_track_angular_velocity_axes
    if not math.isfinite(std) or std <= 0.:
        raise ValueError('straight yaw std must be finite and positive')
    term = cfg.rewards['track_angular_velocity']
    if term.func is not hd_track_angular_velocity_axes:
        raise ValueError('straight yaw requires separate axes and no all-axis tracking mean')
    term.func = hd_cycle_velocity_tracking
    term.params.update(kind='angular', mean_seconds=.4, straight_std=std)


def configure_airtime_height_gate(cfg, mode):
    if mode not in ('off', 'gentle', 'on'):
        raise ValueError('unknown airtime height gate')
    if mode != 'off':
        from .mdp import hd_sole_swing_height
        term = cfg.rewards['air_time']
        term.func = hd_sole_swing_height
        term.params.update(height_sensor_name='foot_height_scan',
                           target_height=.020, reward_air_time=True)
        if mode == 'gentle':
            term.params['air_time_quality_fraction'] = .25


def configure_feedback_age(cfg, max_steps):
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps not in range(9):
        raise ValueError('feedback age must be an integer within 0..8 control steps')
    term = cfg.observations['actor'].terms.get('joint_state')
    if term is None:
        raise ValueError('feedback age course requires coherent joint snapshots')
    term.delay_min_lag = min(term.delay_min_lag, max_steps)
    term.delay_max_lag = max_steps


def configure_bilateral_clearance_bonus(cfg, weight, target_height=.025):
    import math
    if not math.isfinite(weight) or weight < 0.:
        raise ValueError('bilateral clearance bonus must be finite and nonnegative')
    if not math.isfinite(target_height) or not .005 <= target_height <= .05:
        raise ValueError('bilateral clearance target must be within 5..50 mm')
    if weight == 0.:
        return
    from mjlab.managers import RewardTermCfg
    from .mdp import hd_sole_swing_height
    cfg.rewards['hd_bilateral_clearance_quality'] = RewardTermCfg(
        func=hd_sole_swing_height, weight=weight,
        params={**cfg.rewards['foot_swing_height'].params, 'target_height': target_height,
                'reward_bilateral': True, 'reward_bilateral_quality': True})


def configure_clearance_course(cfg, target_mm):
    """Opt-in, single-definition height course; do not alter existing recipes."""
    from .mdp import hd_sole_swing_height
    if target_mm not in (12., 15., 20., 25.):
        raise ValueError('clearance course must be 12, 15, 20, or 25 mm')
    if cfg.scene.terrain.terrain_type != 'plane':
        raise ValueError('clearance course requires a plane')
    # The ray/site velocity-weighted cost has a different zero and used to
    # penalize trajectories beyond 20 mm. Do not retain that competing target.
    cfg.rewards.pop('foot_clearance', None)
    swing = cfg.rewards['foot_swing_height']
    swing.func = hd_sole_swing_height
    for term in cfg.rewards.values():
        if term.func is hd_sole_swing_height:
            term.params.update(target_height=target_mm / 1000., tolerance=0., shortfall_only=True)
        elif 'target_height' in term.params:
            raise ValueError('unconverted height reward in clearance course')


def cap_resume_exploration(algorithm, max_std):
    """One-time fine-tuning intervention; inference means are unchanged."""
    import math
    if not math.isfinite(max_std) or max_std <= 0.:
        raise ValueError('exploration std cap must be finite and positive')
    distribution = algorithm.actor.distribution
    if distribution.std_type != 'scalar':
        raise ValueError('exploration cap requires scalar-parameterized Gaussian std')
    std = distribution.std_param
    if not torch.isfinite(std).all() or not (std > 0.).all():
        raise ValueError('parent exploration std is invalid')
    before = std.detach().cpu().tolist()
    if (std > max_std).any():
        with torch.no_grad():
            std.clamp_(max=max_std)
        # Only this parameter changed; do not replay its old Adam momentum.
        algorithm.optimizer.state.pop(std, None)
    return {'before': before, 'after': std.detach().cpu().tolist()}


def freeze_exploration(algorithm, value):
    """Fix rollout noise only; do not alter deterministic actor outputs."""
    import math
    if not math.isfinite(value) or value <= 0.:
        raise ValueError('fixed exploration std must be finite and positive')
    distribution = algorithm.actor.distribution
    if distribution.std_type != 'scalar':
        raise ValueError('fixed exploration requires scalar Gaussian std')
    std = distribution.std_param
    with torch.no_grad():
        std.fill_(value)
    std.requires_grad_(False)
    std.grad = None
    algorithm.optimizer.state.pop(std, None)


def restore_joint_snapshot_metadata(path):
    """mjlab.save attaches generic term names after runner-specific export."""
    import onnx
    model = onnx.load(path)
    metadata = {p.key: p.value for p in model.metadata_props}
    if metadata.get('joint_snapshot_training') != 'coherent_pos_vel_delay_v1':
        return
    if model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value != 61:
        raise ValueError('coherent joint snapshot must preserve the 61D input')
    metadata['observation_names'] = (
        'base_ang_vel,projected_gravity,joint_pos,joint_vel,'
        'actions,command,head_command,body_command')
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, path)


class XgoBamActuator(FrictionDRBamActuator):
    def initialize(self, *args):
        super().initialize(*args)
        self._target_unset = torch.ones(self.kp_scale.shape[0], dtype=torch.bool, device=self.kp_scale.device)
        self._bam_model.actuator.q_target_smooth = torch.zeros_like(self._prev_motor_torque)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._target_unset[slice(None) if env_ids is None else env_ids] = True

    def compute(self, cmd):
        # STS3215 has a stateful firmware goal-slew, unlike XL330. Initialize it
        # per environment after reset; never reuse a previous episode's target.
        actuator = self._bam_model.actuator
        actuator.q_target_smooth = torch.where(self._target_unset[:, None], cmd.pos, actuator.q_target_smooth)
        self._target_unset[:] = False
        return super().compute(cmd)


@dataclass(kw_only=True)
class XgoBamActuatorCfg(FrictionDRBamActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return XgoBamActuator(self, entity, target_ids, target_names)


def make_xgo_bam_env_cfg(play=False, motion_refine=False, locomotion_refine=False, transfer_refine=False,
                         repair_variant=None):
    if repair_variant == 'gait_luwu_curriculum_v20':
        from copy import deepcopy
        from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_luwu_recipe_v18')
        upstream = make_microduck_velocity_env_cfg()
        curriculum = deepcopy(upstream.curriculum['standing_envs'])
        cfg.commands['twist'].rel_standing_envs = curriculum.params['standing_stages'][0]['rel_standing_envs']
        if not play:
            cfg.curriculum['standing_envs'] = curriculum
        return cfg
    if repair_variant in ('gait_luwu_scaled_v19', 'gait_luwu_curriculum_scaled_v21',
                          'gait_luwu_linear_only_v22'):
        from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
        base = ('gait_luwu_recipe_v18' if repair_variant == 'gait_luwu_scaled_v19'
                else 'gait_luwu_curriculum_v20')
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant=base)
        upstream = make_microduck_velocity_env_cfg(play=play)
        for reward, axis in (('track_linear_velocity', 'lin_vel_x'),
                             ('track_angular_velocity', 'ang_vel_z')):
            if repair_variant == 'gait_luwu_linear_only_v22' and axis == 'ang_vel_z':
                continue
            source_range = getattr(upstream.commands['twist'].ranges, axis)
            target_range = getattr(cfg.commands['twist'].ranges, axis)
            ratio = max(abs(v) for v in target_range) / max(abs(v) for v in source_range)
            cfg.rewards[reward].params['std'] *= ratio
        return cfg
    if repair_variant == 'gait_luwu_recipe_v18':
        from copy import deepcopy
        # Isolate reward changes: retain measured payload, age and action contract.
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_payload_v8')
        upstream = make_xgo_bam_env_cfg(play=play)
        cfg.rewards = deepcopy(upstream.rewards)
        for name, term in list(cfg.curriculum.items()):
            if 'reward_name' in term.params:
                del cfg.curriculum[name]
        for name, term in upstream.curriculum.items():
            if 'reward_name' in term.params:
                cfg.curriculum[name] = deepcopy(term)
        return cfg
    if repair_variant == 'gait_weak_quality_v17':
        from mjlab.managers import RewardTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_lift_release_v16')
        cfg.rewards['hd_weak_sole_quality'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
            weight=4., params={**cfg.rewards['hd_sole_progress'].params, 'reward_bilateral_quality': True})
        return cfg
    if repair_variant == 'gait_lift_release_v16':
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_bilateral_v11')
        cfg.rewards['action_rate_l2'].weight = -.1
        cfg.rewards['hd_slew_demand'].weight = -.5
        return cfg
    if repair_variant == 'gait_cycle_yaw_v15':
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_bilateral_v11')
        cfg.rewards['hd_velocity_error'].params['yaw_average_s'] = .20
        return cfg
    if repair_variant == 'gait_bilateral_stage20_v14':
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_bilateral_lift_v13')
        for name in ('air_time', 'foot_swing_height', 'hd_sole_progress', 'hd_weak_sole_deficit'):
            cfg.rewards[name].params['target_height'] = .020
        cfg.rewards['air_time'].params['threshold_max'] = .5
        cfg.rewards['hd_weak_sole_deficit'].weight = -8.
        return cfg
    if repair_variant == 'gait_bilateral_lift_v13':
        from mjlab.managers import RewardTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_bilateral_mirror_v12')
        cfg.rewards['hd_weak_sole_deficit'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
            weight=-2., params={**cfg.rewards['hd_sole_progress'].params, 'bilateral_deficit': True})
        return cfg
    if repair_variant in ('gait_bilateral_v11', 'gait_bilateral_mirror_v12'):
        from mjlab.managers import RewardTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_sole_support_v9')
        cfg.rewards['hd_sole_progress'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
            weight=8., params={**cfg.rewards['foot_swing_height'].params,
                              'reward_bilateral': True, 'threshold_min': .10})
        return cfg
    if repair_variant in ('gait_sole_support_v9', 'gait_sole_demand_v10'):
        from mjlab.managers import RewardTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_payload_v8')
        air = cfg.rewards['air_time']
        cfg.rewards['air_time'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
            weight=air.weight, params={**cfg.rewards['foot_swing_height'].params,
                'reward_air_time': True,
                'threshold_min': air.params['threshold_min'],
                'threshold_max': air.params['threshold_max']})
        # The ray starts at a foot site, not the lowest rotated sole vertex.
        # Keep the collision-sole landing/progress rewards as the height target.
        cfg.rewards.pop('foot_clearance')
        if repair_variant == 'gait_sole_demand_v10':
            # Applied target slew/rate remain unchanged. The latent request tax
            # dominated the measured lift reward in v9; test it independently.
            cfg.rewards['hd_slew_demand'].weight = -.5
        return cfg
    if repair_variant == 'gait_payload_v8':
        from mjlab_microduck.actuator.payload_uncertainty import configure_payload_uncertainty
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_head_dc_stride_v4')
        cfg.curriculum.pop('head_pose_bias_weight')
        cfg.rewards['head_pose_bias'].weight = 5.
        if not play:
            configure_payload_uncertainty(cfg)
        return cfg
    if repair_variant == 'gait_forward_tail_v7':
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_head_dc_stride_v4')
        cfg.curriculum.pop('head_pose_bias_weight')
        cfg.rewards['head_pose_bias'].weight = 5.
        cfg.commands['twist'].forward_probability = .65
        return cfg
    if repair_variant in ('gait_timing_v6', 'gait_forward_balance_v6'):
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_head_dc_stride_v4')
        cfg.curriculum.pop('head_pose_bias_weight')
        cfg.rewards['head_pose_bias'].weight = 5.
        # Fresh hardware snapshots mostly age 10-20 ms with a 33 ms tail.
        # Use a 0-40 ms training envelope; retain 80 ms as held-out stress.
        if not play:
            cfg.observations['actor'].terms['joint_state'].delay_max_lag = 2
        if repair_variant == 'gait_forward_balance_v6':
            cfg.commands['twist'].forward_probability = .7
            cfg.rewards['hd_trunk_balance'].weight = -5.
            cfg.rewards['head_pose_bias'].params.update(
                gate_tilt_full_deg=10., gate_tilt_zero_deg=25.)
        return cfg
    if repair_variant == 'gait_head_lift_v5':
        from mjlab.managers import CurriculumTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_head_dc_stride_v4')
        cfg.curriculum.pop('head_pose_bias_weight')
        cfg.rewards['head_pose_bias'].weight = 5.
        # Retain the completed head curriculum while testing locomotion reward
        # balance. Physics, output slew and the 25 mm sole objective stay fixed.
        for name, start, end in (('track_linear_velocity', 4., 6.),
                                  ('foot_swing_height', -8., -16.),
                                  ('hd_sole_progress', 8., 16.)):
            cfg.curriculum[name + '_weight'] = CurriculumTermCfg(func=mdp.reward_weight,
                params={'reward_name': name, 'weight_stages': [
                    {'step': 0, 'weight': start},
                    {'step': 60*24, 'weight': (start + end)/2},
                    {'step': 120*24, 'weight': end}]})
        return cfg
    if repair_variant in ('gait_head_dc_v4', 'gait_head_dc_stride_v4'):
        from mjlab.managers import CurriculumTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_head_force_sole_v3')
        # Follow upstream's DC-bias recipe: do not charge unavoidable step-frequency
        # head motion. Keep pitch tracking active during modest balance corrections.
        cfg.rewards['head_pose_tracking'].params.pop('fine_std', None)
        cfg.rewards['head_pose_tracking'].params.pop('fine_weight', None)
        cfg.rewards['head_pose_bias'].params.update(
            axis_weights=(2., 4., 1., 1.5),
            gate_tilt_full_deg=15., gate_tilt_zero_deg=35.)
        cfg.curriculum['head_pose_bias_weight'] = CurriculumTermCfg(func=mdp.reward_weight,
            params={'reward_name': 'head_pose_bias', 'weight_stages': [
                {'step': 0, 'weight': 3.}, {'step': 40*24, 'weight': 4.},
                {'step': 100*24, 'weight': 5.}]})
        if repair_variant == 'gait_head_dc_stride_v4':
            # Reduce only the penalty on requested changes, not the actual target
            # slew contract. Keep the same 25 mm collision-sole height objective.
            cfg.rewards['hd_slew_demand'].weight = -4.
            cfg.rewards['hd_velocity_error'].weight = -1.2
            cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] = 2.5
        return cfg
    if repair_variant in ('gait_head_force_v3', 'gait_head_force_sole_v3'):
        from mjlab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg
        from mjlab.envs.mdp.events import apply_body_impulse
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_head_follow_v2')
        # Normal locomotion sits near 10 cm; reserve fading for genuine collapse.
        cfg.rewards['head_pose_bias'].params.update(gate_height_low=.075, gate_height_high=.10)
        cfg.rewards['head_pose_tracking'].weight = 3.
        if not play:
            cfg.events['head_impulse'] = EventTermCfg(func=apply_body_impulse, mode='step',
                params={'asset_cfg': SceneEntityCfg('robot', body_names=('jaw_soft',)),
                        'force_range': (-.6, .6), 'torque_range': (0., 0.),
                        'duration_s': (.1, .2), 'cooldown_s': (4., 7.)})
        if repair_variant == 'gait_head_force_sole_v3':
            swing = cfg.rewards['foot_swing_height']
            swing.func, swing.weight = mdp.hd_sole_swing_height, -8.
            swing.params['target_height'] = .025
            cfg.rewards['hd_sole_progress'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
                weight=8., params={**swing.params, 'reward_progress': True})
        return cfg
    if repair_variant in ('gait_head_follow_v2', 'gait_head_stride_v2'):
        from dataclasses import fields
        from mjlab.managers import RewardTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_head_balance')
        old = cfg.commands['head_pose']
        cfg.commands['head_pose'] = mdp.LocomotionHeadCommandCfg(
            **{f.name: getattr(old, f.name) for f in fields(old) if f.init})
        cfg.commands['head_pose'].resampling_time_range = (3., 6.)
        cfg.rewards['head_pose_tracking'].params.update(fine_std=.15, fine_weight=.25)
        cfg.rewards['head_pose_bias'].weight = 3.
        cfg.rewards['head_pose_bias'].params['axis_weights'] = (2., 2., 1., 1.5)
        if repair_variant == 'gait_head_stride_v2':
            cfg.rewards['hd_velocity_error'].weight = -1.2
            cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] = 2.5
            swing = cfg.rewards['foot_swing_height']
            swing.func, swing.weight = mdp.hd_sole_swing_height, -.8
            swing.params['target_height'] = .025
            cfg.rewards['hd_sole_progress'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
                weight=1., params={**swing.params, 'reward_progress': True})
        return cfg
    if repair_variant in ('gait_head_commands', 'gait_head_balance'):
        from mjlab.managers import RewardTermCfg
        from .mdp import hd_head_gaze_envelope_cost
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_coherent_age')
        if not play:
            cfg.observations['actor'].terms['joint_state'].delay_min_lag = 0
        # Exercise head commands jointly with locomotion; retain neutral commands.
        cfg.commands['head_pose'].ranges = ((-.35, .35), (-.35, .35), (-.12, .12), (-.04, .04))
        cfg.commands['head_pose'].zero_command_prob = .3
        cfg.commands['head_pose'].resampling_time_range = (2., 4.)
        cfg.rewards.pop('hd_head_upward_excess', None)
        if repair_variant == 'gait_head_balance':
            cfg.rewards['hd_trunk_balance'].weight = -3.
            cfg.rewards['head_pose_bias'].params.update(
                gate_height_low=.09, gate_height_high=.115,
                gate_tilt_full_deg=8., gate_tilt_zero_deg=22.)
            cfg.rewards['hd_head_gaze_envelope'] = RewardTermCfg(
                func=hd_head_gaze_envelope_cost, weight=-.15)
        return cfg
    if repair_variant in ('gait_joint_age', 'gait_age_only', 'gait_age_mixed', 'gait_joint_age_stress', 'gait_coherent_age', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_coherent_age'):
        base = ('gait_yaw_hold_head' if repair_variant in ('gait_joint_age', 'gait_age_mixed', 'gait_joint_age_stress', 'gait_coherent_age') else
                'gait_yaw_hold' if repair_variant == 'gait_age_only' else 'step_delay_curriculum')
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant=base)
        if repair_variant in ('gait_coherent_age', 'step_coherent_age'):
            from copy import deepcopy
            from .mdp import hd_joint_state_rel
            terms = cfg.observations['actor'].terms
            joint_state = deepcopy(terms['joint_pos'])
            joint_state.func = hd_joint_state_rel
            joint_state.params['training_noise'] = not play
            joint_state.noise = None
            joint_state.delay_min_lag = 0 if play else 1
            joint_state.delay_max_lag = 0 if play else 4
            joint_state.delay_hold_prob = 0. if play else .8
            cfg.observations['actor'].terms = {
                ('joint_state' if name == 'joint_pos' else name):
                (joint_state if name == 'joint_pos' else term)
                for name, term in terms.items() if name != 'joint_vel'}
        elif not play:
            for term_name in ('joint_pos', 'joint_vel'):
                term = cfg.observations['actor'].terms[term_name]
                term.delay_min_lag = 0 if repair_variant == 'gait_age_mixed' and term_name == 'joint_pos' else 1
                term.delay_max_lag = 4
                term.delay_hold_prob = .8
        if repair_variant == 'step_lift35_age':
            cfg.rewards['step_heights'].params['lift'] = .035
            cfg.rewards['step_contacts'].params['lift'] = .035
        if repair_variant == 'gait_joint_age_stress':
            cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] = 5.
            cfg.rewards['hd_head_upward_excess'].weight = -.4
            if not play:
                cfg.actions['joint_pos'].command_loss_probability_range = (.05, .25)
        if repair_variant == 'step_age_drift':
            from mjlab.managers import RewardTermCfg
            from .mdp import step_radial_drift_cost
            cfg.rewards['step_radial_drift'] = RewardTermCfg(
                func=step_radial_drift_cost, weight=-10.)
        return cfg
    if repair_variant == 'head_lateral_quiet':
        from mjlab.managers import RewardTermCfg
        from .mdp import hd_head_lateral_cost
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='pitch_retention')
        cfg.rewards['hd_head_lateral'] = RewardTermCfg(func=hd_head_lateral_cost, weight=-.5)
        return cfg
    if repair_variant in ('gait_yaw_only', 'gait_yaw_hold'):
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='pitch_retention')
        cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] = 1.5
        if repair_variant == 'gait_yaw_hold' and not play:
            cfg.actions['joint_pos'].command_loss_probability_range = (.02, .08)
            cfg.actions['joint_pos'].command_hold_max_steps = 2
            cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag = 8
        return cfg
    if repair_variant == 'gait_yaw_hold_head':
        from mjlab.managers import RewardTermCfg
        from .mdp import hd_head_upward_excess_cost
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='gait_yaw_hold')
        cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] = 2.
        cfg.rewards['hd_head_upward_excess'] = RewardTermCfg(
            func=hd_head_upward_excess_cost, weight=-.2)
        return cfg
    if repair_variant == 'gait_head_limit':
        from mjlab.managers import RewardTermCfg
        from .mdp import hd_head_upward_excess_cost
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='pitch_retention')
        cfg.rewards['hd_head_upward_excess'] = RewardTermCfg(
            func=hd_head_upward_excess_cost, weight=-.1)
        return cfg
    if repair_variant == 'gait_delivery':
        from mjlab.managers import RewardTermCfg
        from . import mdp
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='head_lateral_quiet')
        cfg.actions['joint_pos'].command_loss_probability_range = (.05, .20) if not play else (0., 0.)
        cfg.actions['joint_pos'].command_hold_max_steps = 3 if not play else 0
        cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag = 10 if not play else 6
        cfg.rewards['foot_swing_height'].func = mdp.hd_sole_swing_height
        cfg.rewards['foot_swing_height'].weight = -1.
        cfg.rewards['hd_sole_lift_progress'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
            weight=2., params={**cfg.rewards['foot_swing_height'].params, 'reward_progress': True})
        cfg.rewards['hd_head_upward_excess'] = RewardTermCfg(
            func=mdp.hd_head_upward_excess_cost, weight=-.15)
        return cfg
    if repair_variant in ('step', 'step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_lift35'):
        from copy import deepcopy
        from mjlab.managers import RewardTermCfg
        from .microduck_step_env_cfg import make_microduck_step_env_cfg
        from .microduck_hd1910_env_cfg import configure_hd1910_position_bounds
        from . import mdp
        cfg = make_microduck_step_env_cfg(play=play)
        reference = make_xgo_bam_env_cfg(play=play, repair_variant='head_lateral_quiet')
        cfg.scene.entities['robot'] = deepcopy(reference.scene.entities['robot'])
        configure_hd1910_position_bounds(cfg)
        cfg.actions['joint_pos'].max_step_rad = .10
        cfg.rewards['action_rate_l2'].func = mdp.hd_applied_action_rate_cost
        cfg.rewards['action_rate_l2'].params = {}
        cfg.rewards['step_heights'].params['lift'] = .025
        cfg.rewards['step_contacts'].params['lift'] = .025
        if repair_variant in ('step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum'):
            cfg.rewards['step_heights'] = RewardTermCfg(
                func=mdp.hd_step_sole_phase_error,
                weight=-6. if repair_variant in ('step_mild_right', 'step_hold_robust', 'step_delay_curriculum') else -4., params={
                    'lift': .025,
                    'side_weights': ((1., 1.5) if repair_variant in ('step_mild_right', 'step_hold_robust', 'step_delay_curriculum') else
                                     (1., 1.75) if repair_variant in ('step_right_lift', 'step_anchored') else (1., 1.))})
        if repair_variant in ('step_mild_right', 'step_hold_robust', 'step_delay_curriculum'):
            cfg.rewards['step_drift'].weight = -24.
            cfg.rewards['step_velocity'].weight = -4.
        if repair_variant == 'step_lift35':
            cfg.rewards['step_heights'].func = mdp.hd_step_sole_phase_error
            cfg.rewards['step_heights'].weight = -6.
            cfg.rewards['step_heights'].params = {'lift': .035, 'side_weights': (1., 1.5)}
            cfg.rewards['step_contacts'].params['lift'] = .035
            cfg.rewards['step_drift'].weight = -24.
            cfg.rewards['step_velocity'].weight = -4.
        if repair_variant == 'step_delay_curriculum' and not play:
            cfg.actions['joint_pos'].command_loss_probability_range = (.01, .08)
            cfg.actions['joint_pos'].command_hold_max_steps = 2
        if repair_variant == 'step_hold_robust' and not play:
            cfg.actions['joint_pos'].command_loss_probability_range = (.05, .20)
            cfg.actions['joint_pos'].command_hold_max_steps = 3
        if repair_variant == 'step_anchored':
            cfg.rewards['step_drift'].weight = -40.
            cfg.rewards['step_velocity'].weight = -5.
            if not play:
                cfg.actions['joint_pos'].command_loss_probability_range = (.05, .15)
                cfg.actions['joint_pos'].command_hold_max_steps = 3
                cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag = 10
        cfg.commands['head_pose'].ranges = ((0., 0.),)*4
        cfg.rewards['hd_head_lateral'] = RewardTermCfg(func=mdp.hd_head_lateral_cost, weight=-.5)
        return cfg
    if repair_variant in ('pitch_retention', 'pitch_retention_delay'):
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='head_omni')
        # Keep the qualified parent's reward/pose contract. Concentrate practice
        # on sagittal disturbances, retaining 20% lateral coverage.
        if not play:
            cfg.events['push_robot'].params['lateral_probability'] = .2
        if repair_variant == 'pitch_retention_delay':
            cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag = 10
        return cfg
    if repair_variant in ('head_balance', 'delay_robust'):
        cfg = make_xgo_bam_env_cfg(play=play, repair_variant='head_omni')
        if repair_variant == 'head_balance':
            # Fade the DC head target before a perturbation becomes a fall.
            cfg.rewards['head_pose_bias'].params.update(
                neutral_offset_rad=(.02618, -.02618, 0., 0.),
                gate_height_low=.10, gate_height_high=.115,
                gate_tilt_full_deg=6., gate_tilt_zero_deg=12.)
        else:
            cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag = 10
        return cfg
    if repair_variant == 'sitstand':
        return make_xgo_bam_sitstand_env_cfg(play)
    if repair_variant == 'roulade':
        return make_xgo_bam_roulade_env_cfg(play)
    from copy import deepcopy
    from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
    from .microduck_hd1910_env_cfg import configure_hd1910_position_bounds
    from .mdp import hd_applied_action_rate_cost
    from mjlab_microduck.actuator.reference_hd1910 import make_hd1910_spec
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play))
    from .mdp import hd_feet_swing_height
    cfg.rewards['foot_swing_height'].func = hd_feet_swing_height
    if locomotion_refine or transfer_refine or repair_variant:
        from .microduck_hd1910_env_cfg import make_refined_hd1910_velocity_env_cfg
        refined = make_refined_hd1910_velocity_env_cfg(play=play)
        refined.events['expand_bam_friction_fields'] = cfg.events['expand_bam_friction_fields']
        cfg = refined
    robot = cfg.scene.entities['robot']
    robot.spec_fn = partial(make_hd1910_spec, clear_actuators=False)
    robot.articulation.actuators = (XgoBamActuatorCfg(
        target_names_expr=(r'^(?!passive_).*',), json_path=str(PROFILE_PATH),
        kp_fw=KP_FW, vin_range=(7.4, 8.0), vin_drop_gain_range=(0., .2), vin_min=7.,
        delay_min_lag=3, delay_max_lag=6),)
    configure_hd1910_position_bounds(cfg)
    cfg.actions['joint_pos'].max_step_rad = .10
    cfg.rewards['action_rate_l2'].func = hd_applied_action_rate_cost
    cfg.rewards['action_rate_l2'].params = {}
    cfg.commands['twist'].ranges.lin_vel_x = (-.15, .15)
    cfg.commands['twist'].ranges.lin_vel_y = (-.04, .04)
    cfg.commands['twist'].ranges.ang_vel_z = (-.5, .5)
    if motion_refine:
        from mjlab.managers import RewardTermCfg
        from .mdp import hd_slew_demand_cost
        # Penalize both applied changes and requests hidden by the slew clamp.
        # This is a separate training experiment, never an extra runtime filter.
        cfg.curriculum.pop('action_rate_weight', None)
        cfg.rewards['action_rate_l2'].weight = -5.
        cfg.rewards['hd_slew_demand'] = RewardTermCfg(func=hd_slew_demand_cost, weight=-20.)
    if transfer_refine or repair_variant:
        from mjlab.managers import CurriculumTermCfg
        from . import mdp
        cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] = .25
        cfg.rewards['head_pose_bias'].params['axis_weights'] = (1., 1., 2., 3.)
        # Warm-start gait first, then gradually price chatter and persistent head bias.
        schedules = (
            ('action_rate_l2', ((0, -.1), (200, -1.), (500, -3.))),
            ('hd_slew_demand', ((0, 0.), (200, -2.), (500, -10.))),
            ('head_pose_bias', ((0, .25), (200, 1.), (500, 2.))),
        )
        for reward, stages in schedules:
            cfg.rewards[reward].weight = stages[0][1]
            cfg.curriculum[reward + '_weight'] = CurriculumTermCfg(
                func=mdp.reward_weight,
                params={'reward_name': reward, 'weight_stages': [
                    {'step': iteration*24, 'weight': weight} for iteration, weight in stages]})
        if repair_variant:
            if repair_variant not in ('control', 'reversal', 'yaw', 'balance', 'balance_lift',
                                       'lift_focus', 'lift_robust', 'pitch_robust', 'pitch_body', 'head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint', 'recovery', 'recovery_support', 'recovery_all'):
                raise ValueError('Unknown M6 repair variant')
            # Continue the completed B recipe without resetting its smoothing curriculum.
            for reward, stages in schedules:
                cfg.curriculum.pop(reward + '_weight')
                cfg.rewards[reward].weight = stages[-1][1]
            if repair_variant in ('reversal', 'yaw'):
                from mjlab.managers import RewardTermCfg
                cfg.rewards['hd_action_reversal'] = RewardTermCfg(
                    func=mdp.hd_action_reversal_cost, weight=-2.)
            if repair_variant == 'yaw':
                cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] = .75
            if repair_variant in ('balance', 'balance_lift', 'lift_focus', 'lift_robust', 'pitch_robust', 'pitch_body', 'head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint'):
                from mjlab.managers import RewardTermCfg
                cfg.rewards['hd_trunk_balance'] = RewardTermCfg(
                    func=mdp.hd_trunk_balance_cost, weight=-2.)
                cfg.rewards['upright'].weight = 3.
            if repair_variant in ('balance_lift', 'lift_focus', 'lift_robust', 'pitch_robust', 'pitch_body', 'head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint'):
                for name in ('foot_clearance', 'foot_swing_height'):
                    cfg.rewards[name].params['target_height'] = .025
            if repair_variant in ('lift_focus', 'lift_robust'):
                cfg.rewards['foot_clearance'].weight = -4.
                cfg.rewards['foot_swing_height'].weight = -2.5
            if repair_variant == 'lift_robust' and not play:
                import math
                pose = cfg.events['reset_base'].params['pose_range']
                pose['roll'] = (-math.radians(8), math.radians(8))
                pose['pitch'] = (-math.radians(8), math.radians(8))
                robot.articulation.actuators[0].vin_range = (7., 8.)
            if repair_variant in ('pitch_robust', 'pitch_body', 'head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint') and not play:
                import math
                pose = cfg.events['reset_base'].params['pose_range']
                pose['pitch'] = (-math.radians(12), math.radians(12))
                push = cfg.events['push_robot']
                push.interval_range_s = (2., 4.)
                push.params['velocity_range'] = {'x': (-.25, .25), 'y': (-.1, .1),
                                                  'pitch': (-1.2, 1.2)}
                robot.articulation.actuators[0].vin_range = (7., 8.)
                if repair_variant in ('pitch_body', 'head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint'):
                    push.func = mdp.hd_sagittal_push
                    push.params = {}
                if repair_variant in ('head_lateral', 'head_omni', 'head_lower', 'head_lower_joint'):
                    pose['roll'] = (-math.radians(12), math.radians(12))
                    push.params = {'lateral_probability': 1. if repair_variant == 'head_lateral' else .5}
            if repair_variant in ('head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint'):
                cfg.rewards['hd_head_motion'] = RewardTermCfg(func=mdp.hd_head_motion_cost, weight=0.)
                cfg.curriculum['hd_head_motion_weight'] = CurriculumTermCfg(func=mdp.reward_weight,
                    params={'reward_name': 'hd_head_motion', 'weight_stages': [
                        {'step': 0, 'weight': 0.}, {'step': 50*24, 'weight': -.05},
                        {'step': 150*24, 'weight': -.1}]})
                if repair_variant in ('head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint'):
                    cfg.curriculum.pop('hd_head_motion_weight')
                    cfg.rewards['hd_head_motion'].weight = -.1
            if repair_variant == 'head_lower':
                cfg.rewards['hd_head_optical_pitch'] = RewardTermCfg(
                    func=mdp.hd_head_optical_pitch_cost, weight=-.25)
                cfg.curriculum['hd_head_optical_pitch_weight'] = CurriculumTermCfg(
                    func=mdp.reward_weight, params={'reward_name': 'hd_head_optical_pitch',
                        'weight_stages': [{'step': 0, 'weight': -.25},
                                          {'step': 100*24, 'weight': -1.}]})
            if repair_variant == 'head_lower_joint':
                offset = (.02618, -.02618, 0., 0.)
                for name in ('head_pose_tracking', 'head_pose_bias'):
                    cfg.rewards[name].params['neutral_offset_rad'] = offset
            if repair_variant == 'head_sole':
                cfg.rewards['foot_swing_height'].func = mdp.hd_sole_swing_height
                cfg.rewards['foot_swing_height'].weight = -.25
                cfg.curriculum['sole_height_weight'] = CurriculumTermCfg(func=mdp.reward_weight,
                    params={'reward_name': 'foot_swing_height', 'weight_stages': [
                        {'step': 0, 'weight': -.25}, {'step': 50*24, 'weight': -.5},
                        {'step': 150*24, 'weight': -1.}]})
            if repair_variant == 'head_lift_progress':
                cfg.rewards['hd_sole_lift_progress'] = RewardTermCfg(func=mdp.hd_sole_swing_height,
                    weight=.5, params={**cfg.rewards['foot_swing_height'].params, 'reward_progress': True})
                cfg.curriculum['sole_lift_progress_weight'] = CurriculumTermCfg(func=mdp.reward_weight,
                    params={'reward_name': 'hd_sole_lift_progress', 'weight_stages': [
                        {'step': 0, 'weight': .5}, {'step': 50*24, 'weight': 2.},
                        {'step': 150*24, 'weight': 3.}]})
            if repair_variant in ('recovery', 'recovery_support', 'recovery_all'):
                configure_m6_recovery(cfg, play, support=repair_variant != 'recovery',
                                      side_prob=.33 if repair_variant == 'recovery_all' else 0.)
    return cfg


def make_xgo_bam_sitstand_env_cfg(play=False):
    """Preserve the original sit/stand task, swapping only motor physics and bounds."""
    from copy import deepcopy
    from .microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg
    from .microduck_hd1910_env_cfg import configure_hd1910_position_bounds
    from .mdp import hd_applied_action_rate_cost
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUNDCONTACT_XML
    from mjlab_microduck.actuator.reference_hd1910 import make_hd1910_spec

    cfg = deepcopy(make_microduck_sitstand_env_cfg(play=play))
    robot = cfg.scene.entities['robot']
    robot.spec_fn = partial(make_hd1910_spec, xml_path=MICRODUCK_GROUNDCONTACT_XML,
                            clear_actuators=False)
    robot.articulation.actuators = (XgoBamActuatorCfg(
        target_names_expr=(r'^(?!passive_).*',), json_path=str(PROFILE_PATH),
        kp_fw=KP_FW, vin_range=(7.4, 8.0), vin_drop_gain_range=(0., .2), vin_min=7.,
        delay_min_lag=3, delay_max_lag=6),)
    configure_hd1910_position_bounds(cfg)
    cfg.actions['joint_pos'].max_step_rad = .10
    cfg.actions['joint_pos'].reset_from_joint_state = True
    cfg.rewards['action_rate_l2'].func = hd_applied_action_rate_cost
    cfg.rewards['action_rate_l2'].params = {}
    return cfg


def make_xgo_bam_roulade_env_cfg(play=False):
    """Original contact-gated forward roll with the HD1910M M6 actuator."""
    from copy import deepcopy
    from .microduck_roulade_env_cfg import make_microduck_roulade_env_cfg
    from .microduck_hd1910_env_cfg import configure_hd1910_position_bounds
    from .mdp import hd_applied_action_rate_cost
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUNDCONTACT_XML
    from mjlab_microduck.actuator.reference_hd1910 import make_hd1910_spec

    cfg = deepcopy(make_microduck_roulade_env_cfg(play=play))
    robot = cfg.scene.entities['robot']
    robot.spec_fn = partial(make_hd1910_spec, xml_path=MICRODUCK_GROUNDCONTACT_XML,
                            clear_actuators=False)
    robot.articulation.actuators = (XgoBamActuatorCfg(
        target_names_expr=(r'^(?!passive_).*',), json_path=str(PROFILE_PATH),
        kp_fw=KP_FW, vin_range=(7.4, 8.0), vin_drop_gain_range=(0., .2), vin_min=7.,
        delay_min_lag=3, delay_max_lag=6),)
    configure_hd1910_position_bounds(cfg)
    cfg.actions['joint_pos'].max_step_rad = .10
    cfg.actions['joint_pos'].reset_from_joint_state = True
    cfg.rewards['action_rate_l2'].func = hd_applied_action_rate_cost
    cfg.rewards['action_rate_l2'].params = {}
    return cfg


def configure_m6_recovery(cfg, play, support=False, side_prob=0.):
    """Separate get-up skill; preserve walking observations and M6 action units."""
    from copy import deepcopy
    from mjlab.managers import EventTermCfg, RewardTermCfg
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG, MICRODUCK_GROUNDCONTACT_XML
    from mjlab_microduck.actuator.reference_hd1910 import make_hd1910_spec
    from .microduck_standup_env_cfg import make_microduck_standup_env_cfg
    from . import mdp
    robot = cfg.scene.entities['robot']
    cfg.actions['joint_pos'].reset_from_joint_state = True
    robot.spec_fn = partial(make_hd1910_spec, xml_path=MICRODUCK_GROUNDCONTACT_XML, clear_actuators=False)
    robot.collisions = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG.collisions)
    stand = make_microduck_standup_env_cfg(play=play)
    names = ('pose_stand_legs', 'pose_stand_l1', 'height_stand_l1',
             'upright_linear', 'upright_sharp', 'standing_composite', 'gentle_rise')
    action_rate = deepcopy(cfg.rewards['action_rate_l2'])
    action_rate.weight = -.15
    cfg.rewards = {name: deepcopy(stand.rewards[name]) for name in names}
    cfg.rewards['action_rate_l2'] = action_rate
    cfg.rewards['upright_progress'] = RewardTermCfg(func=mdp.upright_progress, weight=5.)
    cfg.rewards['height_progress'] = RewardTermCfg(func=mdp.height_progress, weight=30.)
    if support:
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
            name='recovery_nonfeet_contact',
            primary=ContactMatch(mode='body', pattern=r'^(trunk_base|hip_l(_2)?|leg(_2)?|jaw_soft)$', entity='robot'),
            secondary=ContactMatch(mode='body', pattern='terrain'), fields=('found',),
            reduce='none', num_slots=1))
        cfg.rewards['standing_composite'].func = mdp.hd_recovery_standing_score
    # A timeout is a failed attempt, not a successful recovery. Keep NaN guard.
    cfg.terminations.pop('fell_over', None)
    cfg.episode_length_s = 10.
    cfg.curriculum = {}
    cfg.events.pop('push_robot', None)
    cfg.events['recovery_reset'] = EventTermCfg(func=mdp.maybe_set_random_prone_orientation,
        mode='reset', params=dict(prone_prob=.4, face_down_prob=.5, crouch_prob=.3,
                                 prone_z_min=.05, prone_z_max=.09, side_prob=side_prob))
    cfg.commands['twist'].ranges.lin_vel_x = (0., 0.)
    cfg.commands['twist'].ranges.lin_vel_y = (0., 0.)
    cfg.commands['twist'].ranges.ang_vel_z = (0., 0.)
    cfg.commands['twist'].rel_world_envs = 0.
    cfg.commands['twist'].rel_turn_in_place_envs = 0.
    cfg.commands['head_pose'].ranges = ((0., 0.),)*4
    cfg.commands['body_pose'].ranges = ((0., 0.),)*6


def configure_voltage_domain(cfg, domain):
    """Single-factor sim transfer; low-voltage M6 remains an extrapolation."""
    if domain not in ('nominal', 'static_home'):
        raise ValueError('unknown voltage domain')
    if domain == 'static_home':
        motor = cfg.scene.entities['robot'].articulation.actuators[0]
        motor.vin_range = (6.6, 7.4)
        motor.vin_min = 6.6


def register_task(installation_path=None, motion_refine=False, locomotion_refine=False, transfer_refine=False,
                  repair_variant=None, head_bias_course=None, action_rate_weight=None,
                  tracking_axes=None, swing_reference=None, standing_fraction=None,
                  mirror_loss_weight=None, tracking_mean_seconds=None, airtime_height_gate=None,
                  feedback_age_max_steps=None, bilateral_clearance_weight=None,
                  bilateral_clearance_target=None, clearance_course_mm=None,
                  resume_exploration_std_cap=None, fixed_exploration_std=None, walking_hip_roll_std=None,
                  action_rate_domain=None,
                  airtime_window_shift=None, terrain_course=None, slew_demand_weight=None,
                  forward_probability=None, walking_flexion_scale=None, straight_yaw_std=None,
                  voltage_domain=None):
    from copy import deepcopy
    from mjlab.tasks.registry import register_mjlab_task
    from . import MicroduckOnPolicyRunner
    from .microduck_velocity_env_cfg import MicroduckRlCfg
    from .microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
    from .microduck_roulade_env_cfg import MicroduckRouladeRlCfg
    import json
    from mjlab_microduck.actuator.radxa_alignment import alignment_contract, geometry_contract
    train_cfg = make_xgo_bam_env_cfg(motion_refine=motion_refine, locomotion_refine=locomotion_refine,
                                   transfer_refine=transfer_refine, repair_variant=repair_variant)
    if voltage_domain is not None:
        configure_voltage_domain(train_cfg, voltage_domain)
    if head_bias_course is not None:
        if repair_variant not in ('gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
            raise ValueError('head bias ablation requires the paired Luwu curriculum recipes')
        configure_head_bias_course(train_cfg, head_bias_course)
    if action_rate_weight is not None:
        if repair_variant not in ('gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
            raise ValueError('action rate ablation requires the paired Luwu curriculum recipes')
        configure_action_rate_weight(train_cfg, action_rate_weight)
    if tracking_axes is not None:
        if repair_variant not in ('gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
            raise ValueError('tracking axes ablation requires paired Luwu curriculum recipes')
        configure_tracking_axes(train_cfg, tracking_axes)
    if swing_reference is not None:
        if repair_variant not in ('gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
            raise ValueError('swing reference ablation requires paired Luwu curriculum recipes')
        configure_swing_reference(train_cfg, swing_reference)
    if standing_fraction is not None:
        configure_standing_fraction(train_cfg, standing_fraction)
    if mirror_loss_weight is not None:
        if repair_variant not in ('gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
            raise ValueError('mirror loss ablation requires the Luwu gait recipes')
        mirror_loss_config(mirror_loss_weight)
    if tracking_mean_seconds is not None:
        if repair_variant not in ('gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
            raise ValueError('tracking mean ablation requires the Luwu gait recipes')
        configure_tracking_mean(train_cfg, tracking_mean_seconds)
    if straight_yaw_std is not None:
        configure_straight_yaw(train_cfg, straight_yaw_std)
    if airtime_height_gate is not None:
        if repair_variant not in ('gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
            raise ValueError('airtime height ablation requires the Luwu gait recipes')
        configure_airtime_height_gate(train_cfg, airtime_height_gate)
    if feedback_age_max_steps is not None:
        configure_feedback_age(train_cfg, feedback_age_max_steps)
    if bilateral_clearance_weight is not None:
        configure_bilateral_clearance_bonus(train_cfg, bilateral_clearance_weight,
            .025 if bilateral_clearance_target is None else bilateral_clearance_target)
    elif bilateral_clearance_target is not None:
        raise ValueError('clearance target requires an explicit bilateral bonus weight')
    if walking_hip_roll_std is not None:
        configure_walking_hip_roll_std(train_cfg, walking_hip_roll_std)
    if airtime_window_shift is not None:
        configure_airtime_window_shift(train_cfg, airtime_window_shift)
    if terrain_course is not None:
        configure_terrain_course(train_cfg, terrain_course)
    if slew_demand_weight is not None:
        configure_slew_demand_weight(train_cfg, slew_demand_weight)
    if forward_probability is not None:
        configure_forward_probability(train_cfg, forward_probability)
    if walking_flexion_scale is not None:
        configure_walking_flexion_scale(train_cfg, walking_flexion_scale)
    if action_rate_domain is not None:
        configure_action_rate_domain(train_cfg, action_rate_domain)
    if clearance_course_mm is not None:
        configure_clearance_course(train_cfg, clearance_course_mm)
    task_id = (STEP_TASK_ID if repair_variant in ('step', 'step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_lift35', 'step_coherent_age') else
               SITSTAND_TASK_ID if repair_variant == 'sitstand' else
               ROULADE_TASK_ID if repair_variant == 'roulade' else TASK_ID)
    contract = alignment_contract(installation_path) if installation_path else None
    geometry = geometry_contract(train_cfg) if contract else None
    if contract and json.loads(PROFILE_PATH.read_text())['q_offset'] != 0.0:
        raise ValueError('Radxa model coordinates require a zero testbench q_offset profile')

    class XgoReferenceRunner(MicroduckOnPolicyRunner):
        def load(self, path, *args, **kwargs):
            result = super().load(path, *args, **kwargs)
            if resume_exploration_std_cap is not None:
                change = cap_resume_exploration(self.alg, resume_exploration_std_cap)
                print('[HD1910] explicit resume exploration cap: ' + json.dumps(change))
            if fixed_exploration_std is not None:
                freeze_exploration(self.alg, fixed_exploration_std)
                print(f'[HD1910] fixed training exploration std: {fixed_exploration_std}')
            return result

        def __init__(self, env, train_cfg, log_dir=None, device='cpu', **kwargs):
            if repair_variant in ('gait_bilateral_mirror_v12', 'gait_bilateral_lift_v13', 'gait_bilateral_stage20_v14'):
                from .symmetry import SYMMETRY_CFG
                # Inject after CLI parsing; Tyro cannot serialize dict | None.
                train_cfg = deepcopy(train_cfg)
                train_cfg['algorithm']['symmetry_cfg'] = deepcopy(SYMMETRY_CFG)
            if mirror_loss_weight is not None:
                train_cfg = deepcopy(train_cfg)
                train_cfg['algorithm']['symmetry_cfg'] = mirror_loss_config(mirror_loss_weight)
            super().__init__(env, train_cfg, log_dir, device, **kwargs)
            if fixed_exploration_std is not None:
                freeze_exploration(self.alg, fixed_exploration_std)

        def save(self, path, infos=None):
            super().save(path, infos)
            _, _, onnx_path = self._get_export_paths(path)
            restore_joint_snapshot_metadata(onnx_path)

        def export_policy_to_onnx(self, path, filename='policy.onnx', verbose=False):
            super().export_policy_to_onnx(path, filename, verbose)
            import onnx
            target = Path(path) / filename
            model = onnx.load(target)
            metadata = {p.key: p.value for p in model.metadata_props}
            metadata.update(task_id=task_id, hardware_profile='HD1910-XgoBam-reference',
                calibration_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
                calibration_status='external_reference_unvalidated', deployment_ready='false',
                actuator_backend='xgoduck_bam_m6', kp_fw=str(KP_FW),
                policy_role=('step' if repair_variant in ('step', 'step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_lift35', 'step_coherent_age') else
                             'roulade' if repair_variant == 'roulade' else
                             'sitstand' if repair_variant == 'sitstand' else
                             'recovery' if repair_variant in ('recovery','recovery_support','recovery_all') else 'locomotion'),
                collision_model='groundcontact' if repair_variant in ('recovery','recovery_support','recovery_all','sitstand','roulade') else 'walk',
                kd_fw='20', kd_modelled='false', mouth_kp_fw='10',
                command_semantics=('phase_cos_sin_zero' if repair_variant in ('step', 'step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_lift35', 'step_coherent_age') else
                                   'sit_flag_zero_zero' if repair_variant == 'sitstand' else
                                   'episodic_roll_zero_command' if repair_variant == 'roulade' else 'twist_head_body'),
                control_hz='50',
                actuator_delay_physics_steps=str((
                    train_cfg.scene.entities['robot'].articulation.actuators[0].delay_min_lag,
                    train_cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag)),
                training_recipe=('m6_repair_' + repair_variant + '_v1' if repair_variant else
                                 'm6_transfer_refine_v1' if transfer_refine else
                                 'm6_locomotion_refine_v1' if locomotion_refine else
                                 'm6_motion_refine_v1' if motion_refine else 'm6_reference_v1'))
            if repair_variant in ('gait_coherent_age', 'step_coherent_age', 'gait_head_commands', 'gait_head_balance', 'gait_head_follow_v2', 'gait_head_stride_v2', 'gait_head_force_v3', 'gait_head_force_sole_v3', 'gait_head_dc_v4', 'gait_head_dc_stride_v4', 'gait_head_lift_v5', 'gait_timing_v6', 'gait_forward_balance_v6', 'gait_forward_tail_v7', 'gait_payload_v8', 'gait_sole_support_v9', 'gait_sole_demand_v10', 'gait_bilateral_v11', 'gait_bilateral_mirror_v12', 'gait_bilateral_lift_v13', 'gait_bilateral_stage20_v14', 'gait_cycle_yaw_v15', 'gait_lift_release_v16', 'gait_weak_quality_v17', 'gait_luwu_recipe_v18', 'gait_luwu_scaled_v19', 'gait_luwu_curriculum_v20', 'gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
                metadata['observation_names'] = (
                    'base_ang_vel,projected_gravity,joint_pos,joint_vel,'
                    'actions,command,head_command,body_command')
                metadata['joint_snapshot_training'] = 'coherent_pos_vel_delay_v1'
            if repair_variant in ('gait_payload_v8', 'gait_sole_support_v9', 'gait_sole_demand_v10', 'gait_bilateral_v11', 'gait_bilateral_mirror_v12', 'gait_bilateral_lift_v13', 'gait_bilateral_stage20_v14', 'gait_cycle_yaw_v15', 'gait_lift_release_v16', 'gait_weak_quality_v17', 'gait_luwu_recipe_v18', 'gait_luwu_scaled_v19', 'gait_luwu_curriculum_v20', 'gait_luwu_curriculum_scaled_v21', 'gait_luwu_linear_only_v22'):
                from mjlab_microduck.actuator.payload_uncertainty import mass_contract
                payload = mass_contract()
                metadata['mass_profile_sha256'] = payload['profile_sha256']
                metadata['mass_calibration_status'] = payload['status']
                (Path(path) / 'payload_contract.json').write_text(json.dumps(payload, indent=2)+'\n')
            if repair_variant == 'sitstand':
                metadata.update(stand_flag='0', sit_flag='1', posture_ramp_s='2.0')
            if repair_variant in ('step', 'step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_lift35', 'step_coherent_age'):
                metadata['step_period_s'] = '1.0'
            if repair_variant == 'head_lower_joint':
                metadata['head_neutral_offset_rad'] = '0.02618,-0.02618,0,0'
            if contract:
                metadata.update(installation_sha256=contract['source_sha256'],
                    joint_zero_semantics='native_zero_ticks_model_coordinates',
                    testbench_q_offset_rad='0.0')
                (Path(path) / 'joint_alignment.json').write_text(json.dumps(contract, indent=2)+'\n')
                (Path(path) / 'geometry_contract.json').write_text(json.dumps(geometry, indent=2)+'\n')
            if head_bias_course is not None:
                metadata['head_bias_course'] = head_bias_course
            if voltage_domain is not None:
                motor_cfg = train_cfg.scene.entities['robot'].articulation.actuators[0]
                metadata.update(voltage_domain=voltage_domain,
                    training_voltage_range_v=json.dumps(motor_cfg.vin_range),
                    training_voltage_min_v=str(motor_cfg.vin_min),
                    voltage_calibration_status='telemetry_informed_sensitivity_not_identified')
            if action_rate_weight is not None:
                metadata['fixed_action_rate_weight'] = str(action_rate_weight)
            if tracking_axes is not None:
                metadata['tracking_axes'] = tracking_axes
            if swing_reference is not None:
                metadata['swing_height_reference'] = swing_reference
            if standing_fraction is not None:
                metadata['fixed_standing_fraction'] = str(standing_fraction)
            if mirror_loss_weight is not None:
                metadata['mirror_loss_weight'] = str(mirror_loss_weight)
            if tracking_mean_seconds is not None:
                metadata['tracking_mean_seconds'] = str(tracking_mean_seconds)
            if straight_yaw_std is not None:
                metadata['straight_yaw_std'] = str(straight_yaw_std)
                metadata['straight_yaw_mean_seconds'] = '0.4'
            if airtime_height_gate is not None:
                metadata['airtime_height_gate'] = airtime_height_gate
            if feedback_age_max_steps is not None:
                metadata['feedback_age_max_steps'] = str(feedback_age_max_steps)
            if bilateral_clearance_weight is not None:
                metadata['bilateral_clearance_weight'] = str(bilateral_clearance_weight)
                metadata['bilateral_clearance_target'] = str(
                    .025 if bilateral_clearance_target is None else bilateral_clearance_target)
            if resume_exploration_std_cap is not None:
                metadata['resume_exploration_std_cap'] = str(resume_exploration_std_cap)
            if fixed_exploration_std is not None:
                metadata['fixed_exploration_std'] = str(fixed_exploration_std)
            if action_rate_domain is not None:
                metadata['action_rate_domain'] = action_rate_domain
            if clearance_course_mm is not None:
                metadata['clearance_course_mm'] = str(clearance_course_mm)
                metadata['clearance_course_semantics'] = 'lowest_collision_sole_shortfall_v1'
                metadata['swing_height_reference'] = 'collision'
                if bilateral_clearance_weight is not None:
                    metadata['bilateral_clearance_target'] = str(clearance_course_mm / 1000.)
            if walking_hip_roll_std is not None:
                metadata['walking_hip_roll_std'] = str(walking_hip_roll_std)
            if airtime_window_shift is not None:
                metadata['airtime_window_shift'] = str(airtime_window_shift)
            if terrain_course is not None:
                metadata['terrain_course'] = terrain_course
            if slew_demand_weight is not None:
                metadata['slew_demand_weight'] = str(slew_demand_weight)
            if forward_probability is not None:
                metadata['forward_probability'] = str(forward_probability)
            if walking_flexion_scale is not None:
                metadata['walking_flexion_scale'] = str(walking_flexion_scale)
            onnx.helper.set_model_props(model, metadata)
            onnx.save(model, target)
            (Path(path) / 'motor_calibration.json').write_bytes(PROFILE_PATH.read_bytes())

    runner = deepcopy(MicroduckSitStandRlCfg if repair_variant == 'sitstand' else
                      MicroduckRouladeRlCfg if repair_variant == 'roulade' else MicroduckRlCfg)
    if repair_variant == 'roulade':
        # Tyro cannot serialize the upstream dict | None default in this runner.
        runner.algorithm.symmetry_cfg = None
    runner.experiment_name = 'microduck_hd1910_xgobam' + ('_p6' if KP_FW == 6 else '') + (
        '_sitstand' if repair_variant == 'sitstand' else
        '_roulade' if repair_variant == 'roulade' else '')
    play_cfg = make_xgo_bam_env_cfg(play=True, motion_refine=motion_refine,
                                  locomotion_refine=locomotion_refine, transfer_refine=transfer_refine,
                                  repair_variant=repair_variant)
    if voltage_domain is not None:
        configure_voltage_domain(play_cfg, voltage_domain)
    if head_bias_course is not None:
        configure_head_bias_course(play_cfg, head_bias_course)
    if action_rate_weight is not None:
        configure_action_rate_weight(play_cfg, action_rate_weight)
    if tracking_axes is not None:
        configure_tracking_axes(play_cfg, tracking_axes)
    if swing_reference is not None:
        configure_swing_reference(play_cfg, swing_reference)
    if standing_fraction is not None:
        configure_standing_fraction(play_cfg, standing_fraction)
    if tracking_mean_seconds is not None:
        configure_tracking_mean(play_cfg, tracking_mean_seconds)
    if straight_yaw_std is not None:
        configure_straight_yaw(play_cfg, straight_yaw_std)
    if airtime_height_gate is not None:
        configure_airtime_height_gate(play_cfg, airtime_height_gate)
    if bilateral_clearance_weight is not None:
        configure_bilateral_clearance_bonus(play_cfg, bilateral_clearance_weight,
            .025 if bilateral_clearance_target is None else bilateral_clearance_target)
    if walking_hip_roll_std is not None:
        configure_walking_hip_roll_std(play_cfg, walking_hip_roll_std)
    if airtime_window_shift is not None:
        configure_airtime_window_shift(play_cfg, airtime_window_shift)
    if slew_demand_weight is not None:
        configure_slew_demand_weight(play_cfg, slew_demand_weight)
    if walking_flexion_scale is not None:
        configure_walking_flexion_scale(play_cfg, walking_flexion_scale)
    if action_rate_domain is not None:
        configure_action_rate_domain(play_cfg, action_rate_domain)
    if clearance_course_mm is not None:
        configure_clearance_course(play_cfg, clearance_course_mm)
    register_mjlab_task(task_id=task_id, env_cfg=train_cfg, play_env_cfg=play_cfg,
        rl_cfg=runner, runner_cls=XgoReferenceRunner)
