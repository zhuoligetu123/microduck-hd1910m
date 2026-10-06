"""HD velocity recipes: identified, provisional reference, and opt-in bounded.

The first two retain the stock action contract. The bounded variant changes
applied action history and adds a saturation cost without replacing the base task.
"""
from mjlab_microduck.robot.hd1910 import MotorFit, adapt_env
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def make_hd1910_velocity_env_cfg(calibration, play=False):
    fit = MotorFit.load(calibration)
    return adapt_env(make_microduck_velocity_env_cfg(play=play), fit)


def make_reference_hd1910_velocity_env_cfg(play=False):
    from copy import deepcopy
    from mjlab_microduck.actuator.reference_hd1910 import (
        Hd1910ActuatorCfg, make_hd1910_spec, finalize_hd1910_env,
    )
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play))
    robot = cfg.scene.entities['robot']
    robot.spec_fn = make_hd1910_spec
    robot.articulation.actuators = (Hd1910ActuatorCfg(target_names_expr=(r'^(?!passive_).*',)),)
    return finalize_hd1910_env(cfg)


def configure_hd1910_position_bounds(cfg):
    """Apply the same trained target/history contract to each HD task family."""
    from dataclasses import fields
    from mjlab.managers import RewardTermCfg
    from mjlab_microduck.actuator.bounded_position import BoundedPositionActionCfg
    from .mdp import hd_target_saturation_cost
    import math
    model = cfg.scene.entities['robot'].spec_fn().compile()
    margin = math.radians(2)
    bounds = {j.name: (float(j.range[0])+margin, float(j.range[1])-margin)
              for i in range(model.njnt) if (j := model.joint(i)).name
              and not j.name.startswith('passive_') and model.jnt_limited[i]}
    if len(bounds) != 14 or any(lo >= hi for lo, hi in bounds.values()):
        raise ValueError('expected 14 finite bounded position joints')
    old = cfg.actions['joint_pos']
    action = BoundedPositionActionCfg(**{f.name: getattr(old, f.name) for f in fields(old) if f.init})
    action.clip = bounds
    cfg.actions['joint_pos'] = action
    for group in ('actor', 'critic'):
        cfg.observations[group].terms['actions'].params = {'action_name': 'joint_pos'}
    cfg.rewards['hd_target_saturation'] = RewardTermCfg(func=hd_target_saturation_cost, weight=-2.)
    return cfg


def make_bounded_hd1910_velocity_env_cfg(play=False):
    return configure_hd1910_position_bounds(make_reference_hd1910_velocity_env_cfg(play=play))


def make_slew_hd1910_velocity_env_cfg(play=False, max_step_rad=.10):
    """A new trained contract, never a post-export filter on old policies."""
    import math
    cfg = make_bounded_hd1910_velocity_env_cfg(play=play)
    from .mdp import hd_applied_action_rate_cost
    cfg.actions['joint_pos'].max_step_rad = max_step_rad
    cfg.rewards['action_rate_l2'].func = hd_applied_action_rate_cost
    cfg.rewards['action_rate_l2'].params = {}
    twist = cfg.commands['twist']
    twist.ranges.lin_vel_x = (-.15, .15)
    twist.ranges.lin_vel_y = (-.04, .04)
    twist.ranges.ang_vel_z = (-.5, .5)
    twist.rel_standing_envs = .25
    twist.rel_turn_in_place_envs = .35
    cfg.curriculum.pop('standing_envs', None)
    # Keep nonzero head command inputs, without making large head motions a
    # competing task during the initial low-speed locomotion qualification.
    cfg.curriculum.pop('head_pose_range', None)
    cfg.commands['head_pose'].ranges = ((-.05,.05),(-.05,.05),(-.07,.07),(-.015,.015))
    cfg.rewards['track_linear_velocity'].weight = 4.
    cfg.rewards['track_linear_velocity'].params['std'] = math.sqrt(.01)
    cfg.rewards['track_angular_velocity'].weight = 4.
    cfg.rewards['track_angular_velocity'].params['std'] = math.sqrt(.08)
    return cfg


def make_discovery_hd1910_velocity_env_cfg(play=False):
    """Learn stepping before strong smoothing; retain the v2 action contract."""
    from mjlab.managers import CurriculumTermCfg
    from .mdp import standing_envs_curriculum
    from .microduck_velocity_env_cfg import NUM_STEPS_PER_ENV

    cfg = make_slew_hd1910_velocity_env_cfg(play=play)
    # A narrow hip-roll pose prior and early -1 smoothing made standing a
    # local optimum. These are learning costs, not hardware safety limits.
    cfg.rewards['pose'].weight = .25
    for key in ('std_walking', 'std_running'):
        cfg.rewards['pose'].params[key] = dict(cfg.rewards['pose'].params[key])
        cfg.rewards['pose'].params[key][r'.*hip_roll.*'] = .15
    cfg.rewards['action_rate_l2'].weight = -.02
    cfg.curriculum['action_rate_weight'].params['weight_stages'] = [
        {'step': step * NUM_STEPS_PER_ENV, 'weight': weight}
        for step, weight in ((0, -.02), (2500, -.05), (4000, -.1), (6000, -.2))
    ]
    cfg.commands['twist'].rel_standing_envs = .05
    cfg.curriculum['standing_envs'] = CurriculumTermCfg(
        func=standing_envs_curriculum,
        params={'command_name': 'twist', 'standing_stages': [
            {'step': step * NUM_STEPS_PER_ENV, 'rel_standing_envs': fraction}
            for step, fraction in ((0, .05), (2500, .1), (4000, .2), (5000, .25))
        ]},
    )
    return cfg


def configure_head_centering(cfg):
    """Opt-in DC-head repair for a learned gait; do not clamp runtime actions."""
    cfg.rewards['head_pose_bias'].params['axis_weights'] = (1., 1., 1., 3.)
    cfg.curriculum['head_pose_bias_weight'].params['weight_stages'] = [
        {'step': step * 24, 'weight': weight}
        for step, weight in ((0, 1.), (200, 2.), (400, 3.))]
    return cfg


def make_refined_hd1910_velocity_env_cfg(play=False):
    """Post-replay repair pilot; keep physics and v2 deployment contract intact."""
    from dataclasses import fields
    from mjlab.managers import RewardTermCfg
    from . import mdp

    cfg = make_discovery_hd1910_velocity_env_cfg(play=play)
    old = cfg.commands['twist']
    command = mdp.HdLowSpeedCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old) if f.init})
    command.rel_forward_envs = 0.
    command.rel_standing_envs = .20
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (3., 6.)
    cfg.commands['twist'] = command
    cfg.curriculum.pop('standing_envs', None)
    cfg.rewards['track_linear_velocity'] = RewardTermCfg(func=mdp.hd_planar_velocity_tracking, weight=4.)
    cfg.rewards['track_angular_velocity'] = RewardTermCfg(func=mdp.hd_yaw_velocity_tracking, weight=4.)
    cfg.rewards['hd_velocity_error'] = RewardTermCfg(func=mdp.hd_velocity_error_cost, weight=-1.,
                                                   params={'yaw_square_weight': 0.})
    # Keep smoothing constant during this comparison, not a time-driven -1 ramp.
    cfg.curriculum.pop('action_rate_weight', None)
    cfg.rewards['action_rate_l2'].weight = -.10
    cfg.rewards['hd_slew_demand'] = RewardTermCfg(func=mdp.hd_slew_demand_cost, weight=0.)
    cfg.events['push_robot'].params['velocity_range']['yaw'] = (0., 0.)
    import os
    if os.environ.get('MICRODUCK_HEAD_CENTER') == '1':
        configure_head_centering(cfg)
    return cfg


def refine_hd1910_posture(cfg):
    """Remove the 30-degree standing compromise without changing SIT/stand flags."""
    from mjlab.managers import RewardTermCfg
    from .mdp import hd_posture_goal_progress
    cfg.rewards['posture_pose_legs'].weight = 1.5
    cfg.rewards['posture_pose_l1'].weight = .5
    cfg.rewards['upright_linear'].weight = 5.
    cfg.rewards['posture_composite'].weight = 6.
    cfg.rewards['posture_composite'].params['upright_std'] = .30
    cfg.rewards['posture_stillness'].params.update(tilt_full_deg=10., tilt_zero_deg=25.)
    cfg.curriculum.pop('action_rate_weight', None)
    cfg.rewards['action_rate_l2'].weight = -.2
    cfg.curriculum.pop('head_pose_range', None)
    cfg.commands['head_pose'].ranges = ((-.05,.05),(-.05,.05),(-.07,.07),(-.015,.015))
    cfg.rewards['hd_posture_progress'] = RewardTermCfg(func=hd_posture_goal_progress,weight=0.,
        params={'command_name':'twist','sit_z':cfg.commands['twist'].sit_z,
                'stand_z':cfg.commands['twist'].stand_z})
    return cfg


def balance_hd1910_posture(cfg):
    """Preserve the proven seated prior; soften HOME only for upright stance."""
    from . import mdp
    cfg = refine_hd1910_posture(cfg)
    cfg.rewards['posture_pose_legs'].func = mdp.hd_posture_pose_match
    cfg.rewards['posture_pose_legs'].weight = 4.
    cfg.rewards['posture_pose_l1'].func = mdp.hd_posture_pose_l1
    cfg.rewards['posture_pose_l1'].weight = 1.
    return cfg
