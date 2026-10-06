"""In-place stepping only, phase-conditioned 61D observations, 50 Hz control.

The 3D command slot is [cos(phase), sin(phase), 0], NOT a velocity command.
The physical target is zero translation and zero heading change. BAM is still
the stock XL330 model: this task is not a validated HD1910 digital twin.
"""
from copy import deepcopy
import math
from mjlab.managers import RewardTermCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg, make_microduck_velocity_env_cfg,
)

STEP_PERIOD = 1.0
CONTROL_HZ = 50
HEAD_COMMAND = (math.radians(10.), 0., 0., 0.)


def make_microduck_step_env_cfg(play=False):
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play))
    cfg.decimation = round(1 / CONTROL_HZ / cfg.sim.mujoco.timestep)
    cfg.episode_length_s = 12.0
    cfg.commands['twist'] = mdp.GroundPickPhaseCommandCfg(
        entity_name='robot', resampling_time_range=(1e9, 1e9),
        ranges=deepcopy(cfg.commands['twist'].ranges),
        heading_command=False, rel_standing_envs=0.0,
        period=STEP_PERIOD, randomize_phase=False,
    )
    for axis in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
        setattr(cfg.commands['twist'].ranges, axis, (0.0, 0.0))
    cfg.commands['twist'].ranges.heading = None
    cfg.commands['head_pose'].ranges = tuple((v, v) for v in HEAD_COMMAND)
    cfg.commands['body_pose'].ranges = ((0.0, 0.0),) * 6
    cfg.curriculum = {name: cfg.curriculum[name] for name in
                      ('action_rate_weight', 'head_pose_bias_weight')}
    # Head-up is an explicit task objective; retain the original EMA rather than
    # taxing the unavoidable instantaneous head oscillation during stepping.
    for stage in cfg.curriculum['head_pose_bias_weight'].params['weight_stages']:
        stage['weight'] *= 3.0
    cfg.events.pop('push_robot', None)
    cfg.events['reset_base'].params['pose_range'].update(x=(0., 0.), y=(0., 0.), yaw=(0., 0.))
    for name in ('track_linear_velocity', 'track_angular_velocity', 'pose',
                 'air_time', 'foot_clearance', 'foot_swing_height', 'body_pose_tracking'):
        cfg.rewards.pop(name, None)
    cfg.rewards['foot_slip'].params['command_threshold'] = -1.0
    cfg.rewards['step_heights'] = RewardTermCfg(func=mdp.step_heights, weight=5.0)
    cfg.rewards['step_contacts'] = RewardTermCfg(func=mdp.step_contacts, weight=2.0)
    cfg.rewards['step_drift'] = RewardTermCfg(func=mdp.step_drift_cost, weight=-16.0)
    cfg.rewards['step_velocity'] = RewardTermCfg(func=mdp.step_velocity_cost, weight=-3.0)
    cfg.rewards['heading_hold'] = RewardTermCfg(func=mdp.heading_hold_reward, weight=4.0)
    return cfg


MicroduckStepRlCfg = deepcopy(MicroduckRlCfg)
MicroduckStepRlCfg.experiment_name = 'step_in_place_50hz'
MicroduckStepRlCfg.run_name = 'step_in_place'
MicroduckStepRlCfg.logger = 'tensorboard'
MicroduckStepRlCfg.upload_model = False
MicroduckStepRlCfg.max_iterations = 3000
MicroduckStepRlCfg.save_interval = 100
