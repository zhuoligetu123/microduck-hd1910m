"""Head-up, double-support sway. Phase command, stock XL330 physics, 50 Hz.

The mouth remains the original independent channel, outside the 14D policy.
"""
from copy import deepcopy
import math

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp
from .microduck_step_env_cfg import make_microduck_step_env_cfg, MicroduckStepRlCfg

SWAY_PERIOD = 4.0
SWAY_AMPLITUDE = math.radians(8.)
HEAD_PITCH = math.radians(10.)
WIDE_SWAY_AMPLITUDE = math.radians(12.)
HEAD_YAW_AMPLITUDE = math.radians(15.)
WIDE_SWAY_PERIOD = 2.5
# HOME forward kinematics, measured from the sole sites to trunk_base.
TRUNK_HEIGHT = 0.1171277719213554


def make_microduck_sway_env_cfg(play=False):
    cfg = make_microduck_step_env_cfg(play=play)
    cfg.episode_length_s = 32.
    cfg.commands['twist'].period = SWAY_PERIOD
    for name in ('step_heights', 'step_contacts'):
        cfg.rewards.pop(name)
    cfg.curriculum.pop('head_pose_bias_weight')
    cfg.rewards['head_pose_bias'].weight = 3.
    cfg.rewards['sway_roll'] = RewardTermCfg(
        func=mdp.sway_roll_tracking, weight=6.,
        params={'amplitude': SWAY_AMPLITUDE, 'std': math.radians(4.)})
    cfg.rewards['planted_feet'] = RewardTermCfg(func=mdp.sway_planted_feet, weight=3.)
    cfg.rewards['trunk_height'] = RewardTermCfg(
        func=mdp.sway_trunk_height, weight=5., params={'height': TRUNK_HEIGHT, 'std': .008})
    cfg.rewards['head_gaze_error'] = RewardTermCfg(
        func=mdp.head_gaze_error, weight=-8., params={
            'pitch': HEAD_PITCH,
            'asset_cfg': SceneEntityCfg('robot', site_names=('head_camera',)),
        })
    return cfg


MicroduckSwayRlCfg = deepcopy(MicroduckStepRlCfg)
MicroduckSwayRlCfg.experiment_name = 'sway_in_place_50hz'
MicroduckSwayRlCfg.run_name = 'head_up_sway'


def make_microduck_sway_head_env_cfg(play=False):
    cfg = make_microduck_sway_env_cfg(play=play)
    cfg.commands['twist'].period = WIDE_SWAY_PERIOD
    cfg.rewards['sway_roll'].params['amplitude'] = WIDE_SWAY_AMPLITUDE
    cfg.rewards['sway_head_yaw'] = RewardTermCfg(
        func=mdp.sway_head_yaw_tracking, weight=6., params={
            'amplitude': HEAD_YAW_AMPLITUDE, 'std': math.radians(6.),
            'asset_cfg': SceneEntityCfg('robot', site_names=('head_camera',)),
        })
    return cfg


MicroduckSwayHeadRlCfg = deepcopy(MicroduckSwayRlCfg)
MicroduckSwayHeadRlCfg.experiment_name = 'sway_head_50hz'
MicroduckSwayHeadRlCfg.run_name = 'wide_sway_head'
