"""HD twins of the stock registry, preserving task commands and geometry.

Raw twins retain the original actions. Opt-in SitStand/GroundPick slew pilots
use the same bounded action history in training and export as the walking pilot.
Training/export smoke success is not task mastery or hardware approval.
"""
from functools import partial

SUFFIX = '-HD1910-Reference'
BOUNDED_PILOTS = ('Mjlab-SitStand-Flat-MicroDuck', 'Mjlab-GroundPick-Flat-MicroDuck')


def hd_spec(factory):
    from mjlab_microduck.actuator.reference_hd1910 import adapt_hd1910_spec
    return adapt_hd1910_spec(factory())


def adapt_task(cfg, slew=False):
    from copy import deepcopy
    from mjlab_microduck.actuator.reference_hd1910 import Hd1910ActuatorCfg, finalize_hd1910_env
    cfg = deepcopy(cfg)
    robot = cfg.scene.entities['robot']
    robot.spec_fn = partial(hd_spec, robot.spec_fn)
    robot.articulation.actuators = (Hd1910ActuatorCfg(target_names_expr=(r'^(?!passive_).*',)),)
    cfg = finalize_hd1910_env(cfg)
    if slew:
        from .microduck_hd1910_env_cfg import configure_hd1910_position_bounds
        from .mdp import hd_applied_action_rate_cost
        cfg = configure_hd1910_position_bounds(cfg)
        cfg.actions['joint_pos'].max_step_rad = .10
        if 'action_rate_l2' in cfg.rewards:
            cfg.rewards['action_rate_l2'].func = hd_applied_action_rate_cost
            cfg.rewards['action_rate_l2'].params = {}
    return cfg


def register_suite(runner_cls):
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, register_mjlab_task
    stock = [t for t in list_tasks() if 'MicroDuck' in t and '-HD1910' not in t]
    for task in stock:
        if task + SUFFIX in list_tasks():
            continue
        agent = load_rl_cfg(task)
        agent.experiment_name = 'hd_suite_' + task.removeprefix('Mjlab-').lower().replace('-', '_')
        register_mjlab_task(task + SUFFIX, adapt_task(load_env_cfg(task)),
                           adapt_task(load_env_cfg(task, play=True)), agent, runner_cls)
    for task in BOUNDED_PILOTS:
        agent = load_rl_cfg(task)
        agent.experiment_name = 'hd_suite_' + task.removeprefix('Mjlab-').lower().replace('-', '_') + '_slew'
        register_mjlab_task(task + SUFFIX + '-Slew', adapt_task(load_env_cfg(task), slew=True),
                           adapt_task(load_env_cfg(task, play=True), slew=True), agent, runner_cls)
    from .microduck_hd1910_env_cfg import refine_hd1910_posture
    task = 'Mjlab-SitStand-Flat-MicroDuck'
    agent = load_rl_cfg(task + SUFFIX + '-Slew')
    agent.experiment_name += '_refine'
    register_mjlab_task(task + SUFFIX + '-Slew-Refine',
                       refine_hd1910_posture(adapt_task(load_env_cfg(task), slew=True)),
                       refine_hd1910_posture(adapt_task(load_env_cfg(task, play=True), slew=True)),
                       agent, runner_cls)
    from .microduck_hd1910_env_cfg import balance_hd1910_posture
    agent = load_rl_cfg(task + SUFFIX + '-Slew')
    agent.experiment_name += '_balanced'
    register_mjlab_task(task + SUFFIX + '-Slew-Balanced',
                       balance_hd1910_posture(adapt_task(load_env_cfg(task), slew=True)),
                       balance_hd1910_posture(adapt_task(load_env_cfg(task, play=True), slew=True)),
                       agent, runner_cls)
