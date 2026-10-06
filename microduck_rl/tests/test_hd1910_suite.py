"""Task adaptation must preserve command and geometry semantics, not just dims."""
import dataclasses
import math
import mujoco
import pytest
import mjlab_microduck.tasks
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.tasks.hd1910_suite import adapt_task


def test_warm_start_is_explicit_not_ordinary_resume(monkeypatch):
    from types import SimpleNamespace
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner, VelocityOnPolicyRunner
    def fixture_load(self, path, load_cfg=None, map_location=None):
        assert map_location == 'cpu'
        if load_cfg is None or load_cfg.get('iteration', False):
            self.current_learning_iteration = 5496
        self.alg.optimizer.param_groups[0]['lr'] = .001
        return {'loaded': path}
    monkeypatch.setattr(VelocityOnPolicyRunner, 'load', fixture_load)
    runner = object.__new__(MicroduckOnPolicyRunner)
    runner.device = 'cpu'
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    runner.cfg = {'num_steps_per_env': 24}
    runner.alg = SimpleNamespace(learning_rate=.00003, optimizer=SimpleNamespace(param_groups=[{}]))
    monkeypatch.delenv('MICRODUCK_HD1910_WARM_START', raising=False)
    assert runner.load('fixture')['loaded'] == 'fixture'
    assert runner.current_learning_iteration == 5497
    assert runner.env.unwrapped.common_step_counter == 5497 * 24
    assert runner.alg.learning_rate == .001
    assert runner.alg.optimizer.param_groups[0]['lr'] == .001
    monkeypatch.setenv('MICRODUCK_HD1910_WARM_START', '1')
    runner.alg.learning_rate = .00003
    runner.load('fixture')
    assert runner.current_learning_iteration == runner.env.unwrapped.common_step_counter == 0
    assert runner.alg.optimizer.param_groups[0]['lr'] == .00003
    monkeypatch.delenv('MICRODUCK_HD1910_WARM_START')
    runner.load('fixture', load_cfg={'actor': True})
    assert runner.current_learning_iteration == runner.env.unwrapped.common_step_counter == 0


@pytest.mark.parametrize('task', [t for t in list_tasks() if 'MicroDuck' in t and '-HD1910' not in t])
def test_every_stock_task_keeps_geometry_and_commands(task):
    original = load_env_cfg(task)
    cfg = adapt_task(load_env_cfg(task))
    old = original.scene.entities['robot'].spec_fn().compile()
    new = cfg.scene.entities['robot'].spec_fn().compile()
    assert old.njnt == new.njnt and old.ngeom == new.ngeom
    assert new.nu == 0  # DC actuator manager creates the 14 motors, not duplicate XML motors.
    assert list(original.commands) == list(cfg.commands)
    assert [type(c) for c in original.commands.values()] == [type(c) for c in cfg.commands.values()]
    assert dataclasses.asdict(original.actions['joint_pos']) == dataclasses.asdict(cfg.actions['joint_pos'])
    assert list(original.observations['actor'].terms) == list(cfg.observations['actor'].terms)
    assert original.rewards.keys() == cfg.rewards.keys()
    assert math.isclose(sum(new.body_mass) - sum(old.body_mass), 15 * (.021 - .018), abs_tol=1e-6)
    for i in range(new.njnt):
        joint = new.joint(i)
        if joint.name.startswith('passive_') and joint.name.endswith('_backlash'):
            assert math.isclose(joint.range[1] - joint.range[0], math.radians(.5))
    assert all('bam' not in term.func.__name__ for term in cfg.events.values())


@pytest.mark.parametrize('task', ('Mjlab-SitStand-Flat-MicroDuck', 'Mjlab-GroundPick-Flat-MicroDuck'))
def test_slew_pilots_preserve_skill_commands_and_physics(task):
    stock = load_env_cfg(task)
    raw = adapt_task(stock)
    bounded = adapt_task(stock, slew=True)
    assert stock.actions['joint_pos'].clip is None
    assert raw.actions['joint_pos'].clip is None
    assert len(bounded.actions['joint_pos'].clip) == 14
    assert bounded.actions['joint_pos'].max_step_rad == .10
    assert bounded.commands == raw.commands
    assert bounded.terminations == raw.terminations
    assert bounded.events == raw.events
    for group in ('actor', 'critic'):
        assert bounded.observations[group].terms.keys() == raw.observations[group].terms.keys()
        assert bounded.observations[group].terms['actions'].params == {'action_name': 'joint_pos'}
    for lo, hi in bounded.actions['joint_pos'].clip.values():
        assert lo < hi and math.isfinite(lo) and math.isfinite(hi)
