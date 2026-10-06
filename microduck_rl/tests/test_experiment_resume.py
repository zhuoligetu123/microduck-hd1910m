"""Experimental continuation must not silently reset the curriculum."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from run_head_balance_experiment import training_start
from mjlab_microduck.tasks import MicroduckOnPolicyRunner, VelocityOnPolicyRunner


def test_resume_stages_original_completed_iteration(tmp_path):
    (tmp_path/'parent').mkdir()
    torch.save({'iter': 199}, tmp_path/'parent/model.pt')
    args, start = training_start(tmp_path, 'gait_reference_linear_only_v22', True)
    assert start == 200
    assert '--warm-start-checkpoint' not in args
    assert args[-2:] == ['--resume-checkpoint', str(tmp_path/'parent/model_199.pt')]
    assert (tmp_path/'parent/model_199.pt').read_bytes() == (tmp_path/'parent/model.pt').read_bytes()


def test_warm_start_is_still_explicit(tmp_path):
    args, start = training_start(tmp_path, 'gait_reference_linear_only_v22')
    assert start == 0
    assert '--warm-start-checkpoint' in args and '--resume-checkpoint' not in args


def test_native_resume_preserves_curriculum_and_optimizer_lr(monkeypatch):
    monkeypatch.delenv('MICRODUCK_HD1910_WARM_START', raising=False)
    env = SimpleNamespace(common_step_counter=0)
    algorithm = SimpleNamespace(learning_rate=.1,
        optimizer=SimpleNamespace(param_groups=[{'lr': .00005}]))
    runner = object.__new__(MicroduckOnPolicyRunner)
    runner.device = 'cpu'
    runner.current_learning_iteration = 199
    runner.env = SimpleNamespace(unwrapped=env)
    runner.cfg = {'num_steps_per_env': 24}
    runner.alg = algorithm
    with patch.object(VelocityOnPolicyRunner, 'load', return_value={'loaded': True}):
        result = MicroduckOnPolicyRunner.load(runner, 'unused.pt')
    assert result == {'loaded': True}
    assert runner.current_learning_iteration == 200
    assert env.common_step_counter == 4800
    assert algorithm.learning_rate == .00005
