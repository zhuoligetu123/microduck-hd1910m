"""Failed native training must not be reported as a successful Docker job."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('skills_campaign', SCRIPTS / 'run_m6_skills_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_native_signal_is_failure(tmp_path):
    with patch.object(sys, 'argv', ['campaign', str(tmp_path), '--queue', 'roll']), \
         patch.object(campaign, 'run_training', return_value=({'returncode': -4}, {})):
        assert campaign.main() == 1
    assert json.loads((tmp_path/'status.json').read_text())['stage'] == 'failed'
    assert 'exit=-4' in json.loads((tmp_path/'summary.json').read_text())['outcomes'][0]['error']


def test_incomplete_zero_exit_is_failure(tmp_path):
    with patch.object(sys, 'argv', ['campaign', str(tmp_path), '--queue', 'roll']), \
         patch.object(campaign, 'run_training', return_value=({'returncode': 0, 'iterations_reported': 3}, {})):
        assert campaign.main() == 1


def test_resume_finishes_original_budget_and_keeps_failed_acceptance_visible(tmp_path):
    checkpoint = tmp_path/'model_750.pt'
    checkpoint.touch()
    policy = tmp_path/'runs/roulade/logs/test_roulade.onnx'
    policy.parent.mkdir(parents=True)
    policy.touch()
    with patch.object(sys, 'argv', ['campaign', str(tmp_path), '--queue', 'roll',
                                   '--resume-checkpoint', str(checkpoint)]), \
         patch.object(campaign, 'run_training', side_effect=[
             ({'returncode': 0, 'iterations_reported': 5}, {}),
             ({'returncode': 0, 'iterations_reported': 449}, {})]) as train, \
         patch.object(campaign, 'evaluate', return_value={'roll': {'returncode': 2}}):
        assert campaign.main() == 0
    assert train.call_args_list[1].args[3] == 449
    result = json.loads((tmp_path/'summary.json').read_text())['outcomes'][0]
    assert result['stage'] == 'review_required'
    assert result['checks']['roll']['returncode'] == 2
    assert result['deployment_ready'] is False
