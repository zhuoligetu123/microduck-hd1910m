import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from run_head_balance_experiment import compare


def report(policy_hash='abc'):
    case = dict(no_fall=True, completed=True, baseline_check_passed=False,
                max_tilt_deg=10., command=[.1, 0., .4],
                mean_body_velocity_after_1s=[.08, 0., -.4],
                foot_clearance_p95_m=[.01, .02],
                settled_prefall_head_metrics={'head_mean_error_deg': [3., 7., 0., 0.]})
    return dict(policy_sha256=policy_hash, head_command_offset_deg=[20., -20.],
                cases=[case, {**case, 'no_fall': False, 'max_tilt_deg': 160.}])


def test_summary_preserves_falls_and_wrong_turn_direction(tmp_path):
    directory = tmp_path / 'evaluations' / 'candidate'
    directory.mkdir(parents=True)
    (directory / 'combined_age4_seed42.json').write_text(json.dumps(report()))
    compare(tmp_path)
    data = json.loads((tmp_path / 'metric_comparison.json').read_text())
    result = data['comparison']['candidate']
    assert result['cases'] == 2 and result['no_fall'] == 1
    assert result['max_tilt_deg'] == 160.
    assert result['survived_only_means']['turn_signed_progress_rad_s'] == -.4
    assert result['survived_only_means']['moving_sole_p95_mean_mm'] == 15.
    assert result['metric_sample_counts']['head_down_abs_dc_error_deg'] == 1
    assert data['deployment_ready'] is False


def test_summary_rejects_mixed_policy_exports(tmp_path):
    directory = tmp_path / 'evaluations' / 'candidate'
    directory.mkdir(parents=True)
    (directory / 'combined_age4_seed42.json').write_text(json.dumps(report()))
    (directory / 'combined_age4_seed7.json').write_text(json.dumps(report('different')))
    with pytest.raises(ValueError, match='mixed policy hashes'):
        compare(tmp_path)
