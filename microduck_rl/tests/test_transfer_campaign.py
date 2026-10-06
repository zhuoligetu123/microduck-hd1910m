import sys
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from run_m6_transfer_campaign import CASES, screen_score


def reports():
    return [{'cases': [dict(case=name, completed=True, no_fall=True,
            baseline_check_passed=True, motion_quality_check_passed=True,
            head_center_check_passed=True) for name in CASES]} for _ in range(2)]


def test_empty_and_stationary_do_not_unlock_long_training():
    assert screen_score([]) is None
    rows = reports()
    for report in rows:
        for row in report['cases']:
            row['baseline_check_passed'] = row['case'] == 'stand'
    assert screen_score(rows) is None


def test_missing_or_falling_cases_are_rejected():
    rows = reports()
    rows[0]['cases'].pop()
    assert screen_score(rows) is None
    rows = reports()
    rows[1]['cases'][2]['no_fall'] = False
    assert screen_score(rows) is None


def test_both_engines_must_show_forward_motion():
    rows = reports()
    assert screen_score(rows) == (10, 10, 10)
    changed = deepcopy(rows)
    changed[1]['cases'][1]['baseline_check_passed'] = False
    assert screen_score(changed) is None
