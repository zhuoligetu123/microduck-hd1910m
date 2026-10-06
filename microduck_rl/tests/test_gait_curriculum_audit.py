from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_gait_curriculum import audit, check_case


def good_case():
    return dict(case='forward', command=[.1, 0., 0.], completed=True, no_fall=True,
                steps=1000, no_fall_or_head_contact=True, baseline_check_passed=True,
                target_limit_violations=0, prefall_feet=dict(
                    swing_peak_median_mm=[25., 25.], complete_swing_count=[10, 10]),
                prefall_straight_path=dict(max_heading_error_deg=10., max_cross_track_m=.1))


def test_stage_height_is_not_replaced_by_velocity_pass():
    row = good_case()
    assert check_case(row, 25.) == []
    for heights in ([4., 4.], [11., 25.], [None, 25.], [float('nan'), 25.]):
        row['prefall_feet']['swing_peak_median_mm'] = heights
        assert 'bilateral_clearance' in check_case(row, 25.)


def test_missing_head_contact_evidence_is_not_success():
    row = good_case()
    row['no_fall_or_head_contact'] = None
    assert 'head_contact_check_failed_or_unavailable' in check_case(row, 12.)


def test_missing_task_and_partial_audit_cannot_promote():
    data = dict(policy_sha256='fixture', cases=[good_case()])
    result = audit(data, 25., ['stand', 'forward'])
    assert result['missing_tasks'] == ['stand']
    assert not result['supplied_evidence_passed']
    result = audit(data, 25., ['forward'])
    assert result['supplied_evidence_passed']
    assert not result['course_promotion_authorized'] and not result['deployment_ready']


def test_stand_needs_no_lift_and_straight_error_is_checked():
    row = good_case()
    stand = deepcopy(row)
    stand.update(case='stand', command=[0., 0., 0.], prefall_feet={})
    assert check_case(stand, 25.) == []
    row['prefall_straight_path']['max_heading_error_deg'] = 45.
    assert 'max_heading_error_deg' in check_case(row, 12.)


def test_natural_gait_has_no_25mm_gate_but_does_not_assert_visual_acceptance():
    row = good_case()
    row['prefall_feet']['swing_peak_median_mm'] = [9., 10.]
    assert 'bilateral_clearance' in check_case(row, 25.)
    assert check_case(row, None) == []
    result = audit(dict(cases=[row]), None, ['forward'])
    assert result['acceptance_mode'] == 'natural_gait'
    assert result['visual_review_required']
    assert not result['course_promotion_authorized']
    row['prefall_feet']['swing_peak_median_mm'] = [0., 10.]
    assert 'bilateral_clearance' in check_case(row, None)
