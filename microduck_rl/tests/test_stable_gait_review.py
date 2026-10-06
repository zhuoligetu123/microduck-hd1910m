from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from review_stable_gait import rank_key, stability_metrics


def row():
    return dict(completed=True, no_fall_or_head_contact=True, target_limit_violations=0,
                command=[.1, 0., 0.], max_tilt_deg=12.,
                prefall_feet=dict(complete_swing_count=[30, 31], swing_peak_median_mm=[4., 4.]))


def test_tracking_regression_and_height_targets_do_not_affect_selection():
    a = row()
    b = deepcopy(a)
    b.update(baseline_check_passed=False, straight_gait_check_passed=False,
             mean_body_velocity_after_1s=[-.05, .2, -3.], motion_quality_check_passed=False,
             prefall_straight_path=dict(max_heading_error_deg=360., max_cross_track_m=5.))
    b['prefall_feet']['swing_peak_median_mm'] = [25., 25.]
    assert stability_metrics([a]) == stability_metrics([b])


def test_fall_and_missing_head_contact_evidence_cannot_win():
    a = row()
    for evidence in (False, None):
        b = deepcopy(a)
        b['no_fall_or_head_contact'] = evidence
        assert rank_key(dict(name='a', metrics=stability_metrics([a]))) < rank_key(
            dict(name='b', metrics=stability_metrics([b])))


def test_no_swings_cannot_masquerade_as_stable_gait():
    a = row()
    a['prefall_feet']['complete_swing_count'] = [0, 30]
    assert stability_metrics([a])['bilateral_swing_cases'] == 0
