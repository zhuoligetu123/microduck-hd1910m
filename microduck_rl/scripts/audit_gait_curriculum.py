#!/usr/bin/env python3
"""Audit deterministic replay evidence, without training or promoting a policy."""
import argparse
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_case(row, target_mm):
    failures = []
    if not row.get('completed') or not row.get('no_fall') or row.get('steps', 0) < 1000:
        failures.append('incomplete_or_fall')
    if row.get('no_fall_or_head_contact') is not True:
        failures.append('head_contact_check_failed_or_unavailable')
    if row.get('baseline_check_passed') is not True:
        failures.append('velocity_tracking')
    if row.get('target_limit_violations') != 0:
        failures.append('target_range_check_failed_or_unavailable')
    moving = any(abs(v) > .01 for v in row['command'])
    if moving:
        feet = row.get('prefall_feet', {})
        peaks = feet.get('swing_peak_median_mm', [])
        counts = feet.get('complete_swing_count', [])
        if len(peaks) != 2 or any(not finite(v) or v <= 0 or (target_mm is not None and v < target_mm)
                                  for v in peaks):
            failures.append('bilateral_clearance')
        if len(counts) != 2 or any(not finite(v) or v < 5 for v in counts):
            failures.append('insufficient_complete_swings')
        if len(peaks) == 2 and all(finite(v) and v > 0 for v in peaks):
            if min(peaks) / max(peaks) < .8:
                failures.append('left_right_asymmetry')
        if abs(row['command'][2]) < .01:
            path = row.get('prefall_straight_path') or {}
            for name, limit in [('max_heading_error_deg', 15.), ('max_cross_track_m', .15)]:
                value = path.get(name)
                if not finite(value) or value > limit:
                    failures.append(name)
    return failures


def audit(data, target_mm, tasks):
    if target_mm is not None and target_mm not in (12., 15., 20., 25.):
        raise ValueError('course height must be 12, 15, 20, or 25 mm')
    rows = data.get('results', data.get('cases', []))
    if not isinstance(rows, list):
        raise ValueError('expected replay cases or evaluation results')
    selected = [r for r in rows if r['case'] in tasks]
    missing = sorted(set(tasks) - {r['case'] for r in selected})
    results = [dict(condition=r.get('condition'), seed=r.get('seed'), case=r['case'],
                    failures=check_case(r, target_mm)) for r in selected]
    return dict(policy_sha256=data.get('sha256', data.get('policy_sha256')),
                acceptance_mode='natural_gait' if target_mm is None else 'historical_height_course',
                visual_review_required=True,
                target_mm=target_mm, required_tasks=tasks, missing_tasks=missing,
                rows=results, passed=sum(not r['failures'] for r in results),
                supplied_evidence_passed=bool(results) and not missing
                    and all(not r['failures'] for r in results),
                course_promotion_authorized=False, deployment_ready=False, hardware_tested=False,
                note='Partial evidence audit only; seed coverage, dynamic reachability, reward ranking, '
                     'swing distributions, transitions and final holdout remain separate requirements.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--target-mm', type=float, choices=(12., 15., 20., 25.),
                      help='Historical diagnostic only; not the current natural-gait objective')
    mode.add_argument('--natural-gait', action='store_true',
                      help='No fixed height threshold; retain separate mandatory visual review')
    parser.add_argument('--tasks', nargs='+', default=['stand', 'forward', 'backward', 'turn', 'turn_right'])
    args = parser.parse_args()
    result = audit(json.loads(args.input.read_text()), args.target_mm, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
