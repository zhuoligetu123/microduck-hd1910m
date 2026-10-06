#!/usr/bin/env python3
"""Re-evaluate the discussed gait arms. Stability ranking never uses tracking gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    reports = sorted((ROOT / 'training_runs').glob('*/**/evaluation/summary.json'))
    reports.append(ROOT / 'training_runs/natural_straight_yaw_20261004/candidate_ground/summary.json')
    models = {}
    for report in reports:
        data = json.loads(report.read_text())
        policy = Path(data['policy']).resolve(strict=True)
        sha = digest(policy)
        if sha != data['sha256']:
            raise ValueError(f'Policy changed since historical evaluation: {report}')
        name = str(report.parent.parent.relative_to(ROOT / 'training_runs'))
        item = models.setdefault(sha, dict(name=name, policy=str(policy), sha256=sha,
                                           historical_reports=[]))
        item['historical_reports'].append(str(report))
    return list(models.values())


def stability_metrics(rows):
    stable = [r for r in rows if r.get('completed') and r.get('no_fall_or_head_contact') is True
              and r.get('target_limit_violations') == 0]
    moving = [r for r in rows if any(abs(v) > .01 for v in r['command'])]
    bilateral = 0
    for row in moving:
        feet = row.get('prefall_feet', {})
        counts = feet.get('complete_swing_count', [])
        peaks = feet.get('swing_peak_median_mm', [])
        if (row in stable and len(counts) == len(peaks) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 5 for v in counts)
                and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in peaks)):
            bilateral += 1
    tilts = [r['max_tilt_deg'] for r in rows
             if isinstance(r.get('max_tilt_deg'), (float, int)) and math.isfinite(r['max_tilt_deg'])]
    return dict(cases=len(rows), stable_cases=len(stable), moving_cases=len(moving),
                bilateral_swing_cases=bilateral,
                mean_case_max_tilt_deg=statistics.mean(tilts) if len(tilts) == len(rows) and tilts else None,
                worst_tilt_deg=max(tilts) if tilts else None,
                numeric_bilateral_evidence_only=True, visual_review_required=True)


def rank_key(item):
    m = item['metrics']
    return (-m['stable_cases'] / max(m['cases'], 1),
            -m['bilateral_swing_cases'] / max(m['moving_cases'], 1),
            m['mean_case_max_tilt_deg'] if m['mean_case_max_tilt_deg'] is not None else math.inf,
            item['name'])


def evaluate(item, output, seeds):
    folder = output / item['sha256'][:12]
    command = [sys.executable, str(ROOT / 'scripts/evaluate_forward_transfer.py'),
               '--policy', item['policy'], '--output', str(folder), '--ground-contact',
               '--seconds', '20', '--seeds', *map(str, seeds)]
    save(output / (item['sha256'][:12] + '.command.json'), command)
    with (output / (item['sha256'][:12] + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise RuntimeError(f'{item["name"]}: exit {result.returncode}; see {log.name}')
    report = folder / 'summary.json'
    data = json.loads(report.read_text())
    if (data['sha256'] != item['sha256'] or data['ground_contact_model'] is not True
            or len(data['results']) != len(seeds) * 20):
        raise ValueError(f'incomplete/mismatched evaluation: {report}')
    return dict(**item, report=str(report), metrics=stability_metrics(data['results']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seeds', nargs=2, type=int, default=[2027, 4099])
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or len(set(args.seeds)) != 2:
        parser.error('Use 1..8 workers and two distinct evaluation seeds')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    models = inventory()
    save(output / 'inventory.json', dict(models=models, evaluation_seeds=args.seeds,
        scope='All 48 historical completed gait evaluation arms plus the straight-yaw candidate; '
              'not every intermediate checkpoint, smoke test, or non-gait skill.',
        excluded_ranking_metrics=['velocity_tracking', 'yaw_tracking', 'cross_track',
                                 'absolute_foot_height', 'head_center', 'old_baseline_pass'],
        ranking_order=['completed_without_fall_head_contact_or_target_violation',
                       'bilateral_complete_swings', 'mean_case_max_tilt'],
        source_sha256={n: digest(ROOT / 'scripts' / n) for n in
                       ('replay_hd1910.py', 'evaluate_forward_transfer.py', Path(__file__).name)},
        hardware_tested=False))
    print(f'Evaluating {len(models)} unique policies with {args.workers} CPU workers', flush=True)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, m, output, args.seeds): m for m in models}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(item['name'], result['metrics']['stable_cases'], '/40 stable', flush=True)
            except Exception as exc:
                failures.append(dict(name=item['name'], error=str(exc)))
                print('ERROR', item['name'], str(exc), flush=True)
            save(output / 'progress.json', dict(completed=len(results), total=len(models), failures=failures))
    results.sort(key=rank_key)
    save(output / 'ranking.json', dict(ranked=results, failures=failures,
        provisional_candidates=results[:2] if not failures else [],
        selection_semantics='Stability and visible-step evidence only; no velocity/direction regression gate',
        hardware_tested=False, deployment_ready=False))
    lines = ['# Stability-Only Gait Review', '',
             'All rows are fresh CPU MuJoCo tests. Direction/speed/height targets do not affect ranking.', '',
             '| Rank | Model | Stable | Bilateral swing cases | Mean / worst max tilt (deg) |',
             '| --- | --- | --- | --- | --- |']
    for i, item in enumerate(results, 1):
        m = item['metrics']
        lines.append(f'| {i} | {item["name"]} | {m["stable_cases"]}/{m["cases"]} | '
                     f'{m["bilateral_swing_cases"]}/{m["moving_cases"]} | '
                     f'{m["mean_case_max_tilt_deg"]:.2f} / {m["worst_tilt_deg"]:.2f} |')
    (output / 'ranking.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(evaluated=len(results), failures=failures,
                         shortlist=[r['name'] for r in results[:2]])), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
