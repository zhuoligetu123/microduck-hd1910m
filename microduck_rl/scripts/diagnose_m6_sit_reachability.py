#!/usr/bin/env python3
"""Check whether symmetric sit targets remain upright in the M6 CPU model."""
import argparse
import hashlib
import json
import math
import os
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

from replay_hd1910 import DEFAULT_POSE, load_replay_model, step_control_period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('MICRODUCK_BAM_KP') != '6' or not os.environ.get('MICRODUCK_BAM_PROFILE'):
        parser.error('set MICRODUCK_BAM_KP=6 and MICRODUCK_BAM_PROFILE to the M6 profile')
    profile = Path(os.environ['MICRODUCK_BAM_PROFILE']).resolve()

    model, data, motor = load_replay_model(7.4, bam_reference=True, repair_variant='sitstand')
    stand = DEFAULT_POSE.copy()
    rows = []
    hip_values = (-1.4, -1.1, -.8, -.6, -.5, -.4, -.2, 0.)
    knee_values = (.3, .6, .9, 1., 1.2, 1.4)
    ankle_values = (-.4, -.25, 0., .25, .4)
    combinations = list(product(hip_values, knee_values, ankle_values))
    for hip, knee, ankle in combinations:
        target = stand.copy()
        target[1] = target[10] = 0.
        target[2], target[11] = hip, -hip
        target[3], target[12] = knee, -knee
        target[4], target[13] = ankle, -ankle
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0., 0., .125, 1., 0., 0., 0.]
        data.qpos[motor.qids] = stand
        mujoco.mj_forward(model, data)
        motor.reset(data.qpos)
        for step in range(250):
            blend = min(max((step - 25) / 100, 0.), 1.)
            motor.q_target = stand * (1. - blend) + target * blend
            step_control_period(model, data, motor)
        _, qx, qy, _ = data.qpos[3:7]
        tilt = math.degrees(math.acos(float(np.clip(1. - 2. * (qx*qx + qy*qy), -1., 1.))))
        rows.append(dict(hip_pitch_rad=hip, knee_rad=knee, ankle_rad=ankle,
                         trunk_height_m=float(data.qpos[2]), tilt_deg=tilt))
    acceptable = [row for row in rows if abs(row['trunk_height_m'] - .06) <= .015
                  and row['tilt_deg'] < 20.]
    report = dict(voltage_v=7.4, model='M6 P6 groundcontact',
                  profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), candidates=len(rows),
                  target_height_m=.06, acceptable_count=len(acceptable),
                  best_by_tilt=sorted(rows, key=lambda row: row['tilt_deg'])[:5],
                  rows=rows, hardware_opened=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('candidates', 'acceptable_count', 'best_by_tilt')}))


if __name__ == '__main__':
    main()
