"""Confirmed component inventory and explicitly uncalibrated mass sensitivity.

CAD inertias are aggregated by link. Without the old component masses, adding
the replacement hardware would double count it. Preserve the nominal CAD;
train/evaluate around it until individual or assembled link masses are known.
"""
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

PROFILE_PATH = Path(__file__).with_name('replica_mass_20261002.json')
PROFILE = json.loads(PROFILE_PATH.read_text())
SCENARIOS = ('cad_nominal', 'head_heavy', 'back_heavy', 'light_links')


def mass_contract():
    return dict(profile_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
                **PROFILE)


def configure_payload_uncertainty(cfg):
    from mjlab.envs.mdp import dr
    from mjlab.managers import EventTermCfg, SceneEntityCfg

    # Replace, rather than stack, the old trunk-only +/-5% mass randomization.
    cfg.events.pop('randomize_mass_inertia', None)
    for body, (lo, hi) in PROFILE['body_mass_scale_ranges'].items():
        cfg.events['payload_mass_' + body] = EventTermCfg(
            func=dr.pseudo_inertia, mode='startup', params={
                'asset_cfg': SceneEntityCfg('robot', body_names=(body,)),
                'alpha_range': (math.log(lo)/2, math.log(hi)/2)})
    for event, curriculum, key in (
        ('randomize_com', 'com_range', 'trunk_com_offset_m'),
        ('randomize_head_com', 'head_com_range', 'head_com_offset_m')):
        cfg.curriculum.pop(curriculum, None)
        bound = PROFILE[key]
        cfg.events[event].params['ranges'] = (-bound, bound)


def apply_mass_scenario(model, data, scenario):
    """Apply once to a fresh replay model; these are hypotheses, not a new CAD.

    Uniform link-density scaling preserves positive, physically valid inertia.
    COM shifts use each body's CAD frame, not an assumed world forward axis.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f'unknown mass scenario: {scenario}')
    before = float(model.body_mass.sum())
    scales = {}
    offsets = {}
    if scenario == 'head_heavy':
        scales = {'jaw_soft': 1.2, 'trunk_base': .8}
        offsets = {'jaw_soft': [.008, 0., 0.]}
    elif scenario == 'back_heavy':
        scales = {'jaw_soft': .8, 'trunk_base': 1.2}
        offsets = {'trunk_base': [-.008, 0., 0.]}
    elif scenario == 'light_links':
        scales = {name: .9 for name in PROFILE['body_mass_scale_ranges']
                  if name not in ('jaw_soft', 'trunk_base')}
    for name, scale in scales.items():
        bid = model.body(name).id
        model.body_mass[bid] *= scale
        model.body_inertia[bid] *= scale
    for name, offset in offsets.items():
        model.body_ipos[model.body(name).id] += np.asarray(offset)
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return dict(scenario=scenario, nominal_mass_kg=before,
                scenario_mass_kg=float(model.body_mass.sum()),
                link_scales=scales, link_frame_com_offsets_m=offsets,
                mass_profile_sha256=mass_contract()['profile_sha256'],
                physical_mass_calibrated=False)
