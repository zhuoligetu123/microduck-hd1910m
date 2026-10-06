#!/usr/bin/env python3
"""Verify actual GPU inertias are per-environment and resets do not accumulate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg
from mjlab_microduck.actuator.payload_uncertainty import PROFILE, mass_contract
from mjlab_microduck.actuator.reference_hd1910 import _rot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_payload_v8')
    cfg.scene.num_envs = 64
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        env.reset(seed=42)
        model = env.sim.mj_model
        initial = env.sim.model.body_mass.clone()
        rows = {}
        for name, (lo, hi) in PROFILE['body_mass_scale_ranges'].items():
            bid = model.body('robot/' + name).id
            scale = initial[:, bid].cpu().numpy() / model.body_mass[bid]
            assert np.isfinite(scale).all()
            assert scale.min() >= lo - 1e-5 and scale.max() <= hi + 1e-5
            assert np.unique(scale).size > 1, name
            inertia = env.sim.model.body_inertia[:, bid].cpu().numpy()
            rotations = [_rot(q) for q in env.sim.model.body_iquat[:, bid].cpu().numpy()]
            actual = np.array([r @ np.diag(i) @ r.T for r, i in zip(rotations, inertia)])
            r = _rot(model.body_iquat[bid])
            expected = r @ np.diag(model.body_inertia[bid]) @ r.T
            np.testing.assert_allclose(actual, scale[:, None, None]*expected,
                                       rtol=2e-4, atol=1e-9)
            rows[name] = dict(scale_min=float(scale.min()), scale_max=float(scale.max()),
                              unique_values=int(np.unique(scale).size))
        for _ in range(20):
            env.reset()
            for event in ('randomize_com', 'randomize_head_com'):
                term = env.event_manager.get_term_cfg(event)
                selector = term.params['asset_cfg']
                ids = env.scene['robot'].indexing.body_ids[selector.body_ids]
                default = env.sim.get_default_field('body_ipos')[ids]
                delta = env.sim.model.body_ipos[:, ids] - default
                lo, hi = term.params['ranges']
                assert torch.isfinite(delta).all()
                assert delta.min() >= lo - 1e-6 and delta.max() <= hi + 1e-6
        assert torch.equal(initial, env.sim.model.body_mass)
        report = dict(passed=True, environments=64, resets=20,
                      body_mass_unchanged_after_resets=True, com_offsets_bounded=True, bodies=rows,
                      mass_contract=mass_contract(), hardware_tested=False)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('bodies', 'mass_contract')}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
