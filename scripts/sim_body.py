#!/usr/bin/env python3
"""Serve simulated joints/IMU to robotd, optionally with a MuJoCo viewer."""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'microduck_rl/src'))


def main():
    import mujoco
    from mjlab_microduck.sim.hd1910_body import HdWorld, HdBody, HdHandler, validate_launch
    from mjlab_microduck.sim.body_server import Server, run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--supported', action='store_true', help='Protocol tests only, not gait validation')
    args = parser.parse_args()
    bundle = ROOT / 'sim'
    validate_launch(bundle, 17803)
    world = HdWorld(bundle)
    body = HdBody(world, 0)
    body.supported_start = args.supported
    names = [world.model.joint(int(j)).name for j in world.motor.joint_ids]
    home = json.loads((bundle / 'home.json').read_text())
    body.place(dict(zip(names, home)), .15, 0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    with Server(('127.0.0.1', 17803), HdHandler) as server:
        server.body = body
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print('SIMULATION ONLY: no hardware access', flush=True)
        run(world, headless=not args.viewer)


if __name__ == '__main__':
    main()
