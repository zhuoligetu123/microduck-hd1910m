#!/usr/bin/env python3
"""Run APK -> Rust backend -> native robotd -> local MuJoCo, no physical I/O."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from configure import ROOT, configure


def main():
    import onnxruntime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--supported', action='store_true')
    parser.add_argument('--bind', default='127.0.0.1:38880', help='0.0.0.0:38880 for APK on trusted LAN')
    args = parser.parse_args()
    if not (ROOT / 'sim/hd1910.mjb').exists():
        raise SystemExit('Run scripts/build_sim.py first')
    configure(ROOT / 'sim', sim_port=17803)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'microduck_rl/src'),
               ORT_DYLIB_PATH=str(next((Path(onnxruntime.__file__).parent / 'capi').glob('libonnxruntime.so.*'))),
               MICRODUCK_ROBOTD_SOCKET=str(ROOT / 'sim/robotd.sock'),
               MICRODUCK_APP_BIND=args.bind, MICRODUCK_APP_ALLOW_OPEN_LAN='1')
    env.pop('MICRODUCK_APP_TOKEN', None)
    flags = (['--viewer'] if args.viewer else []) + (['--supported'] if args.supported else [])
    commands = [
        [sys.executable, str(ROOT / 'scripts/sim_body.py'), *flags],
        [str(ROOT / 'out/native/bin/robotd'), '--params', str(ROOT / 'sim/params.toml'),
         '--socket', env['MICRODUCK_ROBOTD_SOCKET']],
        [str(ROOT / 'out/native/bin/microduck-app-server')]]
    processes = []
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        for cmd in commands:
            processes.append(subprocess.Popen(cmd, env=env, cwd=ROOT))
        print(f'APK endpoint: http://{args.bind}; no browser UI source is distributed', flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(.2)
        raise RuntimeError('A subprocess exited; see the preceding error')
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == '__main__':
    main()
