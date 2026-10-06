"""Bounded local release-launcher smoke, never uses hardware."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'out/tests'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'launcher.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_sim.py'),
                                    '--supported', '--bind', '127.0.0.1:38887'],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Launcher exited; inspect launcher.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:38887/api/health', timeout=1) as r:
                        health = json.load(r)
                    if health.get('online'):
                        break
                except OSError:
                    pass
                time.sleep(.2)
            else:
                raise RuntimeError('Native feedback did not become online')
            print(json.dumps({'launcher_ok': True, 'health': health, 'physical_hardware': False}))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
