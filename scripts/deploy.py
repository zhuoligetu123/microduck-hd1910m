#!/usr/bin/env python3
"""Transfer the release with SSH. Does not restart services or alter calibration."""
import argparse
import ipaddress
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', required=True)
    parser.add_argument('--user', default='robot')
    parser.add_argument('--directory', default='~/workspace/microduck-hd1910m')
    args = parser.parse_args()
    ipaddress.ip_address(args.ip)
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_-]*', args.user):
        parser.error('Invalid SSH user')
    if not re.fullmatch(r'(~/|/)[a-zA-Z0-9_./-]+', args.directory) or '..' in Path(args.directory).parts:
        parser.error('Use an absolute or ~/ path without spaces or parent traversal')
    if not (ROOT / 'out/arm64/bin/robotd').is_file():
        parser.error('Run bash scripts/build.sh arm64 first')
    for role in ('walk', 'getup', 'pick', 'roulade'):
        if not (ROOT / f'radxa/references/reference_runtime_20261005/hd1910_{role}.onnx').is_file():
            parser.error('Run scripts/fetch_models.py first')
    host = f'{args.user}@{args.ip}'
    # rsync prompts for SSH credentials; no password is saved or embedded.
    subprocess.run(['ssh', host, 'test ! -e ' + args.directory + ' && mkdir -p ' + args.directory], check=True)
    items = ['scripts', 'radxa', 'out/arm64']
    for item in items:
        subprocess.run(['rsync', '-aR', '--exclude=__pycache__', '--exclude=installation.json', str(ROOT / './') + '/./' + item,
                        host + ':' + args.directory + '/'], check=True)
    print('Transferred only; existing services and calibration left unchanged.')
    print(f'ssh {shlex.quote(host)}')
    print(f'cd {args.directory}')
    print('python3 -m venv .venv && .venv/bin/pip install onnxruntime==1.24.4')
    print('python3 scripts/configure.py --calibration /absolute/path/to/installation.json')
    print('bash scripts/run_hardware.sh')


if __name__ == '__main__':
    main()
