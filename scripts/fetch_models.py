#!/usr/bin/env python3
"""Fetch pinned upstream weights with SHA256 verification; never activate a robot."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'radxa/references/reference_runtime_20261005'


def fetch(directory=MODELS):
    manifest = json.loads((MODELS / 'manifest.json').read_text())
    directory.mkdir(parents=True, exist_ok=True)
    for role, details in manifest['models'].items():
        name = details['file']
        path = directory / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == details['sha256']:
            print(f'OK {name}')
            continue
        repository = manifest['repository'].removeprefix('https://github.com/')
        url = (f'https://raw.githubusercontent.com/{repository}/'
               f'{manifest["commit"]}/{manifest["source_directory"]}/{details["source_file"]}')
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(8 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != details['sha256']:
            raise RuntimeError(f'Checksum mismatch: {name}')
        temporary = path.with_suffix('.part')
        temporary.write_bytes(data)
        temporary.replace(path)
        print(f'DOWNLOADED {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MODELS)
    fetch(parser.parse_args().output)
