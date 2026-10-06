#!/usr/bin/env python3
"""Audit the git index and write a content manifest without credentials or logs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    if not any(files):
        raise SystemExit('Empty index: run git add before auditing')
    hashes, findings = {}, []
    blocked_parts = {'target', '.venv', 'node_modules', 'training_runs', 'logs', 'reports', '__pycache__'}
    forbidden = ('microduck_app/web/', 'microduck_app/android/', 'release-assets/')
    # Print only paths, never captured credential values.
    secrets = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----)')
    for name in filter(None, files):
        path = ROOT / name
        if (blocked_parts.intersection(path.relative_to(ROOT).parts) or name.startswith(forbidden)
                or path.suffix in {'.apk', '.mp4', '.pt', '.ckpt', '.jks', '.keystore', '.pem', '.key'}):
            findings.append(name)
        if path.is_symlink():
            findings.append('symlink: ' + name)
            continue
        data = path.read_bytes()
        if len(data) >= 100 * 1024 * 1024 or secrets.search(data):
            findings.append('size or credential: ' + name)
        if name != 'docs/source_manifest.json':
            hashes[name] = hashlib.sha256(data).hexdigest()
    if findings:
        raise SystemExit(json.dumps({'blocked': findings}, indent=2))
    (ROOT / 'docs/source_manifest.json').write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'tracked_files': len(hashes), 'app_source_files': 0, 'secret_patterns': 0,
                      'training_artifacts': 0, 'source_bytes': sum((ROOT / n).stat().st_size for n in hashes)}))


if __name__ == '__main__':
    main()
