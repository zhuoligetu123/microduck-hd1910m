#!/usr/bin/env python3
"""Record shipped asset hashes and validation evidence; no hardware access."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    urdf = ROOT / 'robot_description/microduck.urdf'
    root = ET.parse(urdf).getroot()
    meshes = [urdf.parent / mesh.attrib['filename'] for mesh in root.findall('.//mesh')]
    assert all(path.is_file() for path in meshes), 'URDF mesh missing'
    assets = {}
    for path in sorted((ROOT / 'release-assets').iterdir()):
        if path.is_file() and path.name != 'SHA256SUMS':
            assets[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    for name in ('app_preview.mp4', 'mujoco_preview.mp4'):
        result = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames:format=duration',
            '-of', 'json', str(ROOT / 'release-assets' / name)])
        data = json.loads(result)
        stream = data['streams'][0]
        assert (stream['width'], stream['height'], stream['r_frame_rate'], stream['nb_frames']) == (1920, 1080, '30/1', '5400')
        assert float(data['format']['duration']) == 180
        assets[name]['video'] = data
    text = ''.join(f'{data["sha256"]}  {name}\n' for name, data in assets.items())
    (ROOT / 'release-assets/SHA256SUMS').write_text(text)
    sim = json.loads((ROOT / 'out/tests/native_sim/summary.json').read_text())
    evidence = {
        'date': '2026-10-06', 'hardware_tested_this_release': False,
        'native_build': 'passed', 'arm64_cross_build': 'passed; Linux glibc >=2.35',
        'rust_tests': {'duck_control':128, 'robotd':145, 'robotd_integration':7,
                       'robotd_params':87, 'backend':7, 'ignored_external_model_tests':3},
        'python_contract_tests': 12, 'release_config_tests': 3,
        'four_graph_hash_contract_and_scheduler': 'passed with real ONNX graphs and native ORT',
        'native_sim_protocol': {key: sim[key] for key in (
            'physical_hardware', 'physics', 'gait_qualified', 'enable_home_without_rl',
            'stop_holds_enabled', 'skills', 'walk_command', 'mouth_target')},
        'release_launcher': 'passed, isolated simulation health online; no physical I/O',
        'apk': {'package':'com.microduck.control', 'version':'0.1.2', 'version_code':3,
                'signature':'APK v2 verified; internal debug-signed build',
                'source_uploaded':False},
        'urdf': {'sha256':digest(urdf), 'visual_mesh_references':len(meshes), 'all_resolve':True},
        'assets':assets,
        'limits': ['No new physical robot test', 'No new long training run',
                   'Native protocol simulation uses fixed base support, not a gait qualification',
                   'Front recovery and dynamic roll landing passed one free-body recording; see skill_validation.json',
                   'Other initial conditions still fail; no general recovery guarantee']}
    (ROOT / 'docs/release_validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'assets':len(assets), 'urdf_meshes':len(meshes), 'physical_hardware':False}))


if __name__ == '__main__':
    main()
