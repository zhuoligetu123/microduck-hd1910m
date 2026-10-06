#!/usr/bin/env python3
"""Reference App/Rust/ONNX/MuJoCo protocol test. Never connects to hardware."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import websockets
import onnxruntime
from test_live import until

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'microduck_app'
REPORT = ROOT / 'out/tests/native_sim'


async def exercise():
    for _ in range(150):
        try:
            ws = await websockets.connect('ws://127.0.0.1:38886/ws', proxy=None)
            break
        except OSError:
            await asyncio.sleep(.1)
    else:
        raise RuntimeError('App server failed to start')
    replies = []
    async with ws:
        def frame(s):
            return s.get('data', {})
        ready = await until(ws, lambda s: s.get('online') and frame(s).get('feedback', {}).get('reference_native'), seconds=20)
        assert set(ready['skills']) == {'stand', 'walk', 'ground_pick', 'recovery', 'roulade'}, ready
        assert not frame(ready)['feedback']['policy_enabled']
        async def send(i, **cmd):
            await ws.send(json.dumps({'id': i, **cmd}))
            answer = await until(ws, lambda s: s.get('id') == i)
            replies.append(answer)
            assert answer['accepted'], answer
        await send(10, type='enable')
        home = await until(ws, lambda s: frame(s).get('feedback', {}).get('homed'), seconds=15)
        assert frame(home)['policy'] == 'held' and frame(home).get('inference') is None
        await send(11, type='mouth', open=.5)
        mouth = await until(ws, lambda s: frame(s).get('targets', [0]*15)[9] > .2)
        await send(12, type='head', neck_pitch=0, head_pitch=0, head_yaw=.1, head_roll=0)
        await until(ws, lambda s: frame(s).get('policy') == 'walk')
        for seq in range(12):
            await ws.send(json.dumps({'type': 'move', 'seq': seq+1, 'vx': .05, 'vy': 0, 'vyaw': .1}))
            await asyncio.sleep(.05)
        walk = await until(ws, lambda s: frame(s).get('inference') is not None and frame(s).get('move', {}).get('applied', [0])[0] > .01)
        await send(40, type='enable')
        await until(ws, lambda s: frame(s).get('policy') == 'homing')
        await until(ws, lambda s: frame(s).get('policy') == 'held' and frame(s).get('feedback', {}).get('homed'), seconds=8)
        await send(13, type='stop')
        await until(ws, lambda s: frame(s).get('policy') == 'held')
        modes = []
        for i, name in enumerate(['ground_pick', 'recovery', 'roulade'], 20):
            await send(i, type='skill', name=name)
            active = await until(ws, lambda s: frame(s).get('policy') == name, seconds=8)
            assert not active['head_control_available']
            modes.append({'mode': name, 'observation': frame(active)['inference']['observation'],
                          'target_written': frame(active)['target_written']})
            await send(i+10, type='stop')
            await until(ws, lambda s: frame(s).get('policy') == 'held')
        await ws.send(json.dumps({'id': 50, 'type': 'skill', 'name': 'sit'}))
        missing = await until(ws, lambda s: s.get('id') == 50)
        assert not missing['accepted']
        return {'physical_hardware': False, 'physics': 'MuJoCo BAM M6 with fixed base support',
                'gait_qualified': False, 'enable_home_without_rl': True, 'stop_holds_enabled': True,
                'skills': ready['skills'], 'unavailable_refused': missing, 'replies': replies, 'modes': modes,
                'walk_command': frame(walk)['move']['applied'], 'mouth_target': frame(mouth)['targets'][9]}


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    processes, logs = [], []
    with tempfile.TemporaryDirectory(prefix='reference-native-') as folder:
        bundle = Path(folder)
        base = ROOT / 'sim'
        for name in ['hd1910.mjb', 'motor_calibration.json']:
            shutil.copy2(base / name, bundle / name)
        walk = ROOT / 'radxa/references/reference_runtime_20261005/hd1910_walk.onnx'
        shutil.copy2(walk, bundle / 'policy.onnx')
        physics = json.loads((base / 'physics.json').read_text())
        physics['policy_sha256'] = hashlib.sha256(walk.read_bytes()).hexdigest()
        (bundle / 'physics.json').write_text(json.dumps(physics))
        config = (ROOT / 'radxa/reference_native.toml').read_text().replace('/home/robot/workspace/huggingface', str(ROOT))
        config = config.replace(f'feetech:{ROOT}/radxa/reference_native.json', 'sim:127.0.0.1:17803')
        config = config.replace(str(walk), str(bundle / 'policy.onnx'))
        (bundle / 'params.toml').write_text(config)
        env = dict(os.environ, PYTHONPATH=str(ROOT/'microduck_rl/src'),
                   ORT_DYLIB_PATH=str(next((Path(onnxruntime.__file__).parent/'capi').glob('libonnxruntime.so.*'))),
                   MICRODUCK_APP_BIND='127.0.0.1:38886', MICRODUCK_ROBOTD_SOCKET=str(bundle/'robotd.sock'))
        env.pop('MICRODUCK_APP_TOKEN', None)
        commands = [
            [sys.executable, '-m', 'mjlab_microduck.sim.hd1910_body', '--bundle', str(bundle), '--port', '17803', '--supported-start'],
            [str(ROOT/'out/native/bin/robotd'), '--params', str(bundle/'params.toml'), '--socket', str(bundle/'robotd.sock')],
            [str(ROOT/'out/native/bin/microduck-app-server')]]
        try:
            for name, cmd in zip(['body','robotd','app'], commands):
                log = (REPORT/f'{name}.log').open('w'); logs.append(log)
                processes.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=APP))
            result = asyncio.run(exercise())
            (REPORT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result))
        finally:
            for process in reversed(processes): process.terminate()
            for process in processes:
                try: process.wait(timeout=8)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            for log in logs: log.close()


if __name__ == '__main__':
    main()
