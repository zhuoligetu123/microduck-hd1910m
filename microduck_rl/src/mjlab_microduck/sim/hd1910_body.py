"""HD1910 body on the upstream sim protocol. CPU physics, no hardware access.

Uses the same provisional motor controller and exported contact model as CPU
replay. Mouth has a protocol slot but no physical joint in this walking model.
Voltage/temperature are synthetic; they do not validate electrical behaviour.
"""
import argparse
import hashlib
import json
import threading
import tomllib
from pathlib import Path
import mujoco
import numpy as np
from .body_server import World, Body, Server, Handler, run, HOME_TRUNK_Z
from ..actuator.cpu_hd1910 import Hd1910CpuController, PROFILE_PATH


class HdWorld(World):
    def __init__(self, bundle):
        meta = json.loads((bundle / 'physics.json').read_text())
        if meta.get('bundle_schema', 1) not in (1, 2):
            raise ValueError('unsupported simulation bundle schema')
        profile_path, controller = PROFILE_PATH, Hd1910CpuController
        backend = meta.get('actuator_backend', 'hd1910_reference_pd')
        if backend == 'hd1910_bam_m6':
            from ..actuator.cpu_hd1910_bam import PROFILE_PATH as profile_path, XgoBamCpuController as controller
        elif backend != 'hd1910_reference_pd':
            raise ValueError('unsupported actuator backend')
        if meta.get('bundle_schema') == 2:
            if meta.get('profile_file') != 'motor_calibration.json':
                raise ValueError('invalid bundled profile name')
            profile_path = bundle/'motor_calibration.json'
        if 'policy_file' in meta:
            if meta['policy_file'] != 'policy.onnx':
                raise ValueError('invalid bundled policy name')
            if hashlib.sha256((bundle/'policy.onnx').read_bytes()).hexdigest() != meta['policy_sha256']:
                raise ValueError('policy checksum mismatch')
        path = bundle / 'hd1910.mjb'
        if meta['mujoco'] != mujoco.__version__:
            raise ValueError('MJB requires the exporting MuJoCo version')
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('MJB checksum mismatch')
        if hashlib.sha256(profile_path.read_bytes()).hexdigest() != meta['profile_sha256']:
            raise ValueError('actuator profile mismatch')
        self.model = mujoco.MjModel.from_binary_path(str(path))
        if 'physics_hz' in meta and not np.isclose(self.model.opt.timestep * meta['physics_hz'], 1.):
            raise ValueError('MJB timestep differs from manifest')
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.bodies = []
        self.state_log = None
        if backend == 'hd1910_bam_m6':
            self.motor = controller(self.model, self.data, meta['voltage'], meta['delay_steps'],
                                    profile_path=profile_path, kp_fw=meta.get('kp_fw', 5.))
        else:
            # The legacy PD controller imports its coefficients at module load.
            if hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() != meta['profile_sha256']:
                raise ValueError('PD runtime profile differs from bundled profile')
            self.motor = controller(self.model, self.data, meta['voltage'], meta['delay_steps'])

    def step(self, times=1):
        with self.lock:
            for _ in range(times):
                body = self.bodies[0]
                if body.torque_on:
                    self.motor.update()
                else:
                    self.data.ctrl[:] = 0
                mujoco.mj_step(self.model, self.data)
                if not body.released:
                    body.restore()
            mujoco.mj_forward(self.model, self.data)
            if self.state_log is not None:
                self.state_log.write(json.dumps(dict(sim_time=float(self.data.time),
                    qpos=self.data.qpos.tolist(), supported=not body.released,
                    enabled=body.torque_on), separators=(',', ':'))+'\n')


class HdBody(Body):
    supported_start = False
    def place(self, *args, **kwargs):
        super().place(*args, **kwargs)
        self.world.motor.reset(self.world.data.qpos)
        self.world.data.ctrl[:] = 0

    def _apply_torque(self):
        # The shared controller uses physical PD units, not XL330 register gain.
        if not self.torque_on:
            self.world.data.ctrl[:] = 0

    def set_gain(self, kp):
        if not isinstance(kp, int) or not 0 <= kp <= 200:
            raise ValueError('unsupported simulation gain')
        with self.world.lock:
            self.kp = kp
            if hasattr(self.world.motor, 'set_gain'):
                self.world.motor.set_gain(kp)
                return
            from ..actuator.cpu_hd1910 import SIM
            self.world.motor.stiffness = SIM['stiffness_nm_per_rad'] * kp / 200

    def set_targets(self, targets):
        values = np.asarray(targets, dtype=float)
        if values.shape != (15,) or not np.isfinite(values).all():
            raise ValueError('expected 15 finite targets')
        with self.world.lock:
            self.world.motor.q_target[self.actuator_slice] = values[self.to_wire]

    def set_torque(self, on):
        with self.world.lock:
            self.torque_on = bool(on)
            if on:
                self.released = not self.supported_start
                self.world.motor.reset(self.world.data.qpos)
            self._apply_torque()


class HdHandler(Handler):
    def dispatch(self, body, request):
        if request.get('op') == 'release_start_support':
            with body.world.lock:
                body.released = True
            return {}
        return super().dispatch(body, request)


def validate_launch(bundle, port):
    meta = json.loads((bundle/'physics.json').read_text())
    if 'policy_file' not in meta:
        return # Legacy physics-only exports have no model-binding contract.
    config = tomllib.loads((bundle/'params.toml').read_text())
    if config['bus']['port'] != f'sim:127.0.0.1:{port}':
        raise ValueError('bound bundle requires its isolated simulation endpoint')
    if Path(config['policy']['walk']).resolve() != (bundle/'policy.onnx').resolve():
        raise ValueError('runtime policy path differs from bundled policy')
    if config['control']['hz'] != meta['physics_hz']/4:
        raise ValueError('runtime control frequency differs from exported 50 Hz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--port', type=int, default=17801)
    parser.add_argument('--supported-start', action='store_true',
                        help='Hold initial pose through native homing; test client explicitly releases it')
    parser.add_argument('--check-only', action='store_true', help='Validate files/config without opening a socket')
    parser.add_argument('--state-log', type=Path, help='Record simulation qpos for offline video; never rerun policy')
    args = parser.parse_args()
    validate_launch(args.bundle, args.port)
    world = HdWorld(args.bundle)
    if args.check_only:
        print('Simulation bundle files and runtime binding verified')
        return
    if args.state_log:
        world.state_log = args.state_log.open('x', buffering=1)
    body = HdBody(world, 0)
    body.supported_start = args.supported_start
    body.place(None, HOME_TRUNK_Z, 0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    with Server(('127.0.0.1', args.port), HdHandler) as server:
        server.body = body
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'HD1910 SIMULATION ONLY on 127.0.0.1:{args.port}', flush=True)
        run(world, headless=True)


if __name__ == '__main__':
    main()
