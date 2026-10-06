"""Reference policy contract on local HD1910/BNO085 calibration; no hardware I/O.

Encoder zero is NOT policy HOME. Quaternion order is wxyz. Native body-frame
IMU data bypasses mounting rotation; raw BNO085 data is rotated exactly once.
The caller owns transport, feedback freshness, travel limits and enable state.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'references/reference_runtime_20261005'
INSTALLATION = ROOT / 'installation.json'
ROLES = ('walk', 'getup', 'pick', 'roulade')
JOINTS = ('left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
          'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
          'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle')
RAD_TICK = 2 * math.pi / 4096
RAD_SPEED = .732 * 2 * math.pi / 60
OBS_NAMES = 'base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command'


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'expected finite vector[{size}]')
    return result


def quat(value):
    result = vector(value, 4)
    length = np.linalg.norm(result)
    if length < 1e-8:
        raise ValueError('zero quaternion')
    return result / length


def conjugate(q):
    return quat(q) * [1, -1, -1, -1]


def multiply(a, b):
    aw, ax, ay, az = quat(a)
    bw, bx, by, bz = quat(b)
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def rotate(q, value):
    q = quat(q)
    value = vector(value, 3)
    return value + 2 * np.cross(q[1:], np.cross(q[1:], value) + q[0]*value)


class LocalCalibration:
    def __init__(self, path=INSTALLATION):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.config = json.loads(raw)
        rows = self.config['joints']
        self.joints = {row['name']: row for row in rows}
        if (len(rows) != 15 or set(self.joints) != set(JOINTS) | {'mouth'}
                or len({row['id'] for row in rows}) != 15):
            raise ValueError('expected 15 distinct named joints and IDs')
        for row in rows:
            if (type(row['id']) is not int or not 1 <= row['id'] <= 253
                    or row['direction'] not in (-1, 1)
                    or type(row['zero_ticks']) is not int
                    or not 0 <= row['zero_ticks'] <= 4095):
                raise ValueError('invalid installation calibration')
        self.mount = quat(self.config['imu_mount_wxyz'])

    def unchanged(self):
        return hashlib.sha256(self.path.read_bytes()).hexdigest() == self.sha256

    def requested_ticks(self, positions, mouth=0.):
        values = dict(zip(JOINTS, vector(positions, 14)))
        values['mouth'] = float(mouth)
        result = {}
        for name, q in values.items():
            row = self.joints[name]
            ticks = row['zero_ticks'] + q / RAD_TICK * row['direction']
            if not math.isfinite(ticks):
                raise ValueError('nonfinite target')
            result[row['id']] = ticks
        return result

    def target_ticks(self, positions, mouth=0., *, saturate=False):
        result = {}
        for servo_id, ticks in self.requested_ticks(positions, mouth).items():
            if saturate:
                # Pinned Reference scs_bus.h satPos; never wrap across the zero seam.
                ticks = float(np.clip(ticks, 0, 4095))
            elif not 0 <= ticks <= 4095:
                raise ValueError(f'ID{servo_id}: target outside single-turn encoder domain: {ticks}')
            result[servo_id] = int(math.floor(ticks + .5))
        return result

    def positions(self, ticks):
        result = []
        for name in JOINTS:
            row = self.joints[name]
            raw = ticks[row['id']]
            if not math.isfinite(raw):
                raise ValueError('nonfinite encoder sample')
            result.append((raw-row['zero_ticks'])*RAD_TICK*row['direction'])
        return np.asarray(result)

    def velocities(self, speed_words):
        result = []
        for name in JOINTS:
            row = self.joints[name]
            word = speed_words[row['id']]
            if type(word) is not int or not 0 <= word <= 65535:
                raise ValueError('invalid HD1910 signed-magnitude speed word')
            speed = -(word & 0x7fff) if word & 0x8000 else word
            result.append(speed * RAD_SPEED * row['direction'])
        return np.asarray(result)

    def simulate_encoders(self, q, dq):
        ticks = self.target_ticks(q)
        speeds = {}
        for name, speed in zip(JOINTS, vector(dq, 14)):
            row = self.joints[name]
            raw = int(round(speed/RAD_SPEED*row['direction']))
            if abs(raw) > 32767:
                raise ValueError('speed cannot be represented by feedback protocol')
            speeds[row['id']] = abs(raw) | (0x8000 if raw < 0 else 0)
        return ticks, speeds

    def raw_imu_to_body(self, world_sensor, gyro_sensor):
        world_body = multiply(world_sensor, conjugate(self.mount))
        return (rotate(self.mount, gyro_sensor),
                rotate(conjugate(world_body), [0, 0, -1]))

    def simulate_imu(self, world_body, gyro_body):
        return (multiply(world_body, self.mount),
                rotate(conjugate(self.mount), gyro_body))


class ReferencePolicy:
    def __init__(self, role, directory=MODEL_DIR):
        if role not in ROLES:
            raise ValueError(f'no published policy for {role}; available: {ROLES}')
        self.role = role
        self.path = Path(directory) / f'hd1910_{role}.onnx'
        manifest = json.loads((Path(directory)/'manifest.json').read_text())
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != manifest['models'][role]['sha256']:
            raise ValueError('model hash differs from pinned manifest')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        if (self.session.get_inputs()[0].shape != [1, 61]
                or self.session.get_outputs()[0].shape != [1, 14]
                or self.metadata['joint_names'].split(',') != list(JOINTS)
                or self.metadata['observation_names'] != OBS_NAMES):
            raise ValueError('unsupported policy observation/action contract')
        self.home = vector([float(v) for v in self.metadata['default_joint_pos'].split(',')], 14)
        scale = [float(v) for v in self.metadata['action_scale'].split(',')]
        self.scale = vector(scale * 14 if len(scale) == 1 else scale, 14)
        self.alpha = .15 if role == 'roulade' else .45
        self.reset()

    def reset(self):
        self.previous = np.zeros(14)
        self.filtered = np.zeros(14)

    def command(self, elapsed, twist=(0, 0, 0), head=(0, 0, 0, 0)):
        elapsed = float(elapsed)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('elapsed must be finite and nonnegative')
        if self.role == 'pick':
            phase = min(elapsed / 4., 1.)
            return (np.array([math.cos(2*math.pi*phase), math.sin(2*math.pi*phase), 0]),
                    np.zeros(4), math.radians(30) if phase < .4 else 0.)
        if self.role != 'walk':
            return np.zeros(3), np.zeros(4), 0.
        return vector(twist, 3), vector(head, 4), 0.

    def observation(self, q, dq, gyro_body, gravity_body, elapsed, twist=(0, 0, 0),
                    head=(0, 0, 0, 0)):
        command, head_command, mouth = self.command(elapsed, twist, head)
        gravity = vector(gravity_body, 3)
        if not .95 <= np.linalg.norm(gravity) <= 1.05:
            raise ValueError('projected gravity must be unit gravity, not m/s^2')
        obs = np.concatenate((vector(gyro_body, 3), gravity, vector(q, 14)-self.home,
                              vector(dq, 14), self.previous, command,
                              head_command, np.zeros(6))).astype(np.float32)
        return obs, mouth

    def infer(self, *args, **kwargs):
        obs, mouth = self.observation(*args, **kwargs)
        action = self.session.run(None, {self.session.get_inputs()[0].name: obs[None]})[0][0]
        vector(action, 14)
        self.previous[:] = action
        self.filtered = self.alpha*self.filtered + (1-self.alpha)*action
        return self.home + self.scale*self.filtered, mouth, obs


class BodyFeedbackFilter:
    """Upstream gyro/velocity EMA after local mounting, in rad/s.

    BNO085 already supplies fused orientation. Do not run the QMI8658
    accelerometer-based gravity estimator over that quaternion a second time.
    """
    def __init__(self):
        self.stamp = None

    def update(self, stamp, gyro, velocity):
        gyro, velocity = vector(gyro,3), vector(velocity,14)
        if not math.isfinite(stamp) or (self.stamp is not None and stamp <= self.stamp):
            raise ValueError('nonmonotonic feedback')
        if self.stamp is None or stamp-self.stamp > .1:
            self.gyro, self.velocity = gyro.copy(), velocity.copy()
        else:
            dt = stamp-self.stamp
            ga, va = .5**(dt/.01), .4**(dt/.01)
            self.gyro = ga*self.gyro + (1-ga)*gyro
            self.velocity = va*self.velocity + (1-va)*velocity
        self.stamp = stamp
        return self.gyro.copy(), self.velocity.copy()


class ReferenceSuite:
    """Explicit task selection; episodic skills return to idle WALK.

    Get-up remains selected until the caller confirms actual supported recovery.
    No hidden fall-triggered auto-restart or hardware commands.
    """
    def __init__(self, directory=MODEL_DIR):
        self.policies = {role: ReferencePolicy(role, directory) for role in ROLES}
        self.select('walk', 0.)

    def select(self, role, now):
        if role not in self.policies:
            raise ValueError(f'unavailable model: {role}')
        if not math.isfinite(now):
            raise ValueError('nonfinite timestamp')
        self.role, self.started = role, now
        self.upright_elapsed = 0.
        self.policies[role].reset()

    def recovery_ready(self, gravity_body, now, dt=.02):
        """Reference get-up exit: tilt below 15 degrees continuously for one second."""
        if self.role != 'getup':
            return False
        gravity = vector(gravity_body, 3)
        if not .95 <= np.linalg.norm(gravity) <= 1.05 or not 0 < dt <= .1:
            self.upright_elapsed = 0.
            return False
        tilt = math.degrees(math.acos(float(np.clip(-gravity[2]/np.linalg.norm(gravity),-1,1))))
        self.upright_elapsed = self.upright_elapsed+dt if tilt < 15 else 0.
        if self.upright_elapsed >= 1.-1e-9:
            self.select('walk', now)
            return True
        return False

    def advance(self, now):
        elapsed = now - self.started
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError('nonmonotonic task time')
        duration = {'pick': 4., 'roulade': 1.9}.get(self.role)
        if duration is not None and elapsed >= duration - 1e-9:
            self.select('walk', now)
            elapsed = 0.
        return self.policies[self.role], elapsed
