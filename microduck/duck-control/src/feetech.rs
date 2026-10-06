//! HD1910M mode 4 directly implementing RobotIo. No Python or motor RPC.
//! FT-SCS packets, not XL330 registers. Hardware writes require an explicit
//! configuration gate; startup and Drop do not enable or disable the robot.
//! rustypot 1.6's V1 parser discards status errors, so this small transport
//! validates both the packet alarm and register 65 instead of hiding faults.
use crate::bno08x::Bno08x;
use crate::io::{IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::{JOINT_NAMES, NUM_JOINTS};
use serde::{Deserialize, Serialize};
use serialport::SerialPort;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

const RAD_TICK: f64 = std::f64::consts::TAU / 4096.;
const CONTROL_PERIOD: Duration = Duration::from_millis(20);
const WRITE_TURN: Duration = Duration::from_millis(25);
const GOAL_WRITE_BUDGET: Duration = Duration::from_millis(3);
const SCHEDULED_PERIOD: Duration = Duration::from_millis(10);
const SCHEDULED_READ_BUDGET: Duration = Duration::from_millis(12);
const SCHEDULED_WRITE_ACK: Duration = Duration::from_millis(20);
fn err(e: impl std::fmt::Display) -> IoError {
    IoError::Bus(format!("Feetech: {e}"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub luwu_native: bool,
    pub port: String,
    pub installation: String,
    pub imu_bus: String,
    pub imu_address: u16,
    #[serde(default)]
    pub allow_motion: bool,
    #[serde(default)]
    pub servo_gain_profile: Option<ServoGainProfile>,
    #[serde(default)]
    pub scheduled_bus: bool,
}
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServoGainProfile { LuwuRuntime }

impl ServoGainProfile {
    fn rows(self, joints: &[Joint]) -> Vec<Vec<u8>> {
        joints.iter().map(|j| vec![if j.id == 15 { 10 } else { 6 }, 20]).collect()
    }
}
#[derive(Clone, Deserialize, Serialize)]
pub struct Joint {
    pub name: String,
    pub id: u8,
    pub zero_ticks: i32,
    pub direction: i32,
}
#[derive(Deserialize)]
struct Installation {
    joints: Vec<Joint>,
    imu_mount_wxyz: [f64; 4],
    #[serde(default)]
    calibration_verified: bool,
    #[serde(default)]
    imu_pose_verified: bool,
}
impl Installation {
    fn validate(&self) -> Result<()> {
        let mut ids = std::collections::HashSet::new();
        if self.joints.len() != NUM_JOINTS {
            return Err(err("expected 15 calibrated joints"));
        }
        for (j, name) in self.joints.iter().zip(JOINT_NAMES) {
            if j.name != name
                || !ids.insert(j.id)
                || !(1..=253).contains(&j.id)
                || ![-1, 1].contains(&j.direction)
                || !(0..4096).contains(&j.zero_ticks)
            {
                return Err(err("invalid joint order/id/zero/direction"));
            }
        }
        Ok(())
    }
}

fn load_config(path: &str) -> Result<(Config, Installation)> {
    let cfg: Config = serde_json::from_slice(&fs::read(path).map_err(err)?).map_err(err)?;
    let installation_path = Path::new(path).parent().unwrap_or(Path::new("."))
        .join(&cfg.installation);
    let installation: Installation =
        serde_json::from_slice(&fs::read(installation_path).map_err(err)?).map_err(err)?;
    installation.validate()?;
    if cfg.allow_motion && (!installation.calibration_verified || !installation.imu_pose_verified) {
        return Err(err("motion requires verified joint and IMU installation; no automatic calibration"));
    }
    Ok((cfg, installation))
}

pub struct FeetechIo {
    luwu_native: bool,
    filtered_velocity: Option<([f64; NUM_JOINTS], Instant)>,
    saturated_ids: Vec<u8>,
    port: serialport::TTYPort,
    pub joints: Vec<Joint>,
    ids: Vec<u8>,
    imu: std::sync::Arc<Bno08x>,
    limits: Vec<(i32, i32, u8)>,
    allow_motion: bool,
    enabled: bool,
    healthy: Option<Instant>,
    slow: Option<SlowSensors>,
    recovery: FeedbackRecovery,
    torque_poll_interval: Option<Duration>,
    torque_checked_at: Option<Instant>,
    telemetry: serde_json::Value,
    servo_gain_profile: Option<ServoGainProfile>,
    servo_gains_verified: bool,
    scheduled_bus: bool,
    transaction_deadline: Option<Instant>,
    alternate_feedback: bool,
}

#[derive(Default, Debug, Clone, Serialize)]
pub struct FeedbackRecovery {
    pub attempts: u64,
    pub recovered: u64,
    pub last_error: Option<String>,
    pub last_elapsed_ms: f64,
}

/// Cached feedback plus either legacy try-lock writes or a single-owner bus scheduler.
/// Scheduled writes wait for a bounded submission acknowledgement, never an unbounded FIFO.
pub struct ThreadedFeetechIo {
    bus: std::sync::Arc<std::sync::Mutex<FeetechIo>>,
    feedback: std::sync::Arc<(std::sync::Mutex<FeedbackState>, std::sync::Condvar)>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    observation_at: Option<Instant>,
    imu: std::sync::Arc<Bno08x>,
    scheduled_bus: bool,
}

struct PendingTarget {
    target: JointTargets,
    observation_at: Instant,
    queued_at: Instant,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    reply: std::sync::mpsc::SyncSender<Result<()>>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FeedbackStats {
    pub complete_reads: u64,
    pub failed_reads: u64,
    pub read_ms: f64,
    pub last_write_wait_ms: f64,
    pub busy_targets: u64,
    pub written_targets: u64,
    pub write_ms: f64,
    pub sample_age_ms: Option<f64>,
    pub last_error: Option<String>,
    pub recovery: FeedbackRecovery,
    pub scheduled_bus: bool,
    pub queued_targets: u64,
    pub expired_targets: u64,
    pub target_queue_ms: f64,
    pub target_interval_ms: f64,
    pub max_target_interval_ms: f64,
}

#[derive(Default)]
struct FeedbackState {
    sample: Option<(Sensors, SlowSensors, Instant)>,
    last_sample_at: Option<Instant>,
    ready: bool,
    fatal: Option<String>,
    stats: FeedbackStats,
    telemetry: serde_json::Value,
    // A missed write reserves the next tick, but cannot starve reads indefinitely.
    write_turn_until: Option<Instant>,
    pending_target: Option<PendingTarget>,
    last_written_at: Option<Instant>,
    angle_alarm_retries: u8,
}

impl FeedbackState {
    fn latch_motor_fault(&mut self, error: &IoError) {
        if !matches!(error, IoError::ImuUnavailable(_)) {
            self.fatal.get_or_insert_with(|| error.to_string());
        }
    }
}

impl ThreadedFeetechIo {
    pub fn open(path: &str) -> Result<Self> {
        Self::from_io(FeetechIo::open(path)?)
    }

    /// Keep this owner alive across motor initialization failures: reopening it resets SH-2.
    pub fn open_imu(path: &str) -> Result<std::sync::Arc<Bno08x>> {
        let (cfg, installation) = load_config(path)?;
        let imu = Bno08x::open(&cfg.imu_bus, cfg.imu_address, installation.imu_mount_wxyz)?;
        imu.set_luwu_filter(cfg.luwu_native);
        Ok(std::sync::Arc::new(imu))
    }

    pub fn open_with_imu(path: &str, imu: std::sync::Arc<Bno08x>) -> Result<Self> {
        Self::from_io(FeetechIo::open_with_imu(path, imu)?)
    }

    fn from_io(mut io: FeetechIo) -> Result<Self> {
        use std::sync::{
            Arc, Condvar, Mutex,
            atomic::{AtomicBool, Ordering},
        };
        // The same fast 56..70 feedback block as the live Python monitor.
        // Register 65 alarms remain in every frame; torque is verified at 10 Hz.
        io.torque_poll_interval = Some(Duration::from_millis(100));
        let scheduled_bus = io.scheduled_bus;
        let imu = io.imu.clone();
        let bus = Arc::new(Mutex::new(io));
        let feedback = Arc::new((Mutex::new(FeedbackState::default()), Condvar::new()));
        feedback.0.lock().unwrap().stats.scheduled_bus = scheduled_bus;
        let stop = Arc::new(AtomicBool::new(false));
        let worker_bus = bus.clone();
        let shared = feedback.clone();
        let done = stop.clone();
        let worker = std::thread::Builder::new()
            .name("feetech-feedback".into())
            .spawn(move || {
                let startup = Instant::now();
                let mut next_read = startup;
                while !done.load(Ordering::Relaxed) {
                    let mut state = shared.0.lock().unwrap();
                    loop {
                        if done.load(Ordering::Relaxed) { return; }
                        if scheduled_bus && state.pending_target.is_some() { break; }
                        let due = state.write_turn_until.unwrap_or(next_read).max(next_read);
                        let Some(wait) = due.checked_duration_since(Instant::now()) else { break; };
                        state = shared.1.wait_timeout(state, wait).unwrap().0;
                    }
                    drop(state);
                    let begin = Instant::now();
                    let mut io = worker_bus.lock().unwrap();
                    if scheduled_bus {
                        let pending = shared.0.lock().unwrap().pending_target.take();
                        if let Some(pending) = pending {
                            Self::submit_scheduled_target(&mut io, &shared, pending);
                        }
                    }
                    // The writer may have reserved a turn between our wait and lock.
                    if shared.0.lock().unwrap().write_turn_until.is_some_and(|t| t > Instant::now()) {
                        drop(io);
                        continue;
                    }
                    let sampled_at = Instant::now();
                    if scheduled_bus { io.transaction_deadline = Some(sampled_at + SCHEDULED_READ_BUDGET); }
                    let result = io
                        .motor_feedback()
                        .and_then(|s| io.slow_sensors().map(|slow| (s, slow, sampled_at)));
                    io.transaction_deadline = None;
                    let mut state = shared.0.lock().unwrap();
                    state.ready = true;
                    state.stats.read_ms = sampled_at.elapsed().as_secs_f64() * 1000.;
                    state.stats.recovery = io.feedback_recovery().clone();
                    match result {
                        Ok(sample) => {
                            state.angle_alarm_retries = 0;
                            io.healthy = Some(sampled_at);
                            state.sample = Some(sample);
                            state.last_sample_at = Some(sampled_at);
                            state.telemetry = io.telemetry.clone();
                            state.stats.complete_reads += 1;
                        }
                        Err(error) => {
                            state.stats.failed_reads += 1;
                            state.stats.last_error = Some(error.to_string());
                            if scheduled_bus && error.to_string().ends_with("alarm 0x02") {
                                // Spread the existing five retries over bounded bus cycles.
                                state.angle_alarm_retries = state.angle_alarm_retries.saturating_add(1);
                                state.sample = None;
                                io.healthy = None;
                                if state.angle_alarm_retries > 5 { state.latch_motor_fault(&error); }
                            } else if !matches!(error, IoError::IncompleteFeedback(_)) {
                                state.sample = None;
                                state.latch_motor_fault(&error);
                            }
                        }
                    }
                    shared.1.notify_all();
                    drop(state);
                    drop(io);
                    // Do not run catch-up bursts after a slow transaction.
                    let period = if scheduled_bus { SCHEDULED_PERIOD } else { CONTROL_PERIOD };
                    next_read = (begin + period).max(Instant::now() + Duration::from_millis(2));
                }
            })
            .map_err(err)?;
        Ok(Self {
            bus,
            feedback,
            stop,
            worker: Some(worker),
            observation_at: None,
            imu,
            scheduled_bus,
        })
    }

    fn submit_scheduled_target(
        io: &mut FeetechIo,
        shared: &(std::sync::Mutex<FeedbackState>, std::sync::Condvar),
        pending: PendingTarget,
    ) {
        use std::sync::atomic::Ordering;
        let mut state = shared.0.lock().unwrap();
        state.stats.target_queue_ms = pending.queued_at.elapsed().as_secs_f64() * 1000.;
        let result = if pending.cancelled.load(Ordering::Acquire)
            || pending.observation_at.elapsed() >= Duration::from_millis(100)
            || pending.queued_at.elapsed() >= SCHEDULED_WRITE_ACK
        {
            state.stats.expired_targets += 1;
            Err(IoError::StaleTarget)
        } else if let Some(error) = &state.fatal {
            Err(err(error))
        } else if !state.sample.is_some_and(|s| s.2.elapsed() < Duration::from_millis(80)) {
            Err(IoError::IncompleteFeedback("no fresh complete joint snapshot".into()))
        } else {
            drop(state);
            let started = Instant::now();
            let result = io.write(&pending.target);
            state = shared.0.lock().unwrap();
            state.stats.write_ms = started.elapsed().as_secs_f64() * 1000.;
            if result.is_ok() {
                let now = Instant::now();
                if let Some(last) = state.last_written_at {
                    state.stats.target_interval_ms = now.duration_since(last).as_secs_f64() * 1000.;
                    state.stats.max_target_interval_ms = state.stats.max_target_interval_ms.max(state.stats.target_interval_ms);
                }
                state.last_written_at = Some(now);
                state.stats.written_targets += 1;
            }
            result
        };
        // An acknowledgement means an actual driver submission, never mailbox acceptance.
        let _ = pending.reply.send(result);
    }

    fn write_scheduled(&mut self, target: &JointTargets) -> Result<()> {
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}, mpsc};
        let observation_at = self.observation_at.ok_or_else(|| err("no control observation"))?;
        if observation_at.elapsed() >= Duration::from_millis(100) { return Err(IoError::StaleTarget); }
        let queued_at = Instant::now();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (reply, receiver) = mpsc::sync_channel(1);
        {
            let mut state = self.feedback.0.lock().map_err(err)?;
            if let Some(error) = &state.fatal { return Err(err(error)); }
            if state.pending_target.is_some() { return Err(IoError::BusBusy); }
            state.pending_target = Some(PendingTarget {
                target: *target, observation_at, queued_at, cancelled: cancelled.clone(), reply,
            });
            state.stats.queued_targets += 1;
        }
        self.feedback.1.notify_all();
        match receiver.recv_timeout(SCHEDULED_WRITE_ACK) {
            Ok(result) => result,
            Err(_) => {
                cancelled.store(true, Ordering::Release);
                let mut state = self.feedback.0.lock().map_err(err)?;
                state.pending_target = None;
                // A write might already have started. Do not call an uncertain outcome BusBusy.
                let error = err("scheduled write acknowledgement timeout; explicit reset required");
                state.latch_motor_fault(&error);
                Err(error)
            }
        }
    }

    #[cfg(test)]
    fn snapshot(&self, deadline: Instant) -> Result<(Sensors, SlowSensors, Instant)> {
        let (lock, changed) = &*self.feedback;
        let mut state = lock.lock().map_err(err)?;
        loop {
            if let Some(error) = &state.fatal {
                return Err(err(error));
            }
            if let Some(sample) = state.sample {
                if sample.2.elapsed() < Duration::from_millis(80) {
                    return Ok(sample);
                }
            }
            let Some(remaining) = deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero())
            else {
                return Err(err(format!(
                    "feedback unavailable/stale: {:?}",
                    state.stats.last_error
                )));
            };
            state = changed.wait_timeout(state, remaining).map_err(err)?.0;
        }
    }

    pub fn feedback_stats(&self) -> FeedbackStats {
        let state = self.feedback.0.lock().unwrap();
        let mut stats = state.stats.clone();
        stats.sample_age_ms = state.sample.map(|s| s.2.elapsed().as_secs_f64() * 1000.);
        stats
    }

    pub fn feedback_recovery(&self) -> FeedbackRecovery {
        self.feedback_stats().recovery
    }

    pub fn torque_enabled_ids(&mut self) -> Result<Vec<u8>> {
        self.bus.lock().map_err(err)?.torque_enabled_ids()
    }
}

impl RobotIo for ThreadedFeetechIo {
    fn read(&mut self) -> Result<Sensors> {
        self.observation_at = None;
        let state = self.feedback.0.lock().map_err(err)?;
        if let Some(error) = &state.fatal { return Err(err(error)); }
        let (mut sensor, _, at) = state.sample
            .filter(|s| s.2.elapsed() < Duration::from_millis(80))
            .ok_or_else(|| IoError::IncompleteFeedback("no fresh complete joint snapshot".into()))?;
        drop(state);
        sensor.imu = self.imu.read()?;
        self.observation_at = Some(at);
        Ok(sensor)
    }

    fn feedback(&self) -> Option<serde_json::Value> {
        let state = self.feedback.0.lock().ok()?;
        let at = state.sample.map(|sample| sample.2).or(state.last_sample_at)?;
        let mut value = state.telemetry.clone();
        value["error"] = serde_json::json!(state.fatal);
        value["transport"] = serde_json::json!(state.stats);
        value["sequence"] = serde_json::json!(state.stats.complete_reads);
        let joint_valid = state.sample.is_some() && at.elapsed() < Duration::from_millis(80) && state.fatal.is_none();
        drop(state);
        value["joint_age_s"] = serde_json::json!(at.elapsed().as_secs_f64());
        value["imu_transport"] = self.imu.report_stats().ok()?;
        let imu = self.imu.snapshot();
        value["imu_valid"] = serde_json::json!(imu.is_ok());
        value["control_valid"] = serde_json::json!(self.observation_at.is_some() && joint_valid && imu.is_ok());
        match imu {
            Ok((sample, at)) => {
                value["imu"] = serde_json::json!({"gyro":sample.gyro,"quat":sample.quat,"gravity":sample.gravity});
                value["imu_age_s"] = serde_json::json!(at.elapsed().as_secs_f64());
            }
            Err(error) => {
                if value["error"].is_null() { value["error"] = serde_json::json!(error.to_string()); }
                value["motion_available"] = serde_json::json!(false);
            }
        }
        Some(value)
    }

    fn manages_feedback_age(&self) -> bool { true }

    fn clear_fault(&mut self) -> Result<()> {
        let mut io = self.bus.lock().map_err(err)?;
        let mut state = self.feedback.0.lock().map_err(err)?;
        let at = Instant::now();
        match io.relax_and_revalidate() {
            Ok((sensors, slow)) => {
                state.sample = Some((sensors, slow, at));
                state.last_sample_at = Some(at);
                state.telemetry = io.telemetry.clone();
                state.fatal = None;
                state.stats.last_error = None;
                state.stats.complete_reads += 1;
                state.ready = true;
                state.write_turn_until = None;
                state.pending_target = None;
                state.angle_alarm_retries = 0;
                state.last_written_at = None;
                self.observation_at = None;
                self.feedback.1.notify_all();
                Ok(())
            }
            Err(error) => {
                state.sample = None;
                state.latch_motor_fault(&error);
                Err(error)
            }
        }
    }

    fn write(&mut self, target: &JointTargets) -> Result<()> {
        if self.scheduled_bus { return self.write_scheduled(target); }
        let deadline = self
            .observation_at
            .ok_or_else(|| err("no control observation"))?
            + Duration::from_millis(100);
        if Instant::now() >= deadline { return Err(IoError::StaleTarget); }
        let waited = Instant::now();
        let mut io = match self.bus.try_lock() {
            Ok(io) => io,
            Err(std::sync::TryLockError::Poisoned(error)) => return Err(err(error)),
            Err(std::sync::TryLockError::WouldBlock) => {
                let mut state = self.feedback.0.lock().map_err(err)?;
                if let Some(error) = &state.fatal { return Err(err(error)); }
                state.stats.busy_targets += 1;
                state.stats.last_write_wait_ms = waited.elapsed().as_secs_f64() * 1000.;
                state.write_turn_until = Some(Instant::now() + WRITE_TURN);
                self.feedback.1.notify_all();
                return Err(IoError::BusBusy);
            }
        };
        let mut state = self.feedback.0.lock().map_err(err)?;
        state.stats.last_write_wait_ms = waited.elapsed().as_secs_f64() * 1000.;
        if let Some(error) = &state.fatal { return Err(err(error)); }
        if Instant::now() >= deadline { return Err(IoError::StaleTarget); }
        if !state.sample.is_some_and(|s| s.2.elapsed() < Duration::from_millis(80)) {
            return Err(IoError::IncompleteFeedback("no fresh complete joint snapshot".into()));
        }
        drop(state);
        // Success means bytes submitted to the serial driver, not just queued here.
        let started = Instant::now();
        let result = io.write(target);
        let mut state = self.feedback.0.lock().map_err(err)?;
        state.stats.write_ms = started.elapsed().as_secs_f64() * 1000.;
        if result.is_ok() { state.stats.written_targets += 1; }
        state.write_turn_until = None;
        self.feedback.1.notify_all();
        result
    }

    fn set_gain(&mut self, kp: u16) -> Result<()> {
        fixed_gain(kp)
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        let mut io = self.bus.lock().map_err(err)?;
        let mut state = self.feedback.0.lock().map_err(err)?;
        if on {
            if let Some(error) = &state.fatal {
                return Err(err(error));
            }
        }
        state.sample = None;
        self.observation_at = None;
        drop(state);
        let result = io.set_torque(on);
        if let Err(error) = &result {
            self.feedback.0.lock().map_err(err)?.latch_motor_fault(error);
        }
        result
    }

    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        let state = self.feedback.0.lock().map_err(err)?;
        if let Some(error) = &state.fatal { return Err(err(error)); }
        state.sample.filter(|s| s.2.elapsed() < Duration::from_millis(80))
            .map(|s| s.1)
            .ok_or_else(|| IoError::IncompleteFeedback("no fresh slow-sensor snapshot".into()))
    }

    fn imu_ready(&self) -> bool {
        self.imu.read().is_ok() && self.feedback.0.lock().unwrap().sample.is_some()
    }
}

impl Drop for ThreadedFeetechIo {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.feedback.1.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        // Dropping a reader never changes torque or replays a pending goal.
    }
}

fn fixed_gain(kp: u16) -> Result<()> {
    if kp != 200 {
        return Err(err("unsupported XL330 gain conversion; fixed PD token is 200"));
    }
    Ok(())
}

fn open_serial(path: &str) -> Result<serialport::TTYPort> {
    // serialport owns both flock and TIOCEXCL; a second fd would lock us out.
    let port = serialport::new(path, 1_000_000)
        .timeout(Duration::from_millis(30))
        .open_native()
        .map_err(err)?;
    nonblocking(&port)?;
    Ok(port)
}

fn nonblocking(port: &serialport::TTYPort) -> Result<()> {
    // TTYPort polls before write(), but opens a blocking fd. POLLOUT only
    // promises SOME space, not enough for the whole packet. Retain O_NONBLOCK
    // so a partial write cannot block past the userspace deadline.
    let fd = port.as_raw_fd();
    // SAFETY: fd is owned and live; both fcntl commands take integer arguments.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags == -1 { return Err(err(std::io::Error::last_os_error())); }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(err(std::io::Error::last_os_error()));
    }
    Ok(())
}

fn signed(raw: [u8; 2]) -> i32 {
    let v = u16::from_le_bytes(raw);
    i32::from(v & 0x7fff) * if v & 0x8000 != 0 { -1 } else { 1 }
}
fn checksum(bytes: &[u8]) -> u8 {
    !bytes.iter().fold(0u8, |sum, b| sum.wrapping_add(*b))
}
fn instruction(id: u8, op: u8, params: &[u8]) -> Result<Vec<u8>> {
    if params.len() > 253 {
        return Err(err("oversized instruction"));
    }
    let mut p = vec![255, 255, id, (params.len() + 2) as u8, op];
    p.extend_from_slice(params);
    p.push(checksum(&p[2..]));
    Ok(p)
}
fn decode_packet(packet: &[u8], id: u8, size: usize) -> Result<Vec<u8>> {
    if packet.len() != size + 6
        || packet[..2] != [255, 255]
        || packet[2] != id
        || packet[3] as usize != size + 2
        || checksum(&packet[2..packet.len() - 1]) != packet[packet.len() - 1]
    {
        return Err(IoError::IncompleteFeedback(format!(
            "ID {id}: malformed/checksum response"
        )));
    }
    if packet[4] != 0 {
        return Err(err(format!("ID {id}: alarm 0x{:02x}", packet[4])));
    }
    Ok(packet[5..packet.len() - 1].to_vec())
}

impl FeetechIo {
    fn relax_and_revalidate(&mut self) -> Result<(Sensors, SlowSensors)> {
        // Explicit operator reset: normalize partial enable before revalidation.
        // set_torque(false) verifies register 40 for every ID and never writes goals.
        self.set_torque(false)?;
        self.imu.reconnect()?;
        let sample = self.read()?;
        Ok((sample, self.slow_sensors()?))
    }

    pub fn open(config_path: &str) -> Result<Self> {
        let imu = ThreadedFeetechIo::open_imu(config_path)?;
        Self::open_with_imu(config_path, imu)
    }

    fn open_with_imu(config_path: &str, imu: std::sync::Arc<Bno08x>) -> Result<Self> {
        let (cfg, installation) = load_config(config_path)?;
        let native = open_serial(&cfg.port)?;
        let ids = installation.joints.iter().map(|j| j.id).collect();
        let mut io = Self {
            luwu_native: cfg.luwu_native,
            filtered_velocity: None,
            saturated_ids: Vec::new(),
            port: native,
            joints: installation.joints,
            ids,
            imu,
            limits: vec![],
            allow_motion: cfg.allow_motion,
            enabled: false,
            healthy: None,
            slow: None,
            recovery: FeedbackRecovery::default(),
            torque_poll_interval: None,
            telemetry: serde_json::Value::Null,
            torque_checked_at: None,
            servo_gain_profile: cfg.servo_gain_profile,
            servo_gains_verified: false,
            scheduled_bus: cfg.scheduled_bus,
            transaction_deadline: None,
            alternate_feedback: false,
        };
        let config = io.sync_read(0, 40)?;
        for (id, row) in io.ids.iter().zip(config) {
            if row[0] != 3
                || !(40..=59).contains(&row[1])
                || row[2..5] != [0, 10, 31]
                || row[5] != *id
                || row[8] == 0
                || row[30] != 1
                || row[33] != 4
            {
                return Err(err(format!(
                    "ID {id}: unverified HD1910M mode-4 firmware/registers"
                )));
            }
            let low = u16::from_le_bytes([row[9], row[10]]) as i32;
            let high = u16::from_le_bytes([row[11], row[12]]) as i32;
            if low >= high || high > 4095 {
                return Err(err("invalid firmware travel limits"));
            }
            io.limits.push((low, high, row[13]));
        }
        // Observe adopted enable state, never write a goal during startup.
        io.enabled = io.sync_read(40, 1)?.iter().all(|r| r[0] == 1);
        io.torque_checked_at = Some(Instant::now());
        if let Some(profile) = io.servo_gain_profile {
            io.servo_gains_verified = io.sync_read(50, 2)? == profile.rows(&io.joints);
        }
        Ok(io)
    }

    fn configure_gains(&mut self, all_off: bool) -> Result<()> {
        let Some(profile) = self.servo_gain_profile else { return Ok(()); };
        self.servo_gains_verified = false;
        let expected = profile.rows(&self.joints);
        let actual = self.sync_read(50, 2)?;
        if actual != expected {
            if !all_off {
                return Err(err("temporary P/D mismatch while enabled; not changing gains under load"));
            }
            // HD1910 volatile P/D registers; no EEPROM unlock or persistent write.
            self.sync_write(50, &expected)?;
            if self.sync_read(50, 2)? != expected {
                return Err(err("temporary P/D readback mismatch; not enabling"));
            }
        }
        self.servo_gains_verified = true;
        Ok(())
    }

    fn send(&mut self, id: u8, op: u8, params: &[u8]) -> Result<()> {
        self.port
            .set_timeout(Duration::from_millis(30))
            .map_err(err)?;
        self.port
            .clear(serialport::ClearBuffer::Input)
            .map_err(err)?;
        let packet = instruction(id, op, params)?;
        if (op == 0x83 && params.first() == Some(&42)) || self.transaction_deadline.is_some() {
            // One deadline for the whole packet, including partial writes/EINTR.
            let deadline = Instant::now() + GOAL_WRITE_BUDGET;
            let mut remaining = packet.as_slice();
            while !remaining.is_empty() {
                let timeout = deadline.checked_duration_since(Instant::now())
                    .filter(|d| !d.is_zero()).ok_or_else(|| err("goal write deadline exceeded"))?;
                self.port.set_timeout(timeout).map_err(err)?;
                match self.port.write(remaining) {
                    Ok(0) => return Err(err("zero-length goal write")),
                    Ok(n) => remaining = &remaining[n..],
                    Err(e) if matches!(e.kind(), std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock) => continue,
                    Err(e) => return Err(err(e)),
                }
            }
            Ok(())
        } else {
            self.port.write_all(&packet).map_err(err)
        }
    }

    fn sync_read_once(&mut self, address: u8, size: u8) -> Result<Vec<Vec<u8>>> {
        let mut params = vec![address, size];
        params.extend(&self.ids);
        self.send(254, 0x82, &params)?;
        let started = Instant::now();
        let deadline = self.transaction_deadline.unwrap_or(started + Duration::from_millis(30));
        let mut rows = vec![None; self.ids.len()];
        let mut packet_error: Option<IoError> = None;
        let mut bytes = Vec::new();
        let mut received = 0;
        let mut discarded = 0;
        let packet_len = size as usize + 6;
        while rows.iter().any(Option::is_none) {
            // Recover framing after noise, truncated packets and out-of-order IDs.
            while bytes.len() >= 4 {
                let id = bytes[2];
                if bytes[..2] != [255, 255]
                    || bytes[3] as usize != size as usize + 2
                    || !self.ids.contains(&id)
                {
                    bytes.remove(0);
                    discarded += 1;
                    continue;
                }
                if bytes.len() < packet_len {
                    break;
                }
                let result = decode_packet(&bytes[..packet_len], id, size as usize);
                if matches!(result, Err(IoError::IncompleteFeedback(_))) {
                    bytes.remove(0);
                    discarded += 1;
                    continue;
                }
                let row = bytes[5..packet_len - 1].to_vec();
                let mut alarm = result.err();
                if address <= 65
                    && address as usize + size as usize > 65
                    && row[65 - address as usize] != 0
                {
                    let status = row[65 - address as usize];
                    if alarm.is_none() || status != 2 {
                        alarm = Some(err(format!("ID {id}: alarm 0x{status:02x}")));
                    }
                }
                if let Some(error) = alarm {
                    // A retryable angle alarm must not hide another servo's hard alarm.
                    if packet_error
                        .as_ref()
                        .is_none_or(|e| e.to_string().ends_with("alarm 0x02"))
                    {
                        packet_error = Some(error);
                    }
                }
                let index = self.ids.iter().position(|v| *v == id).unwrap();
                rows[index] = Some(row);
                bytes.drain(..packet_len);
            }
            if rows.iter().all(Option::is_some) {
                break;
            }
            let missing: Vec<u8> = self
                .ids
                .iter()
                .zip(&rows)
                .filter_map(|(id, row)| row.is_none().then_some(*id))
                .collect();
            let timeout = || {
                IoError::IncompleteFeedback(format!(
                    "sync-read addr={address} size={size} missing_ids={missing:?} received_bytes={received} discarded_bytes={discarded} tail={bytes:02x?} elapsed_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1000.
                ))
            };
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero());
            let Some(remaining) = remaining else {
                return Err(packet_error.unwrap_or_else(timeout));
            };
            self.port.set_timeout(remaining).map_err(err)?;
            let mut chunk = [0u8; 512];
            let n = match self.port.read(&mut chunk) {
                Ok(0) => return Err(packet_error.unwrap_or_else(|| err("serial EOF"))),
                Ok(n) => n,
                Err(error) => {
                    return Err(packet_error.unwrap_or_else(|| match error.kind() {
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => timeout(),
                        _ => err(error),
                    }));
                }
            };
            received += n;
            bytes.extend_from_slice(&chunk[..n]);
        }
        if let Some(error) = packet_error {
            return Err(error);
        }
        Ok(rows.into_iter().map(Option::unwrap).collect())
    }

    fn sync_read(&mut self, address: u8, size: u8) -> Result<Vec<Vec<u8>>> {
        if self.transaction_deadline.is_some() { return self.sync_read_once(address, size); }
        // Retry only the confirmed angle-sensor alarm, never replay motor writes.
        for attempt in 0..=5 {
            match self.sync_read_once(address, size) {
                Err(e) if attempt < 5 && e.to_string().ends_with("alarm 0x02") => continue,
                result => return result,
            }
        }
        unreachable!()
    }

    fn sync_write(&mut self, address: u8, rows: &[Vec<u8>]) -> Result<()> {
        if rows.len() != self.ids.len()
            || rows.is_empty()
            || rows[0].is_empty()
            || rows.iter().any(|v| v.len() != rows[0].len())
        {
            return Err(err("invalid write shape"));
        }
        let mut params = vec![address, rows[0].len() as u8];
        for (id, row) in self.ids.iter().zip(rows) {
            params.push(*id);
            params.extend(row);
        }
        self.send(254, 0x83, &params)
    }

    fn motor_feedback(&mut self) -> Result<Sensors> {
        // Enabled motion reads torque and feedback in one response per servo.
        // This avoids a second short status burst immediately after feedback.
        let check_enable = self.allow_motion && self.enabled;
        let separate_torque = check_enable && self.torque_poll_interval.is_some();
        let scheduled = self.transaction_deadline.is_some();
        let torque_due = check_enable && self.torque_checked_at.is_none_or(|t|
            t.elapsed() >= self.torque_poll_interval.unwrap_or(Duration::ZERO));
        if separate_torque && !scheduled
            && self
                .torque_checked_at
                .is_none_or(|t| t.elapsed() >= self.torque_poll_interval.unwrap())
        {
            if self.sync_read(40, 1)?.iter().any(|r| r[0] != 1) {
                self.enabled = false;
                return Err(err("enable feedback lost; explicit re-init required"));
            }
            self.torque_checked_at = Some(Instant::now());
        }
        let combined = check_enable && (!separate_torque || (scheduled && torque_due));
        let (address, size, mut offset) = if combined {
            (40, 31, 16)
        } else if scheduled && self.alternate_feedback {
            (55, 16, 1)
        } else {
            (56, 15, 0)
        };
        if scheduled { self.alternate_feedback = !self.alternate_feedback; }
        let mut enable_offset = 0;
        let rows = match self.sync_read(address, size) {
            Err(error @ IoError::IncompleteFeedback(_)) if check_enable && !scheduled => {
                self.recovery.attempts += 1;
                self.recovery.last_error = Some(error.to_string());
                let retry_start = Instant::now();
                // No writes or stale observations during recovery. Read an extra
                // preceding register so late original packets fail length checks.
                std::thread::sleep(Duration::from_millis(2));
                let result = self.sync_read_once(address - 1, size + 1);
                self.recovery.last_elapsed_ms = retry_start.elapsed().as_secs_f64() * 1000.;
                offset += 1;
                enable_offset = 1;
                result?
            }
            result => result?,
        };
        if combined && rows.iter().any(|row| row[enable_offset] != 1) {
            self.enabled = false;
            return Err(err("enable feedback lost; explicit re-init required"));
        }
        if combined { self.torque_checked_at = Some(Instant::now()); }
        let mut out = Sensors::default();
        let mut slow = SlowSensors {
            volts: 0.,
            temps_c: [0.; NUM_JOINTS],
        };
        let mut states = Vec::with_capacity(NUM_JOINTS);
        for (i, (row, j)) in rows.iter().zip(&self.joints).enumerate() {
            let row = &row[offset..offset + 15];
            let ticks = signed([row[0], row[1]]);
            let (low, high, max_temp) = self.limits[i];
            if ticks < low || ticks > high || row[7] >= max_temp {
                return Err(err(format!("ID {}: invalid position/temperature", j.id)));
            }
            out.positions[i] = (ticks - j.zero_ticks) as f64 * RAD_TICK * j.direction as f64;
            out.velocities[i] = signed([row[2], row[3]]) as f64 * 0.732 * std::f64::consts::TAU
                / 60.
                * j.direction as f64;
            out.currents_ma[i] = signed([row[13], row[14]]).abs() as f64 * 6.5;
            slow.volts += row[6] as f64 / 10. / NUM_JOINTS as f64;
            slow.temps_c[i] = row[7] as f64;
            states.push(serde_json::json!({"servo_id": j.id, "valid":true,
                "position_ticks":ticks,"velocity_rad_s":out.velocities[i],
                "voltage_v":row[6] as f64/10.,"temperature_c":row[7],"status":row[9],
                "current_a":signed([row[13],row[14]]) as f64*0.0065,
                "torque_enabled":if self.enabled { Some(1) } else { None::<u8> }}));
        }
        if self.luwu_native {
            let now = Instant::now();
            if let Some((old, at)) = self.filtered_velocity {
                let weight = 0.4f64.powf(now.duration_since(at).as_secs_f64() / 0.01);
                for i in 0..NUM_JOINTS { out.velocities[i] = weight * old[i] + (1.0-weight) * out.velocities[i]; }
            }
            self.filtered_velocity = Some((out.velocities, now));
        }
        self.telemetry = serde_json::json!({"positions":out.positions,"velocities":out.velocities,
            "luwu_native_io":self.luwu_native,"saturated_ids":self.saturated_ids,
            "states":states,"joints":self.joints,"backend":"feetech_native",
            "motion_available":self.allow_motion,
            "servo_gain_profile":self.servo_gain_profile,"servo_gains_verified":self.servo_gains_verified});
        self.slow = Some(slow);
        if enable_offset == 1 {
            self.recovery.recovered += 1;
        }
        Ok(out)
    }

    pub fn feedback_recovery(&self) -> &FeedbackRecovery {
        &self.recovery
    }

    /// Passive readback for an explicit supported-test HOLD handoff.
    pub fn torque_enabled_ids(&mut self) -> Result<Vec<u8>> {
        let rows = self.sync_read(40, 1)?;
        Ok(self
            .ids
            .iter()
            .zip(rows)
            .filter_map(|(id, row)| if row[0] == 1 { Some(*id) } else { None })
            .collect())
    }
}

impl RobotIo for FeetechIo {
    fn read(&mut self) -> Result<Sensors> {
        let started = Instant::now();
        let mut out = match self.motor_feedback() {
            Ok(out) => out,
            Err(error) => {
                if !matches!(error, IoError::IncompleteFeedback(_)) { self.healthy = None; }
                return Err(error);
            }
        };
        out.imu = self.imu.read()?;
        if started.elapsed() > Duration::from_millis(80) {
            return Err(err("feedback recovery exceeded 80 ms budget"));
        }
        self.healthy = Some(started);
        Ok(out)
    }
    fn write(&mut self, target: &JointTargets) -> Result<()> {
        if !self.allow_motion
            || !self.enabled
            || self
                .healthy
                .is_none_or(|t| t.elapsed() > Duration::from_millis(100))
        {
            return Err(err("motion disabled/not enabled/stale feedback"));
        }
        if self.servo_gain_profile.is_some() && !self.servo_gains_verified {
            return Err(err("temporary P/D not verified; explicit init required"));
        }
        self.imu.read()?;
        // Validate the complete vector before putting ANY position on the bus.
        let mut rows = Vec::new();
        self.saturated_ids.clear();
        for ((q, j), (low, high, _)) in target.positions.iter().zip(&self.joints).zip(&self.limits)
        {
            let mut ticks = j.zero_ticks as f64 + *q / RAD_TICK * j.direction as f64;
            if !ticks.is_finite() { return Err(err("nonfinite target")); }
            if self.luwu_native && (ticks < *low as f64 || ticks > *high as f64) {
                ticks = ticks.clamp(*low as f64, *high as f64);
                self.saturated_ids.push(j.id);
            }
            if ticks < *low as f64 || ticks > *high as f64 {
                return Err(err("target outside calibrated firmware travel"));
            }
            rows.push((ticks.round() as u16).to_le_bytes().to_vec());
        }
        // ONLY goal position 42/43; 44/45 is current in HD1910, not XL330 time.
        self.sync_write(42, &rows)
    }
    fn set_gain(&mut self, kp: u16) -> Result<()> {
        // Native mode uses the already-configured physical PD, not XL330 gain units.
        fixed_gain(kp)
    }
    fn set_torque(&mut self, on: bool) -> Result<()> {
        if !self.allow_motion {
            return Err(err("read-only configuration"));
        }
        if on {
            self.healthy = None;
            let _fresh = self.motor_feedback()?;
            self.imu.read()?;
            let controls = self.sync_read(40, 4)?;
            let raw = self.sync_read(56, 2)?;
            let all_off = controls.iter().all(|r| r[0] == 0);
            let all_on = controls.iter().all(|r| r[0] == 1);
            if !all_off && !all_on {
                self.enabled = false;
                return Err(err(
                    "partial enable; support robot and explicitly relax before init",
                ));
            }
            self.configure_gains(all_off)?;
            if all_off {
                // Only an explicit enable may preload a limp robot. No startup
                // write, and never overwrite targets of already enabled joints.
                for (q, (low, high, _)) in raw.iter().zip(&self.limits) {
                    let ticks = signed([q[0], q[1]]);
                    if ticks < *low || ticks > *high {
                        return Err(err("invalid preload position"));
                    }
                }
                self.sync_write(42, &raw)?;
                if self.sync_read(42, 2)? != raw {
                    return Err(err("preload readback mismatch; not enabling"));
                }
            }
        }
        self.enabled = false;
        self.sync_write(40, &vec![vec![u8::from(on)]; NUM_JOINTS])?;
        if self.sync_read(40, 1)?.iter().any(|r| r[0] != u8::from(on)) {
            return Err(err("torque readback mismatch"));
        }
        self.enabled = on;
        self.torque_checked_at = Some(Instant::now());
        Ok(())
    }
    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        self.slow.ok_or_else(|| err("no motor sample"))
    }
    fn imu_ready(&self) -> bool {
        self.imu.read().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduled_peer(mut peer: serialport::TTYPort,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        reads: std::sync::mpsc::Sender<()>,
    ) -> std::thread::JoinHandle<Vec<Vec<u8>>> {
        std::thread::spawn(move || {
            let mut goals = vec![];
            peer.set_timeout(Duration::from_millis(40)).unwrap();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let mut header = [0; 4];
                if peer.read_exact(&mut header).is_err() { break; }
                assert_eq!(&header[..2], &[255, 255]);
                let mut body = vec![0; header[3] as usize];
                peer.read_exact(&mut body).unwrap();
                if body[0] == 0x83 {
                    assert_eq!(&body[1..3], &[42, 2], "only goal-position writes allowed");
                    goals.push(body);
                    continue;
                }
                assert_eq!(body[0], 0x82);
                let address = body[1] as usize;
                let size = body[2] as usize;
                let _ = reads.send(());
                std::thread::sleep(Duration::from_millis(6));
                let mut registers = [0; 71];
                registers[40] = 1;
                registers[57] = 8;
                registers[62] = 74;
                registers[63] = 30;
                for id in &body[3..body.len()-1] {
                    let row = &registers[address..address+size];
                    let mut packet = vec![255,255,*id,(size+2) as u8,0];
                    packet.extend(row);
                    packet.push(checksum(&packet[2..]));
                    if peer.write_all(&packet).is_err() { return goals; }
                }
            }
            goals
        })
    }

    #[test]
    fn scheduled_writer_submits_during_reads_and_preserves_local_mapping() {
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}, mpsc};
        let (mut bus, peer) = fixture();
        bus.allow_motion = true;
        bus.scheduled_bus = true;
        let joints = bus.joints.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let (sender, reads) = mpsc::channel();
        let server = scheduled_peer(peer, stop.clone(), sender);
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now()+Duration::from_millis(100)).unwrap();
        let _ = reads.try_iter().count();
        for step in 1..=4 {
            reads.recv_timeout(Duration::from_millis(100)).unwrap();
            io.read().unwrap();
            let started = Instant::now();
            io.write(&JointTargets::new([0.01 * step as f64;15])).unwrap();
            assert!(started.elapsed() < SCHEDULED_WRITE_ACK);
        }
        let stats = io.feedback_stats();
        assert_eq!(stats.written_targets, 4);
        assert_eq!(stats.busy_targets, 0);
        let _keep_bus = io.bus.clone();
        drop(io);
        stop.store(true,Ordering::Relaxed);
        let goals = server.join().unwrap();
        assert_eq!(goals.len(),4);
        for (step, packet) in goals.iter().enumerate() {
            for (j, row) in joints.iter().zip(packet[3..packet.len()-1].chunks_exact(3)) {
                assert_eq!(row[0],j.id);
                let ticks = j.zero_ticks as f64 + (step+1) as f64 * 0.01 / RAD_TICK * j.direction as f64;
                assert_eq!(u16::from_le_bytes([row[1],row[2]]),ticks.round() as u16);
            }
        }
    }

    #[test]
    fn scheduled_ack_timeout_cancels_queued_target_without_late_write() {
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}, mpsc};
        let (mut bus, peer) = fixture();
        bus.allow_motion = true;
        bus.scheduled_bus = true;
        let stop = Arc::new(AtomicBool::new(false));
        let (sender, _) = mpsc::channel();
        let server = scheduled_peer(peer, stop.clone(), sender);
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now()+Duration::from_millis(100)).unwrap();
        io.read().unwrap();
        let bus = io.bus.clone();
        let guard = bus.lock().unwrap();
        let result = io.write(&JointTargets::new([0.1;15]));
        assert!(result.unwrap_err().to_string().contains("acknowledgement timeout"));
        assert!(io.feedback.0.lock().unwrap().pending_target.is_none());
        drop(guard);
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(io.feedback_stats().written_targets,0);
        drop(io);
        stop.store(true,Ordering::Relaxed);
        assert!(server.join().unwrap().is_empty());
    }

    #[test]
    fn scheduled_missing_feedback_uses_one_bounded_attempt() {
        let (mut bus, _peer) = fixture();
        bus.allow_motion = true;
        let started = Instant::now();
        bus.transaction_deadline = Some(started+SCHEDULED_READ_BUDGET);
        assert!(matches!(bus.motor_feedback(), Err(IoError::IncompleteFeedback(_))));
        assert!(started.elapsed() < Duration::from_millis(25));
        assert_eq!(bus.recovery.attempts,0);
    }
    #[test]
    fn motor_initialization_failure_keeps_the_same_imu_owner() {
        let imu = std::sync::Arc::new(Bno08x::fixture());
        let dir = std::env::temp_dir().join(format!("feetech-imu-owner-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("installation.json"), include_str!("../../../radxa/installation.json")).unwrap();
        fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({
            "port":"/missing-servo-test","installation":"installation.json","imu_bus":"/must-not-be-reopened",
            "imu_address":75,"allow_motion":false
        })).unwrap()).unwrap();
        for _ in 0..3 {
            let result = ThreadedFeetechIo::open_with_imu(dir.join("config.json").to_str().unwrap(), imu.clone());
            let error = result.err().unwrap().to_string();
            assert!(error.contains("Feetech:"), "{error}");
            assert!(!error.contains("BNO085"), "must not reopen the missing IMU bus: {error}");
            assert_eq!(std::sync::Arc::strong_count(&imu), 1);
            assert_eq!(imu.report_stats().unwrap()["reconnects"], 0);
        }
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn missing_imu_does_not_block_joint_feedback_or_allow_control() {
        use serialport::SerialPort;
        let (mut bus, mut peer) = fixture();
        bus.imu = std::sync::Arc::new(crate::bno08x::Bno08x::fixture_unavailable());
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            respond(&mut peer, &ids, 0);
            peer
        });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        let deadline = Instant::now() + Duration::from_millis(200);
        while io.feedback_stats().complete_reads < 2 {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(matches!(io.read(), Err(IoError::ImuUnavailable(_))));
        let feedback = io.feedback().unwrap();
        assert_eq!(feedback["positions"].as_array().unwrap().len(), 15);
        assert_eq!(feedback["imu_valid"], false);
        assert_eq!(feedback["control_valid"], false);
        assert!(io.feedback.0.lock().unwrap().fatal.is_none());
        assert!(io.write(&JointTargets::new([0.;15])).is_err());
        let _keep_bus = io.bus.clone();
        drop(io);
        let peer = server.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }
    #[test]
    fn imu_outage_cannot_create_or_clear_a_motor_alarm() {
        let mut state = FeedbackState::default();
        state.latch_motor_fault(&IoError::ImuUnavailable("reconnecting".into()));
        assert!(state.fatal.is_none());
        state.latch_motor_fault(&IoError::Bus("ID 1: alarm 0x04".into()));
        let fault = state.fatal.clone();
        state.latch_motor_fault(&IoError::ImuUnavailable("reconnecting".into()));
        assert_eq!(state.fatal, fault);
    }
    #[test]
    fn serial_open_does_not_conflict_with_itself_and_stays_exclusive() {
        use serialport::SerialPort;
        let (_master, slave) = serialport::TTYPort::pair().unwrap();
        let path = slave.name().unwrap();
        drop(slave);
        let port = open_serial(&path).unwrap();
        assert!(open_serial(&path).is_err());
        drop(port);
        assert!(open_serial(&path).is_ok());
    }

    #[test]
    fn explicit_reset_verifies_all_torque_off_before_clearing_and_never_writes_goals() {
        use std::sync::{Arc, Mutex, Condvar, atomic::AtomicBool};
        use serialport::SerialPort;
        for (alarm, still_enabled) in [(0, false), (0x80, false), (0, true)] {
            let (mut bus, mut peer) = fixture();
            bus.allow_motion = true;
            let ids = bus.ids.clone();
            let imu = bus.imu.clone();
            let server = std::thread::spawn(move || {
                expect_write(&mut peer, &ids, 40, &[0]);
                let mut torque = vec![vec![0]; 15];
                if still_enabled { torque[3][0] = 1; }
                reply_rows(&mut peer, &ids, 40, &torque);
                if !still_enabled { respond(&mut peer, &ids, alarm); }
                peer
            });
            let mut io = ThreadedFeetechIo {
                bus:Arc::new(Mutex::new(bus)),
                feedback:Arc::new((Mutex::new(FeedbackState {
                    fatal:Some("old alarm".into()), ..FeedbackState::default()
                }),Condvar::new())),
                stop:Arc::new(AtomicBool::new(true)),worker:None,
                observation_at:None,imu,
                scheduled_bus:false,
            };
            let result=io.clear_fault();
            let peer=server.join().unwrap();
            assert_eq!(peer.bytes_to_read().unwrap(),0,"reset must not write goals or re-enable");
            assert_eq!(result.is_ok(),alarm==0 && !still_enabled);
            if alarm==0 && !still_enabled {
                assert!(io.read().is_ok());
                assert!(io.feedback().unwrap()["error"].is_null());
            } else {
                let expected = if still_enabled { "torque readback mismatch" } else { "alarm 0x80" };
                assert!(result.unwrap_err().to_string().contains(expected));
                assert!(io.read().is_err());
                assert!(io.feedback.0.lock().unwrap().fatal.is_some());
            }
            assert_eq!(io.feedback_stats().written_targets,0);
            assert!(!io.bus.lock().unwrap().enabled,"reset must not mark torque enabled");
        }
    }

    #[test]
    fn serial_open_respects_an_existing_flock() {
        use serialport::SerialPort;
        let (_master, slave) = serialport::TTYPort::pair().unwrap();
        let path = slave.name().unwrap();
        drop(slave);
        let other = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        other.try_lock().unwrap();
        assert!(open_serial(&path).is_err());
    }

    #[test]
    fn packet_alarms_checksum_and_units_are_not_xl330() {
        assert_eq!(
            instruction(1, 2, &[56, 15]).unwrap(),
            vec![255, 255, 1, 4, 2, 56, 15, 177]
        );
        let mut packet = vec![255, 255, 1, 4, 0, 0, 8];
        packet.push(checksum(&packet[2..]));
        assert_eq!(decode_packet(&packet, 1, 2).unwrap(), vec![0, 8]);
        assert!(decode_packet(&packet, 2, 2).is_err());
        packet[4] = 2;
        let end = packet.len() - 1;
        packet[end] = checksum(&packet[2..end]);
        assert!(
            decode_packet(&packet, 1, 2)
                .unwrap_err()
                .to_string()
                .ends_with("alarm 0x02")
        );
        packet[end] ^= 1;
        assert!(decode_packet(&packet, 1, 2).is_err());
        assert_eq!(signed([1, 128]), -1);
        assert_eq!(signed([0, 8]), 2048);
    }
    #[test]
    fn actual_mapping_is_named_and_motion_still_unverified() {
        let config: Installation =
            serde_json::from_str(include_str!("../../../radxa/installation.json")).unwrap();
        config.validate().unwrap();
        assert_eq!(config.joints[0].id, 10);
        assert_eq!(config.joints[9].id, 15);
    }
    fn fixture() -> (FeetechIo, serialport::TTYPort) {
        let (client, server) = serialport::TTYPort::pair().unwrap();
        nonblocking(&client).unwrap();
        let cfg: Installation =
            serde_json::from_str(include_str!("../../../radxa/installation.json")).unwrap();
        (
            FeetechIo {
                luwu_native: false,
                filtered_velocity: None,
                saturated_ids: Vec::new(),
                port: client,
                ids: cfg.joints.iter().map(|j| j.id).collect(),
                joints: cfg.joints,
                imu: std::sync::Arc::new(Bno08x::fixture()),
                limits: vec![(0, 4095, 80); 15],
                allow_motion: false,
                enabled: true,
                healthy: None,
                slow: None,
                recovery: FeedbackRecovery::default(),
                torque_poll_interval: None,
                torque_checked_at: Some(Instant::now()),
                telemetry: serde_json::Value::Null,
                servo_gain_profile: None,
                servo_gains_verified: false,
                scheduled_bus: false,
                transaction_deadline: None,
                alternate_feedback: false,
            },
            server,
        )
    }
    fn respond(port: &mut serialport::TTYPort, ids: &[u8], alarm: u8) {
        let mut request = vec![0; 23];
        port.read_exact(&mut request).unwrap();
        assert_eq!(
            request,
            instruction(254, 0x82, &[&[56, 15][..], ids].concat()).unwrap()
        );
        for id in ids {
            let mut raw = [0u8; 15];
            raw[1] = 8;
            raw[2] = 1;
            raw[3] = 128;
            raw[6] = 74;
            raw[7] = 30;
            raw[14] = 128;
            raw[13] = 2;
            let mut packet = vec![255, 255, *id, 17, alarm];
            packet.extend(raw);
            packet.push(checksum(&packet[2..]));
            port.write_all(&packet).unwrap();
        }
    }
    fn expect(port: &mut serialport::TTYPort, packet: &[u8]) {
        let mut request = vec![0; packet.len()];
        port.read_exact(&mut request).unwrap();
        assert_eq!(request, packet);
    }
    fn reply_rows(port: &mut serialport::TTYPort, ids: &[u8], address: u8, rows: &[Vec<u8>]) {
        expect(
            port,
            &instruction(
                254,
                0x82,
                &[&[address, rows[0].len() as u8][..], ids].concat(),
            )
            .unwrap(),
        );
        for (id, row) in ids.iter().zip(rows) {
            let mut packet = vec![255, 255, *id, (row.len() + 2) as u8, 0];
            packet.extend(row);
            packet.push(checksum(&packet[2..]));
            port.write_all(&packet).unwrap();
        }
    }
    fn expect_write(port: &mut serialport::TTYPort, ids: &[u8], address: u8, value: &[u8]) {
        let mut params = vec![address, value.len() as u8];
        for id in ids {
            params.push(*id);
            params.extend(value);
        }
        expect(port, &instruction(254, 0x83, &params).unwrap());
    }

    #[test]
    fn luwu_gains_are_volatile_per_id_and_verified() {
        let (mut io, mut peer) = fixture();
        io.servo_gain_profile = Some(ServoGainProfile::LuwuRuntime);
        let ids = io.ids.clone();
        let rows = ServoGainProfile::LuwuRuntime.rows(&io.joints);
        assert_eq!(rows[9], [10, 20]); // Physical mouth ID15, not policy index15.
        assert_eq!(rows[0], [6, 20]);
        let worker = std::thread::spawn(move || {
            reply_rows(&mut peer, &ids, 50, &vec![vec![5, 20]; 15]);
            let mut params = vec![50, 2];
            for (id, row) in ids.iter().zip(&rows) { params.push(*id); params.extend(row); }
            expect(&mut peer, &instruction(254, 0x83, &params).unwrap());
            reply_rows(&mut peer, &ids, 50, &rows);
            peer
        });
        io.configure_gains(true).unwrap();
        assert!(io.servo_gains_verified);
        worker.join().unwrap();
    }

    #[test]
    fn luwu_gains_never_change_under_load_or_accept_bad_readback() {
        for all_off in [false, true] {
            let (mut io, mut peer) = fixture();
            io.servo_gain_profile = Some(ServoGainProfile::LuwuRuntime);
            let ids = io.ids.clone();
            let rows = ServoGainProfile::LuwuRuntime.rows(&io.joints);
            let worker = std::thread::spawn(move || {
                reply_rows(&mut peer, &ids, 50, &vec![vec![5, 20]; 15]);
                if all_off {
                    let mut params = vec![50, 2];
                    for (id, row) in ids.iter().zip(&rows) { params.push(*id); params.extend(row); }
                    expect(&mut peer, &instruction(254, 0x83, &params).unwrap());
                    reply_rows(&mut peer, &ids, 50, &vec![vec![5, 20]; 15]);
                }
                peer
            });
            assert!(io.configure_gains(all_off).is_err());
            assert!(!io.servo_gains_verified);
            let peer = worker.join().unwrap();
            assert_eq!(peer.bytes_to_read().unwrap(), 0);
        }
    }

    #[test]
    fn adopted_enable_cannot_write_with_unverified_luwu_gains() {
        let (mut io, peer) = fixture();
        io.allow_motion = true;
        io.healthy = Some(Instant::now());
        io.servo_gain_profile = Some(ServoGainProfile::LuwuRuntime);
        assert!(io.write(&JointTargets::new([0.; 15])).unwrap_err().to_string().contains("P/D not verified"));
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }
    #[test]
    fn pty_noncenter_zero_offsets_roundtrip_without_adding_home() {
        let (mut io, mut peer) = fixture();
        for joint in &mut io.joints {
            joint.zero_ticks = match joint.id {
                2 | 3 => 2713,
                7 | 8 => 1383,
                _ => 2048,
            };
        }
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            expect_write(&mut peer, &ids, 42, &[0, 8]);
            peer
        });
        let sensors = io.read().unwrap();
        assert!((sensors.positions[3] + 665. * RAD_TICK).abs() < 1e-12);
        assert!((sensors.positions[13] - 665. * RAD_TICK).abs() < 1e-12);
        io.allow_motion = true;
        io.write(&JointTargets::new(sensors.positions)).unwrap();
        worker.join().unwrap();
    }
    #[test]
    fn pty_explicit_enable_preloads_position_and_verifies_before_torque() {
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        io.enabled = false;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            reply_rows(&mut peer, &ids, 40, &vec![vec![0, 0, 123, 0]; 15]);
            reply_rows(&mut peer, &ids, 56, &vec![vec![0, 8]; 15]);
            expect_write(&mut peer, &ids, 42, &[0, 8]);
            reply_rows(&mut peer, &ids, 42, &vec![vec![0, 8]; 15]);
            expect_write(&mut peer, &ids, 40, &[1]);
            reply_rows(&mut peer, &ids, 40, &vec![vec![1]; 15]);
            peer
        });
        io.set_torque(true).unwrap();
        worker.join().unwrap();
        assert!(io.enabled);
    }
    #[test]
    fn pty_bad_preload_does_not_enable() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        io.enabled = false;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            reply_rows(&mut peer, &ids, 40, &vec![vec![0; 4]; 15]);
            reply_rows(&mut peer, &ids, 56, &vec![vec![0, 8]; 15]);
            expect_write(&mut peer, &ids, 42, &[0, 8]);
            reply_rows(&mut peer, &ids, 42, &vec![vec![0, 0]; 15]);
            peer
        });
        assert!(
            io.set_torque(true)
                .unwrap_err()
                .to_string()
                .contains("preload readback")
        );
        let peer = worker.join().unwrap();
        assert!(!io.enabled);
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }
    #[test]
    fn pty_partial_torque_loss_blocks_future_targets() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            let mut rows = vec![vec![0; 31]; 15];
            for row in &mut rows {
                row[0] = 1;
                row[17] = 8;
                row[23] = 30;
            }
            rows[3][0] = 0;
            reply_rows(&mut peer, &ids, 40, &rows);
            peer
        });
        assert!(
            io.read()
                .unwrap_err()
                .to_string()
                .contains("enable feedback lost")
        );
        let peer = worker.join().unwrap();
        assert!(!io.enabled);
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }
    #[test]
    fn pty_timeout_reports_missing_device_and_leaves_output_stale() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        io.enabled = false;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            expect(
                &mut peer,
                &instruction(254, 0x82, &[&[56, 15][..], &ids].concat()).unwrap(),
            );
            let mut packet = vec![255, 255, ids[0], 17, 0];
            packet.extend([0u8; 15]);
            packet.push(checksum(&packet[2..]));
            peer.write_all(&packet).unwrap();
            std::thread::sleep(Duration::from_millis(60));
            peer
        });
        let error = io.read().unwrap_err().to_string();
        assert!(
            error.contains("addr=56 size=15 missing_ids=[9,")
                && error.contains("received_bytes=21"),
            "{error}"
        );
        assert!(io.healthy.is_none());
        assert!(io.write(&JointTargets::new([0.; NUM_JOINTS])).is_err());
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_hold_readback_never_writes_enable_or_targets() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            let mut rows = vec![vec![1]; 15];
            rows[1][0] = 0;
            reply_rows(&mut peer, &ids, 40, &rows);
            peer
        });
        let enabled = io.torque_enabled_ids().unwrap();
        assert_eq!(enabled.len(), 14);
        assert!(!enabled.contains(&9));
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }
    #[test]
    fn pty_group_read_scales_named_feedback_without_actuation() {
        let (mut io, mut peer) = fixture();
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            peer
        });
        let s = io.read().unwrap();
        worker.join().unwrap();
        assert_eq!(s.positions, [0.; 15]);
        assert!((s.velocities[0] - 0.732 * std::f64::consts::TAU / 60.).abs() < 1e-12);
        assert_eq!(s.currents_ma, [13.; 15]);
        assert!((io.slow_sensors().unwrap().volts - 7.4).abs() < 1e-12);
        assert_eq!(io.telemetry["motion_available"], false);
    }
    #[test]
    fn pty_enabled_read_uses_one_coherent_packet_without_extra_request() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            let mut rows = vec![vec![0; 31]; 15];
            for row in &mut rows {
                row[0] = 1;
                row[17] = 8;
                row[18] = 1;
                row[19] = 128;
                row[22] = 74;
                row[23] = 30;
                row[29] = 2;
                row[30] = 128;
            }
            reply_rows(&mut peer, &ids, 40, &rows);
            peer
        });
        let sensor = io.read().unwrap();
        assert_eq!(sensor.positions, [0.; 15]);
        assert_eq!(sensor.currents_ma, [13.; 15]);
        assert!((sensor.velocities[0] - 0.732 * std::f64::consts::TAU / 60.).abs() < 1e-12);
        assert!(io.enabled);
        assert_eq!(io.telemetry["motion_available"], true);
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_combined_feedback_rejects_register_alarm_without_output() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            let mut rows = vec![vec![0; 31]; 15];
            for row in &mut rows {
                row[0] = 1;
                row[17] = 8;
                row[23] = 30;
            }
            rows[1][25] = 4;
            reply_rows(&mut peer, &ids, 40, &rows);
            peer
        });
        assert!(
            io.read()
                .unwrap_err()
                .to_string()
                .ends_with("ID 9: alarm 0x04")
        );
        assert!(io.healthy.is_none());
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    fn enabled_rows() -> Vec<Vec<u8>> {
        let mut rows = vec![vec![0; 31]; 15];
        for row in &mut rows {
            row[0] = 1;
            row[17] = 8;
            row[22] = 74;
            row[23] = 30;
        }
        rows
    }

    #[test]
    fn threaded_feedback_updates_without_control_reads_or_writes() {
        use serialport::SerialPort;
        let (mut bus, mut peer) = fixture();
        bus.allow_motion = true;
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || {
            for ticks in [1, 2, 3] {
                let mut rows: Vec<Vec<u8>> = enabled_rows()
                    .into_iter()
                    .map(|r| r[16..].to_vec())
                    .collect();
                for row in &mut rows {
                    row[0] = ticks;
                }
                reply_rows(&mut peer, &ids, 56, &rows);
            }
            peer
        });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        let first = io.read().unwrap();
        let deadline = Instant::now() + Duration::from_millis(100);
        while io.feedback_stats().complete_reads < 3 {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let latest = io.read().unwrap();
        assert_ne!(first.positions, latest.positions);
        assert_eq!(io.feedback_stats().complete_reads, 3);
        let _keep_bus = io.bus.clone();
        drop(io);
        let peer = server.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn threaded_write_is_acknowledged_and_drop_never_changes_torque() {
        use serialport::SerialPort;
        let (mut bus, mut peer) = fixture();
        bus.allow_motion = true;
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            expect_write(&mut peer, &ids, 42, &[0, 8]);
            peer
        });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        let sample = io.read().unwrap();
        io.write(&JointTargets::new(sample.positions)).unwrap();
        let _keep_bus = io.bus.clone();
        drop(io);
        let peer = server.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn threaded_alarm_blocks_targets_even_with_a_previously_valid_snapshot() {
        use serialport::SerialPort;
        let (mut bus, mut peer) = fixture();
        bus.allow_motion = true;
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            let mut rows: Vec<Vec<u8>> = enabled_rows()
                .into_iter()
                .map(|r| r[16..].to_vec())
                .collect();
            rows[1][9] = 4;
            reply_rows(&mut peer, &ids, 56, &rows);
            peer
        });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        io.read().unwrap();
        let deadline = Instant::now() + Duration::from_millis(100);
        while io.feedback_stats().failed_reads == 0 {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            io.write(&JointTargets::new([0.; 15]))
                .unwrap_err()
                .to_string()
                .contains("alarm 0x04")
        );
        let fault = io.feedback().unwrap();
        assert!(fault["error"].as_str().unwrap().contains("alarm 0x04"));
        assert_eq!(fault["control_valid"], false);
        let _keep_bus = io.bus.clone();
        drop(io);
        let peer = server.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn threaded_expired_observation_cannot_send_target_with_fresh_cached_sample() {
        use serialport::SerialPort;
        let (mut bus, mut peer) = fixture();
        bus.allow_motion = true;
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            peer
        });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        io.read().unwrap();
        io.observation_at = Some(Instant::now() - Duration::from_millis(110));
        assert!(matches!(
            io.write(&JointTargets::new([0.; 15])),
            Err(IoError::StaleTarget)
        ));
        let _keep_bus = io.bus.clone();
        drop(io);
        let peer = server.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn snapshot_is_nonblocking_while_serial_transaction_is_owned() {
        let (bus, mut peer) = fixture();
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || { respond(&mut peer, &ids, 0); peer });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        let bus = io.bus.clone();
        let guard = bus.lock().unwrap();
        let start = Instant::now();
        let sample = io.read().unwrap();
        assert!(start.elapsed() < Duration::from_millis(5));
        let feedback = io.feedback().unwrap();
        assert_eq!(feedback["imu"]["gravity"], serde_json::json!(sample.imu.gravity));
        assert_eq!(feedback["positions"], serde_json::json!(sample.positions));
        drop(guard);
        drop(io);
        let _peer = server.join().unwrap();
    }

    #[test]
    fn busy_bus_does_not_block_control_and_next_tick_sends_only_new_target() {
        use serialport::SerialPort;
        let (mut bus, mut peer) = fixture();
        bus.allow_motion = true;
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            // The discarded 0.1 rad proposal must never reach this port.
            expect_write(&mut peer, &ids, 42, &[0, 8]);
            respond(&mut peer, &ids, 0);
            peer
        });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        let bus = io.bus.clone();
        let guard = bus.lock().unwrap();
        let start = Instant::now();
        io.read().unwrap();
        io.set_gain(200).unwrap();
        assert_eq!(io.slow_sensors().unwrap().temps_c, [30.; 15]);
        assert!(matches!(io.write(&JointTargets::new([0.1; 15])), Err(IoError::BusBusy)));
        let busy_ms = start.elapsed().as_secs_f64() * 1000.;
        assert!(busy_ms < 10., "control blocked for {busy_ms:.3} ms");
        // Simulate the half-duplex reader occupying a complete control period.
        std::thread::sleep(CONTROL_PERIOD);
        drop(guard);
        io.read().unwrap();
        io.write(&JointTargets::new([0.; 15])).unwrap();
        let deadline = Instant::now() + Duration::from_millis(100);
        while io.feedback_stats().complete_reads < 2 {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let stats = io.feedback_stats();
        assert_eq!(stats.busy_targets, 1);
        assert_eq!(stats.written_targets, 1);
        eprintln!("busy control={busy_ms:.3}ms, submitted write={:.3}ms", stats.write_ms);
        let _keep_bus = io.bus.clone();
        drop(io);
        let peer = server.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn slow_sensors_never_wait_for_missing_or_stale_feedback() {
        let (bus, mut peer) = fixture();
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || { respond(&mut peer, &ids, 0); peer });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        let bus = io.bus.clone();
        let guard = bus.lock().unwrap();
        io.feedback.0.lock().unwrap().sample.as_mut().unwrap().2 =
            Instant::now() - Duration::from_millis(90);
        let start = Instant::now();
        assert!(matches!(io.slow_sensors(), Err(IoError::IncompleteFeedback(_))));
        io.feedback.0.lock().unwrap().sample = None;
        assert!(matches!(io.slow_sensors(), Err(IoError::IncompleteFeedback(_))));
        assert!(start.elapsed() < Duration::from_millis(10));
        drop(guard);
        drop(io);
        let _peer = server.join().unwrap();
    }

    #[test]
    fn control_ticks_continue_during_a_70ms_reader_stall() {
        let (mut bus, mut peer) = fixture();
        bus.allow_motion = true;
        let ids = bus.ids.clone();
        let server = std::thread::spawn(move || {
            respond(&mut peer, &ids, 0);
            respond(&mut peer, &ids, 0);
            expect_write(&mut peer, &ids, 42, &[0, 8]);
            peer
        });
        let mut io = ThreadedFeetechIo::from_io(bus).unwrap();
        io.snapshot(Instant::now() + Duration::from_millis(100)).unwrap();
        let bus = io.bus.clone();
        let (ready, started) = std::sync::mpsc::channel();
        let stalled_reader = std::thread::spawn(move || {
            let _guard = bus.lock().unwrap();
            ready.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(70));
        });
        started.recv_timeout(Duration::from_secs(1)).unwrap();
        let begin = Instant::now();
        let mut max_tick = Duration::ZERO;
        let mut busy = 0;
        let mut sent = 0;
        for tick in 0..6 {
            std::thread::sleep((begin + CONTROL_PERIOD * tick).saturating_duration_since(Instant::now()));
            let start = Instant::now();
            io.set_gain(200).unwrap();
            let _ = io.slow_sensors();
            if io.read().is_ok() {
                let target = JointTargets::new([if tick == 5 { 0. } else { 0.1 }; 15]);
                match io.write(&target) {
                    Ok(()) => sent += 1,
                    Err(IoError::BusBusy) => busy += 1,
                    result => panic!("unexpected control result: {result:?}"),
                }
            }
            max_tick = max_tick.max(start.elapsed());
        }
        assert!(busy >= 3, "{busy} busy ticks");
        assert_eq!(sent, 1);
        assert!(max_tick < Duration::from_millis(10), "{max_tick:?}");
        eprintln!("70ms reader stall: {busy} busy ticks, {sent} new target, max control={:.3}ms", max_tick.as_secs_f64() * 1000.);
        stalled_reader.join().unwrap();
        let _keep_bus = io.bus.clone();
        drop(io);
        let _peer = server.join().unwrap();
    }

    #[test]
    fn goal_write_backpressure_has_a_packet_deadline() {
        let (mut io, _peer) = fixture();
        io.port.set_timeout(Duration::from_millis(1)).unwrap();
        let filling = Instant::now();
        loop {
            assert!(filling.elapsed() < Duration::from_secs(1));
            match io.port.write(&[0; 4096]) {
                Ok(n) if n > 0 => {},
                Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) => break,
                result => panic!("unexpected PTY fill result: {result:?}"),
            }
        }
        io.allow_motion = true;
        io.healthy = Some(Instant::now());
        let start = Instant::now();
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        let elapsed = start.elapsed();
        eprintln!("backpressured goal write={:.3}ms", elapsed.as_secs_f64() * 1000.);
        assert!(elapsed < Duration::from_millis(15), "{elapsed:?}");
    }

    #[test]
    fn fast_feedback_still_checks_torque_when_due() {
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        io.torque_poll_interval = Some(Duration::from_millis(100));
        io.torque_checked_at = None;
        let ids = io.ids.clone();
        let server = std::thread::spawn(move || {
            reply_rows(&mut peer, &ids, 40, &vec![vec![1]; 15]);
            respond(&mut peer, &ids, 0);
            respond(&mut peer, &ids, 0);
            peer
        });
        io.read().unwrap();
        io.read().unwrap();
        server.join().unwrap();
        assert!(io.torque_checked_at.is_some());
    }

    #[test]
    fn fast_feedback_never_ignores_partial_enable() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        io.torque_poll_interval = Some(Duration::from_millis(100));
        io.torque_checked_at = None;
        let ids = io.ids.clone();
        let server = std::thread::spawn(move || {
            let mut rows = vec![vec![1]; 15];
            rows[1][0] = 0;
            reply_rows(&mut peer, &ids, 40, &rows);
            peer
        });
        assert!(
            io.read()
                .unwrap_err()
                .to_string()
                .contains("enable feedback lost")
        );
        assert!(!io.enabled);
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        let peer = server.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn fast_feedback_recovery_uses_fresh_alternate_length() {
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        io.torque_poll_interval = Some(Duration::from_millis(100));
        let ids = io.ids.clone();
        let server = std::thread::spawn(move || {
            expect(
                &mut peer,
                &instruction(254, 0x82, &[&[56, 15][..], &ids].concat()).unwrap(),
            );
            peer.write_all(&[255, 255, ids[0], 17, 0]).unwrap();
            let rows: Vec<Vec<u8>> = enabled_rows()
                .into_iter()
                .map(|r| r[15..].to_vec())
                .collect();
            reply_rows(&mut peer, &ids, 55, &rows);
            peer
        });
        assert_eq!(io.read().unwrap().positions, [0.; 15]);
        assert_eq!(io.recovery.recovered, 1);
        server.join().unwrap();
    }

    fn partial_response(peer: &mut serialport::TTYPort, ids: &[u8], alarm: u8) {
        expect(
            peer,
            &instruction(254, 0x82, &[&[40, 31][..], ids].concat()).unwrap(),
        );
        let row = enabled_rows().remove(0);
        let mut packet = vec![255, 255, ids[0], 33, alarm];
        packet.extend(row);
        packet.push(checksum(&packet[2..]));
        peer.write_all(&packet).unwrap();
        peer.write_all(&[255, 255, ids[1], 33]).unwrap();
    }

    #[test]
    fn pty_truncated_motion_feedback_recovers_with_fresh_full_read() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            partial_response(&mut peer, &ids, 0);
            let mut rows = enabled_rows();
            for row in &mut rows {
                row[16] = 10;
                row.insert(0, 0);
            }
            reply_rows(&mut peer, &ids, 39, &rows);
            peer
        });
        let sensor = io.read().unwrap();
        assert!((sensor.positions[0].abs() - 10. * RAD_TICK).abs() < 1e-12);
        assert_eq!(io.recovery.attempts, 1);
        assert_eq!(io.recovery.recovered, 1);
        assert!(io.healthy.is_some());
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_stale_original_packets_cannot_satisfy_recovery() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            partial_response(&mut peer, &ids, 0);
            expect(
                &mut peer,
                &instruction(254, 0x82, &[&[39, 32][..], &ids].concat()).unwrap(),
            );
            for (id, row) in ids.iter().zip(enabled_rows()) {
                let mut packet = vec![255, 255, *id, 33, 0];
                packet.extend(row);
                packet.push(checksum(&packet[2..]));
                peer.write_all(&packet).unwrap();
            }
            std::thread::sleep(Duration::from_millis(50));
            peer
        });
        assert!(io.read().is_err());
        assert_eq!(io.recovery.attempts, 1);
        assert_eq!(io.recovery.recovered, 0);
        assert!(io.healthy.is_none());
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_servo_alarm_before_truncation_is_never_hidden_by_recovery() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            partial_response(&mut peer, &ids, 4);
            std::thread::sleep(Duration::from_millis(50));
            peer
        });
        assert!(io.read().unwrap_err().to_string().ends_with("alarm 0x04"));
        assert_eq!(io.recovery.attempts, 0);
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_retry_alarm_is_not_retried_or_used_to_enable_motion() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            partial_response(&mut peer, &ids, 0);
            let mut rows = enabled_rows();
            rows[1][25] = 4;
            for row in &mut rows {
                row.insert(0, 0);
            }
            reply_rows(&mut peer, &ids, 39, &rows);
            peer
        });
        assert!(io.read().unwrap_err().to_string().ends_with("alarm 0x04"));
        assert_eq!(io.recovery.attempts, 1);
        assert_eq!(io.recovery.recovered, 0);
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_framing_recovers_noise_fragmentation_and_unordered_ids() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            expect(
                &mut peer,
                &instruction(254, 0x82, &[&[40, 31][..], &ids].concat()).unwrap(),
            );
            let mut stream = vec![19, 255, 255, ids[0], 33, 0, 0, 123];
            for (id, row) in ids.iter().rev().zip(enabled_rows()) {
                let mut packet = vec![255, 255, *id, 33, 0];
                packet.extend(row);
                packet.push(checksum(&packet[2..]));
                stream.extend(packet);
            }
            for chunk in stream.chunks(7) {
                peer.write_all(chunk).unwrap();
            }
            peer
        });
        assert_eq!(io.read().unwrap().positions, [0.; 15]);
        assert_eq!(io.recovery.attempts, 0);
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_angle_alarm_cannot_hide_other_joint_hard_alarm() {
        use serialport::SerialPort;
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            let mut rows = enabled_rows();
            rows[0][25] = 2;
            rows[1][25] = 4;
            reply_rows(&mut peer, &ids, 40, &rows);
            peer
        });
        assert!(
            io.read()
                .unwrap_err()
                .to_string()
                .ends_with("ID 9: alarm 0x04")
        );
        assert_eq!(io.recovery.attempts, 0);
        let peer = worker.join().unwrap();
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }

    #[test]
    fn pty_angle_alarm_retries_five_times_then_fails_without_writes() {
        let (mut io, mut peer) = fixture();
        let ids = io.ids.clone();
        let worker = std::thread::spawn(move || {
            for _ in 0..6 {
                respond(&mut peer, &ids, 2)
            }
            peer
        });
        assert!(io.read().unwrap_err().to_string().ends_with("alarm 0x02"));
        worker.join().unwrap();
        assert!(io.healthy.is_none());
    }
    #[test]
    fn pty_write_is_only_position_no_current_gain_or_eeprom() {
        let (mut io, mut peer) = fixture();
        io.allow_motion = true;
        io.healthy = Some(Instant::now());
        io.write(&JointTargets::new([0.; 15])).unwrap();
        let mut packet = vec![0; 53];
        peer.read_exact(&mut packet).unwrap();
        let mut params = vec![42, 2];
        for id in &io.ids {
            params.extend([*id, 0, 8]);
        }
        assert_eq!(packet, instruction(254, 0x83, &params).unwrap());
        assert!(io.set_gain(50).is_err());
    }
    #[test]
    fn luwu_saturation_keeps_firmware_limits_ids_and_position_only_writes() {
        let (mut io, mut peer) = fixture();
        io.allow_motion = true; io.healthy = Some(Instant::now()); io.luwu_native = true;
        io.limits = vec![(100,3900,80);15];
        io.write(&JointTargets::new([10.;15])).unwrap();
        let mut packet=vec![0;53];peer.read_exact(&mut packet).unwrap();
        let mut params=vec![42,2];
        for j in &io.joints {
            let ticks: u16=if j.direction<0 {100}else{3900};
            params.push(j.id);params.extend(ticks.to_le_bytes());
        }
        assert_eq!(packet,instruction(254,0x83,&params).unwrap());
        assert_eq!(io.saturated_ids,io.ids);
        assert!(io.write(&JointTargets::new([f64::NAN;15])).is_err());
        assert_eq!(peer.bytes_to_read().unwrap(),0);
    }
    #[test]
    fn pty_invalid_stale_or_disallowed_goals_emit_no_packet() {
        use serialport::SerialPort;
        let (mut io, peer) = fixture();
        assert!(io.set_torque(true).is_err());
        assert!(io.set_torque(false).is_err());
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        io.allow_motion = true;
        io.healthy = Some(Instant::now());
        let mut q = [0.; 15];
        q[14] = f64::NAN;
        assert!(io.write(&JointTargets::new(q)).is_err());
        q[14] = 10.;
        assert!(io.write(&JointTargets::new(q)).is_err());
        io.healthy = Some(Instant::now() - Duration::from_secs(1));
        assert!(io.write(&JointTargets::new([0.; 15])).is_err());
        assert_eq!(peer.bytes_to_read().unwrap(), 0);
    }
}
