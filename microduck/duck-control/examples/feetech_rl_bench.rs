//! Supported-robot acceptance only; not a walking service or calibration bypass.
//! Native RobotIo + exported reference policy, 50 Hz, finite or continuous M6 run.
//! M6 completion holds the last target; failures still verify torque-off.
//! Legacy explicit HOLD mode never sends torque-off.
//! No PD/current/EEPROM changes or automatic motion retries after a fault.
use duck_control::feetech::ThreadedFeetechIo as FeetechIo;
use duck_control::io::{IoError, JointTargets, RobotIo};
use duck_control::model::{DEFAULT_POSITION, NUM_JOINTS};
use duck_control::obs::{Command, Observation};
use duck_control::policy::{Net, Policy, PolicyPaths};
use serde::Deserialize;
use serde_json::json;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
#[path = "support/bench_control.rs"]
mod bench_control;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    joint_names: Vec<String>,
    ranges: [[f64; 2]; NUM_JOINTS],
}
fn log(value: serde_json::Value) -> Result<()> {
    writeln!(std::io::stdout().lock(), "{value}")?;
    Ok(())
}
fn valid(target: &[f64; NUM_JOINTS], limits: &Limits) -> Result<()> {
    for (i, q) in target.iter().enumerate() {
        if !q.is_finite() || *q < limits.ranges[i][0] || *q > limits.ranges[i][1] {
            return Err(format!(
                "{} target {q} outside model range {:?}",
                limits.joint_names[i], limits.ranges[i]
            )
            .into());
        }
    }
    Ok(())
}
fn check_stop(last: Instant) -> Result<()> {
    if STOP.load(Ordering::Relaxed) {
        return Err("interrupted".into());
    }
    if last.elapsed() > Duration::from_millis(100) {
        return Err(format!(
            "control deadline missed: last successful write {:.1}ms ago",
            last.elapsed().as_secs_f64() * 1000.
        )
        .into());
    }
    Ok(())
}

fn retry_feedback<T>(
    last: Instant,
    mut read: impl FnMut() -> duck_control::io::Result<T>,
) -> Result<T> {
    loop {
        check_stop(last)?;
        match read() {
            Ok(sample) => {
                check_stop(last)?;
                return Ok(sample);
            }
            Err(IoError::IncompleteFeedback(_)) => {
                // The worker may recover a missing frame. Never extend the write
                // deadline, infer from stale data, or retry a reported device fault.
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn window_open(end: Option<Instant>, now: Instant) -> bool {
    end.is_none_or(|deadline| now < deadline)
}

fn scaled_action(raw: [f32; 14], previous: [f32; 14], scale: f32) -> [f32; 14] {
    if scale == 1.0 { return raw; }
    // Scaling an absolute HOME delta after the ONNX slew bound can otherwise
    // exceed that same step bound. History always contains this applied output.
    std::array::from_fn(|i| (raw[i] * scale).clamp(previous[i] - 0.1, previous[i] + 0.1))
}
fn check_attitude(gravity: &[f64; 3], require_upright: bool) -> Result<()> {
    if gravity.iter().any(|v| !v.is_finite()) {
        return Err("invalid attitude feedback".into());
    }
    if require_upright && gravity[2] > -0.94 {
        return Err("support trunk upright".into());
    }
    Ok(())
}

fn check_running_attitude(
    gravity: &[f64; 3],
    falling_since: &mut Option<Instant>,
    now: Instant,
) -> Result<()> {
    check_attitude(gravity, false)?;
    let config = duck_control::safety::SafetyConfig::default();
    if gravity[2] > config.fall_gravity_z {
        let since = falling_since.get_or_insert(now);
        if now.saturating_duration_since(*since) >= config.fall_debounce {
            return Err(format!("sustained fall during RL: gravity_z={:.4}", gravity[2]).into());
        }
    } else {
        *falling_since = None;
    }
    Ok(())
}

fn next_tick(due: Instant, completed: Instant) -> Instant {
    let next = due + Duration::from_millis(20);
    if completed >= next {
        completed + Duration::from_millis(20)
    } else {
        next
    }
}

fn retry_tick(last_write: Instant, now: Instant) -> Instant {
    // BusBusy did not submit a goal. Poll the reader's reserved write slot soon,
    // without extending the watchdog or resending the previous proposal.
    (now + Duration::from_millis(2)).min(last_write + Duration::from_millis(100))
}

fn reach_pose(
    io: &mut FeetechIo,
    limits: &Limits,
    goal: [f64; NUM_JOINTS],
    phase: &str,
    require_upright: bool,
    enable: bool,
) -> Result<()> {
    let warmup = Instant::now();
    // IMU and serial workers start independently; IMU readiness is not joint readiness.
    let initial = loop {
        if STOP.load(Ordering::Relaxed) || warmup.elapsed() > Duration::from_secs(10) {
            return Err(format!("sensor startup: {:?}", io.read()).into());
        }
        if io.imu_ready() {
            match io.read() {
                Ok(sample) => break sample,
                Err(IoError::IncompleteFeedback(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    check_attitude(&initial.imu.gravity, require_upright)?;
    valid(&initial.positions, limits)?;
    valid(&goal, limits)?;
    if enable {
        io.set_torque(true)?;
        // Enabling intentionally invalidates the pre-enable cache. Wait for the
        // next complete sample, without sending a target based on that old cache.
        let enabled_at = Instant::now();
        loop {
            check_stop(enabled_at)?;
            match io.read() {
                Ok(sensor) => {
                    check_attitude(&sensor.imu.gravity, require_upright)?;
                    valid(&sensor.positions, limits)?;
                    break;
                }
                Err(IoError::IncompleteFeedback(_)) => std::thread::sleep(Duration::from_millis(2)),
                Err(error) => return Err(error.into()),
            }
        }
    }
    let mut last = Instant::now();
    // Smooth positioning first; no policy evaluated on the folded startup pose.
    for seq in 0..250 {
        check_stop(last)?;
        let begin = Instant::now();
        let sensor = retry_feedback(last, || io.read())?;
        check_attitude(&sensor.imu.gravity, require_upright)?;
        valid(&sensor.positions, limits)?;
        let t = (seq + 1) as f64 / 250.;
        let t = t * t * (3. - 2. * t);
        let target =
            std::array::from_fn(|i| initial.positions[i] + t * (goal[i] - initial.positions[i]));
        valid(&target, limits)?;
        match io.write(&JointTargets::new(target)) {
            Ok(()) => {}
            Err(IoError::BusBusy | IoError::StaleTarget | IoError::IncompleteFeedback(_)) => {
                std::thread::sleep(Duration::from_millis(20).saturating_sub(begin.elapsed()));
                continue;
            }
            Err(error) => return Err(error.into()),
        }
        last = Instant::now();
        log(
            json!({"phase":phase,"seq":seq,"positions":sensor.positions,"targets":target,
            "gravity":sensor.imu.gravity}),
        )?;
        std::thread::sleep(Duration::from_millis(20).saturating_sub(begin.elapsed()));
    }
    let settling = Instant::now();
    let mut settle_seq = 0;
    while settling.elapsed() < Duration::from_secs(5) {
        check_stop(last)?;
        let sensor = retry_feedback(last, || io.read())?;
        check_attitude(&sensor.imu.gravity, require_upright)?;
        valid(&sensor.positions, limits)?;
        match io.write(&JointTargets::new(goal)) {
            Ok(()) => {}
            Err(IoError::BusBusy | IoError::StaleTarget | IoError::IncompleteFeedback(_)) => {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
            Err(error) => return Err(error.into()),
        }
        last = Instant::now();
        let error = sensor
            .positions
            .iter()
            .zip(goal)
            .map(|(q, target)| (q - target).abs())
            .fold(0.0_f64, f64::max);
        if settle_seq % 5 == 0 {
            log(json!({"phase":"pose_settle","pose":phase,
                "elapsed_s":settling.elapsed().as_secs_f64(),"positions":sensor.positions,
                "targets":goal,"max_error_deg":error.to_degrees(),
                "currents_ma":sensor.currents_ma,"feedback":io.feedback()}))?;
        }
        settle_seq += 1;
        if settling.elapsed() >= Duration::from_secs(1) && error <= 0.07 {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let sensor = retry_feedback(last, || io.read())?;
    let max_error = sensor
        .positions
        .iter()
        .zip(goal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    if max_error > 0.07 {
        log(
            json!({"phase":"pose_not_reached","pose":phase,"positions":sensor.positions,
            "targets":goal,"max_error_deg":max_error.to_degrees(),
            "currents_ma":sensor.currents_ma,"feedback":io.feedback()}),
        )?;
        return Err(format!("{phase} feedback not reached; policy not enabled").into());
    }
    log(
        json!({"phase":"reached", "pose":phase, "positions":sensor.positions,
        "max_error_deg":max_error.to_degrees(), "gravity":sensor.imu.gravity}),
    )?;
    Ok(())
}

fn exercise(
    io: &mut FeetechIo,
    policy: &mut Policy,
    limits: &Limits,
    zero_first: bool,
    duration_s: Option<u32>,
    mut command: Command,
    supported_m6: bool,
    action_scale: f32,
    remote: Option<&std::sync::Arc<std::sync::Mutex<bench_control::Intent>>>,
) -> Result<()> {
    if zero_first {
        reach_pose(io, limits, [0.; NUM_JOINTS], "zero", false, true)?;
    }
    reach_pose(
        io,
        limits,
        DEFAULT_POSITION,
        "stand",
        !zero_first,
        !zero_first,
    )?;
    let mut last = Instant::now();
    let sensor = retry_feedback(last, || io.read())?;
    let mut previous_velocity = sensor.velocities;
    let mut previous_action = [0.; 14];
    let mut last_target = DEFAULT_POSITION;
    let start = Instant::now();
    let end = duration_s.map(|seconds| start + Duration::from_secs(u64::from(seconds)));
    let mut due = start;
    let mut seq = 0;
    let mut dropped_targets = 0;
    let mut recovered = io.feedback_recovery().recovered;
    let mut falling_since = None;
    let mut telemetry_at = start - Duration::from_secs(1);
    if let Some(remote) = remote {
        remote.lock().unwrap().ready = true;
    }
    while window_open(end, due) {
        std::thread::sleep(due.saturating_duration_since(Instant::now()));
        check_stop(last)?;
        if !window_open(end, Instant::now()) {
            break;
        }
        let begin = Instant::now();
        log(json!({"phase":"rl_read_start","seq":seq,"elapsed_s":start.elapsed().as_secs_f64()}))?;
        let sensor = retry_feedback(last, || io.read())?;
        if io.feedback_recovery().recovered != recovered {
            log(json!({"phase":"rl_feedback_recovered","seq":seq,
                "read_ms":begin.elapsed().as_secs_f64()*1000.,
                "recovery":io.feedback_recovery()}))?;
            recovered = io.feedback_recovery().recovered;
        }
        if !window_open(end, Instant::now()) {
            break;
        }
        let slow = retry_feedback(last, || io.slow_sensors())?;
        if supported_m6 {
            check_running_attitude(&sensor.imu.gravity, &mut falling_since, Instant::now())?;
        } else {
            check_attitude(&sensor.imu.gravity, !zero_first)?;
        }
        valid(&sensor.positions, limits)?;
        let mut running = true;
        if let Some(remote) = remote {
            let state = remote.lock().unwrap();
            if state.exit {
                return Err("explicit relax requested".into());
            }
            let intent = state.command(Instant::now());
            running = intent.is_some();
            command.twist = intent.unwrap_or([0.; 3]);
        }
        if remote.is_some() && telemetry_at.elapsed() >= Duration::from_millis(100) {
            log(
                json!({"phase":"qt_state", "feedback":io.feedback(), "positions":sensor.positions,
                "targets":last_target,"mode":if running {"rl"} else {"hold"},
                "action_scale":action_scale, "command":command.twist}),
            )?;
            telemetry_at = Instant::now();
        }
        if !running {
            // HOLD has no pending writes; fresh feedback is still mandatory.
            last = Instant::now();
            previous_velocity = sensor.velocities;
            due = next_tick(due, Instant::now());
            continue;
        }
        let obs = Observation::build(
            &sensor.imu,
            &sensor.positions,
            &previous_velocity,
            &DEFAULT_POSITION,
            &previous_action,
            &command,
        );
        let action = policy.infer(&obs, Net::Walk)?;
        let applied_action = scaled_action(action, previous_action, action_scale);
        let delta = Observation::scatter_action(&applied_action);
        let target = std::array::from_fn(|i| DEFAULT_POSITION[i] + delta[i]);
        log(
            json!({"phase":"rl_proposal","seq":seq,"observation":obs.as_slice(),"action":action,"applied_action":applied_action,"action_scale":action_scale,"targets":target,
                "mean_supply_v":slow.volts,"temperatures_c":slow.temps_c}),
        )?;
        valid(&target, limits)?;
        // Reject, do not clip a discontinuous policy into a claimed success.
        if target
            .iter()
            .zip(last_target)
            .any(|(a, b)| (a - b).abs() > 0.12)
        {
            return Err("policy step exceeds supported-test 6 rad/s bound; not sent".into());
        }
        check_stop(last)?;
        if !window_open(end, Instant::now()) {
            break;
        }
        match io.write(&JointTargets::new(target)) {
            Ok(()) => {}
            Err(IoError::StaleTarget | IoError::BusBusy | IoError::IncompleteFeedback(_)) => {
                dropped_targets += 1;
                log(
                    json!({"phase":"rl_target_discarded","seq":seq,"written":false,
                    "elapsed_s":start.elapsed().as_secs_f64(),"last_write_age_ms":last.elapsed().as_secs_f64()*1000.,
                    "feedback":io.feedback_stats()}),
                )?;
                // Recompute, never resend the old proposal or advance action history.
                check_stop(last)?;
                due = retry_tick(last, Instant::now());
                continue;
            }
            Err(error) => return Err(error.into()),
        }
        last = Instant::now();
        previous_velocity = sensor.velocities;
        previous_action = applied_action;
        last_target = target;
        log(
            json!({"phase":"rl_sent","seq":seq,"positions":sensor.positions,
                "velocities":sensor.velocities,"currents_ma":sensor.currents_ma,
                "gravity":sensor.imu.gravity,"targets":target,"elapsed_s":start.elapsed().as_secs_f64(),
                "feedback":io.feedback_stats(),
                "compute_ms":begin.elapsed().as_secs_f64()*1000.}),
        )?;
        seq += 1;
        // Skip missed slots; never replay queued goals after a read recovery.
        due = next_tick(due, Instant::now());
    }
    let sensor = retry_feedback(last, || io.read())?;
    valid(&sensor.positions, limits)?;
    let slow = io.slow_sensors()?;
    log(
        json!({"phase":"rl_final_feedback", "positions":sensor.positions,
        "gravity":sensor.imu.gravity,"temperatures_c":slow.temps_c,
        "mean_supply_v":slow.volts,"elapsed_s":start.elapsed().as_secs_f64(),
        "rl_writes":seq,"discarded_targets":dropped_targets,"recovery":io.feedback_recovery()}),
    )?;
    Ok(())
}

fn observe_hold(io: &mut FeetechIo) -> Result<()> {
    // No new goal or power command after a failure, even if feedback recovers.
    for seq in 0..100 {
        let begin = Instant::now();
        match io.read() {
            Ok(sensor) => {
                let slow = io.slow_sensors()?;
                log(
                    json!({"phase":"hold_sample","seq":seq,"positions":sensor.positions,
                    "velocities":sensor.velocities,"currents_ma":sensor.currents_ma,
                    "gravity":sensor.imu.gravity,"mean_supply_v":slow.volts,
                    "temperatures_c":slow.temps_c,"motion_writes":0}),
                )?;
            }
            Err(error) => log(
                json!({"phase":"hold_error","seq":seq,"error":error.to_string(),"motion_writes":0}),
            )?,
        }
        std::thread::sleep(Duration::from_millis(100).saturating_sub(begin.elapsed()));
    }
    Ok(())
}

fn finish_output(io: &mut impl RobotIo, hold: bool, limits: &Limits) -> Result<bool> {
    if hold {
        // A normal stop preserves the existing target; no new goal or enable write.
        let sample = io.read()?;
        valid(&sample.positions, limits)?;
        return Ok(false);
    }
    io.set_torque(false)?;
    Ok(true)
}

fn duration_arg(value: &str) -> Result<u32> {
    let value: u32 = value.parse()?;
    if !(1..=120).contains(&value) {
        return Err("supported diagnostic duration must be 1..120 seconds".into());
    }
    Ok(value)
}

fn rl_duration(value: &str, supported_m6: bool) -> Result<Option<u32>> {
    if supported_m6 && value == "continuous" {
        return Ok(None);
    }
    let seconds = duration_arg(value)?;
    if supported_m6 && seconds > 30 {
        return Err(
            "finite M6 trial is limited to 30 seconds; use continuous for an attended run".into(),
        );
    }
    Ok(Some(seconds))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Trajectory {
    schema: u32,
    source: String,
    joint_names: Vec<String>,
    period_ms: u64,
    frames: Vec<[f64; NUM_JOINTS]>,
}

fn validate_trajectory(trajectory: &Trajectory, limits: &Limits) -> Result<()> {
    if trajectory.schema != 1
        || trajectory.source != "supported_identification_reference"
        || trajectory.period_ms != 20
        || !(2..=500).contains(&trajectory.frames.len())
        || trajectory.joint_names != limits.joint_names
    {
        return Err("invalid replay contract".into());
    }
    let mut previous = DEFAULT_POSITION;
    for frame in &trajectory.frames {
        valid(frame, limits)?;
        if frame
            .iter()
            .zip(previous)
            .any(|(a, b)| (a - b).abs() > 0.04)
        {
            return Err("replay reference exceeds 2 rad/s".into());
        }
        previous = *frame;
    }
    for frame in [
        trajectory.frames.first().unwrap(),
        trajectory.frames.last().unwrap(),
    ] {
        if frame
            .iter()
            .zip(DEFAULT_POSITION)
            .any(|(a, b)| (a - b).abs() > 1e-6)
        {
            return Err("replay must start and end at stand".into());
        }
    }
    Ok(())
}

fn replay(io: &mut FeetechIo, trajectory: &Trajectory, limits: &Limits) -> Result<()> {
    reach_pose(io, limits, DEFAULT_POSITION, "stand", true, true)?;
    let start = Instant::now();
    let mut last = start;
    let period = Duration::from_millis(trajectory.period_ms);
    let mut previous = DEFAULT_POSITION;
    for (seq, target) in trajectory.frames.iter().enumerate() {
        let deadline = start + period * seq as u32;
        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        check_stop(last)?;
        let read_start = start.elapsed().as_secs_f64();
        let sensor = io.read()?;
        let read_end = start.elapsed().as_secs_f64();
        let slow = io.slow_sensors()?;
        if sensor.imu.gravity[2] > -0.94 {
            return Err("trunk tilted during supported replay".into());
        }
        if sensor
            .positions
            .iter()
            .zip(previous)
            .any(|(a, b)| (a - b).abs() > 0.15)
        {
            return Err(
                "replay feedback differs from previous target by more than 0.15 rad".into(),
            );
        }
        if Instant::now().saturating_duration_since(deadline) > Duration::from_millis(40) {
            return Err("replay deadline missed; do not burst queued targets".into());
        }
        io.write(&JointTargets::new(*target))?;
        let write_end = start.elapsed().as_secs_f64();
        last = Instant::now();
        log(
            json!({"phase":"replay", "seq":seq, "scheduled_s":seq as f64*0.02,
            "read_start_s":read_start, "read_end_s":read_end, "write_end_s":write_end,
            "positions":sensor.positions, "velocities":sensor.velocities,
            "currents_ma":sensor.currents_ma, "gyro":sensor.imu.gyro,
            "mean_supply_v":slow.volts, "temperatures_c":slow.temps_c,
            "gravity":sensor.imu.gravity, "quat_wxyz":sensor.imu.quat,
            "previous_target":previous, "target":target, "written":true}),
        )?;
        previous = *target;
    }
    // Record a full second of return-to-stand settling, including final feedback.
    for seq in 0..50 {
        check_stop(last)?;
        let sensor = io.read()?;
        if sensor.imu.gravity[2] > -0.94 {
            return Err("trunk tilted while settling".into());
        }
        io.write(&JointTargets::new(DEFAULT_POSITION))?;
        last = Instant::now();
        log(
            json!({"phase":"settle", "seq":seq, "positions":sensor.positions,
            "currents_ma":sensor.currents_ma, "target":DEFAULT_POSITION}),
        )?;
        std::thread::sleep(period);
    }
    let final_sensor = io.read()?;
    if final_sensor
        .positions
        .iter()
        .zip(DEFAULT_POSITION)
        .any(|(a, b)| (a - b).abs() > 0.07)
    {
        return Err("replay return-to-stand feedback not reached".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    let qt_control = if let Some(i) = args.iter().position(|v| v == "--qt-control") {
        args.remove(i);
        true
    } else {
        false
    };
    let explicit_scale = if let Some(i) = args.iter().position(|v| v == "--action-scale") {
        if i + 1 >= args.len() {
            return Err("--action-scale requires a value".into());
        }
        let scale = args.remove(i + 1).parse::<f32>()?;
        args.remove(i);
        if !scale.is_finite() || scale <= 0. || scale > 1. {
            return Err("action scale must be finite in (0,1]".into());
        }
        Some(scale)
    } else {
        None
    };
    let supported_m6 = args.get(4).is_some_and(|s| s == "--supported-m6");
    if !supported_m6 && (qt_control || explicit_scale.is_some()) {
        return Err("Qt/scale options require supported M6 mode".into());
    }
    let action_scale = explicit_scale.unwrap_or(1.0);
    let keep_enabled = args
        .get(4)
        .is_some_and(|s| s == "--supported-zero-stand-rl-hold");
    if (!keep_enabled && !supported_m6 && args.len() != 5)
        || ((keep_enabled || supported_m6) && args.len() != 6)
        || ![
            "--supported",
            "--supported-replay",
            "--supported-zero-stand-rl",
            "--supported-zero-stand-rl-hold",
            "--supported-m6",
        ]
        .contains(&args[4].as_str())
    {
        return Err("usage: feetech_rl_bench VERIFIED_CONFIG POLICY.onnx|TRAJECTORY.json MODEL_LIMITS.json --supported|--supported-replay|--supported-zero-stand-rl|--supported-zero-stand-rl-hold SECONDS|--supported-m6 SECONDS_OR_continuous".into());
    }
    let duration_s = if keep_enabled || supported_m6 {
        rl_duration(&args[5], supported_m6)?
    } else {
        Some(3)
    };
    if supported_m6 {
        let cfg: duck_control::feetech::Config = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        if !cfg.allow_motion || cfg.servo_gain_profile.is_none() {
            return Err(
                "M6 supported trial requires explicit motion and Reference P6/D20 profile".into(),
            );
        }
    }
    // Native motion gating still requires verified joint/IMU installation.
    let limits: Limits = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    if limits
        .joint_names
        .iter()
        .map(String::as_str)
        .ne(duck_control::model::JOINT_NAMES)
        || limits
            .ranges
            .iter()
            .any(|r| !r[0].is_finite() || !r[1].is_finite() || r[0] >= r[1])
    {
        return Err("invalid model limit contract".into());
    }
    if args[4] == "--supported-replay" {
        let trajectory: Trajectory = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        validate_trajectory(&trajectory, &limits)?;
        log(
            json!({"phase":"replay_preflight", "frames":trajectory.frames.len(),
            "source":trajectory.source, "joint_names":trajectory.joint_names,
            "period_ms":trajectory.period_ms, "closed_loop_rl":false,
            "host_wall_unix_s":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64(),
            "timestamp_semantics":"board monotonic software read/write boundaries; not sensor timestamps",
            "current_semantics":"unsigned magnitude mA; not calibrated torque"}),
        )?;
        unsafe {
            libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
            libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
        }
        let mut io = FeetechIo::open(&args[1])?;
        let result = replay(&mut io, &trajectory, &limits);
        let shutdown = io.set_torque(false);
        log(json!({"phase":"summary", "success":result.is_ok(),
            "error":result.as_ref().err().map(ToString::to_string),
            "torque_off_verified":shutdown.is_ok(),
            "shutdown_error":shutdown.as_ref().err().map(ToString::to_string),
            "closed_loop_rl":false}))?;
        shutdown?;
        return result;
    }
    let paths = PolicyPaths {
        walk: args[2].clone().into(),
        ..Default::default()
    };
    let mut policy = if supported_m6 {
        Policy::load_supported_m6(&paths, 0.05)?
    } else {
        Policy::load(&paths, 0.05)?
    };
    // Reject an incompatible nominal stance before opening any hardware.
    let nominal_imu = duck_control::imu::ImuData {
        gyro: [0.; 3],
        gravity: [0., 0., -1.],
        quat: [1., 0., 0., 0.],
    };
    let nominal = Observation::build(
        &nominal_imu,
        &DEFAULT_POSITION,
        &[0.; NUM_JOINTS],
        &DEFAULT_POSITION,
        &[0.; 14],
        &Command::default(),
    );
    let action = policy.infer(&nominal, Net::Walk)?;
    let delta = Observation::scatter_action(&action);
    let target = std::array::from_fn(|i| DEFAULT_POSITION[i] + delta[i]);
    log(
        json!({"phase":"nominal_preflight","action":action,"targets":target,"hardware_opened":false}),
    )?;
    valid(&target, &limits)?;
    let mut io = FeetechIo::open(&args[1])?;
    if supported_m6 && !io.torque_enabled_ids()?.is_empty() {
        return Err(
            "M6 trial requires all motors initially disabled; no automatic recovery".into(),
        );
    }
    unsafe {
        libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
    }
    // This explicit suspended-bench mode observes attitude without an angle cutoff.
    // It never disables native freshness, installation, travel or servo-alarm checks.
    let zero_first = args[4] == "--supported-zero-stand-rl" || keep_enabled;
    let command = Command {
        twist: if supported_m6 { [0.1, 0., 0.] } else { [0.; 3] },
        ..Default::default()
    };
    log(json!({"phase":"trial_start", "supported_m6":supported_m6,
        "run_mode":if duration_s.is_some() {"finite"} else {"continuous"},
        "action_scale":action_scale,"qt_control":qt_control,
        "requested_rl_seconds":duration_s,"command":command.twist}))?;
    let remote = qt_control.then(|| bench_control::start(&STOP));
    let mut result = exercise(
        &mut io,
        &mut policy,
        &limits,
        zero_first,
        duration_s,
        command,
        supported_m6,
        action_scale,
        remote.as_ref(),
    );
    if keep_enabled {
        if result.is_ok() {
            result = reach_pose(
                &mut io,
                &limits,
                DEFAULT_POSITION,
                "return_stand",
                false,
                false,
            );
        }
        log(json!({"phase":"motion_stopped","success":result.is_ok(),
            "error":result.as_ref().err().map(ToString::to_string),"torque_off_requested":false}))?;
        let monitor = observe_hold(&mut io);
        let enabled = io.torque_enabled_ids();
        log(json!({"phase":"summary","success":result.is_ok(),
            "error":result.as_ref().err().map(ToString::to_string),"torque_off_requested":false,
            "enabled_ids":enabled.as_ref().ok(),"hold_verified":enabled.as_ref().is_ok_and(|ids|ids.len()==NUM_JOINTS),
            "readback_error":enabled.as_ref().err().map(ToString::to_string),
            "monitor_error":monitor.as_ref().err().map(ToString::to_string),"requested_rl_seconds":duration_s,
            "recovery":io.feedback_recovery(),"feedback":io.feedback_stats()}))?;
        monitor?;
        enabled?;
        return result;
    }
    let mut hold = supported_m6 && result.is_ok();
    let mut enabled_ids = Vec::new();
    if hold {
        match io.torque_enabled_ids() {
            Ok(ids) if ids.len() == NUM_JOINTS => enabled_ids = ids,
            Ok(ids) => {
                result = Err(format!("incomplete enable on exit: {ids:?}").into());
                hold = false;
            }
            Err(error) => {
                result = Err(error.into());
                hold = false;
            }
        }
    }
    let mut shutdown = finish_output(&mut io, hold, &limits);
    if hold && shutdown.is_err() {
        result = Err(shutdown.err().unwrap());
        hold = false;
        shutdown = finish_output(&mut io, false, &limits);
    }
    log(
        json!({"phase":"summary","success":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),
        "exit_mode":if hold {"hold"} else {"relax"},
        "torque_off_requested":!hold,"torque_off_verified":shutdown.as_ref().is_ok_and(|sent|*sent),
        "hold_verified":hold && shutdown.is_ok(),"enabled_ids":if hold {Some(enabled_ids)} else {None},
        "shutdown_error":shutdown.as_ref().err().map(ToString::to_string),
        "stop_reason":if result.is_ok() {"duration_completed"} else if STOP.load(Ordering::Relaxed) {"manual_interrupt"} else {"fault"},
        "feedback":io.feedback_stats()}),
    )?;
    shutdown?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unscaled_action_preserves_exact_policy_output() {
        let action = std::array::from_fn(|i| (i as f32 - 7.) * 0.037);
        assert_eq!(scaled_action(action, [0.23; 14], 1.0), action);
    }
    #[test]
    fn scaled_action_preserves_home_and_applied_history_units() {
        let action = [0.1f32; 14];
        let applied = scaled_action(action, [0.; 14], 0.7);
        let delta = Observation::scatter_action(&applied);
        let target: [f64; NUM_JOINTS] = std::array::from_fn(|i| DEFAULT_POSITION[i] + delta[i]);
        assert!((applied[0] - 0.07).abs() < 1e-6);
        assert!((target[0] - DEFAULT_POSITION[0] - 0.07).abs() < 1e-6);
        assert_eq!(target[9], DEFAULT_POSITION[9]);
        let reversed = scaled_action([0.13; 14], [0.23; 14], 0.7);
        assert!((reversed[0] - 0.13).abs() < 1e-6);
    }
    #[test]
    fn busy_retry_does_not_sleep_past_remaining_write_budget() {
        let last_write = Instant::now();
        let now = last_write + Duration::from_millis(97);
        assert_eq!(
            retry_tick(last_write, now),
            last_write + Duration::from_millis(99)
        );
        let now = last_write + Duration::from_micros(99500);
        assert_eq!(
            retry_tick(last_write, now),
            last_write + Duration::from_millis(100)
        );
        // The normal successful-write rate stays 50 Hz, not the retry rate.
        assert_eq!(
            next_tick(last_write, last_write),
            last_write + Duration::from_millis(20)
        );
    }
    #[test]
    fn walking_lean_is_not_a_stand_preparation_failure() {
        let mut falling_since = None;
        let now = Instant::now();
        let leaning = [0.5, 0., -0.866025403784];
        assert!(check_attitude(&leaning, true).is_err());
        assert!(check_running_attitude(&leaning, &mut falling_since, now).is_ok());
        assert!(check_running_attitude(
            &leaning,
            &mut falling_since,
            now + Duration::from_secs(60)
        )
        .is_ok());
        assert!(falling_since.is_none());
    }

    #[test]
    fn running_fall_is_debounced_and_resets_after_recovery() {
        let mut falling_since = None;
        let now = Instant::now();
        let tilted = [0.9, 0., -0.435889894354];
        assert!(check_running_attitude(&tilted, &mut falling_since, now).is_ok());
        assert!(check_running_attitude(
            &tilted,
            &mut falling_since,
            now + Duration::from_millis(199)
        )
        .is_ok());
        assert!(check_running_attitude(
            &tilted,
            &mut falling_since,
            now + Duration::from_millis(200)
        )
        .is_err());
        assert!(check_running_attitude(
            &[0., 0., -1.],
            &mut falling_since,
            now + Duration::from_millis(210)
        )
        .is_ok());
        assert!(falling_since.is_none());
        assert!(check_running_attitude(
            &tilted,
            &mut falling_since,
            now + Duration::from_millis(220)
        )
        .is_ok());
        assert!(check_running_attitude(&[f64::NAN, 0., -1.], &mut falling_since, now).is_err());
    }
    #[test]
    fn continuous_has_no_time_limit_but_finite_mode_expires() {
        assert_eq!(rl_duration("continuous", true).unwrap(), None);
        assert!(rl_duration("continuous", false).is_err());
        assert_eq!(rl_duration("30", true).unwrap(), Some(30));
        assert!(rl_duration("0", true).is_err());
        assert!(rl_duration("31", true).is_err());
        let now = Instant::now();
        assert!(window_open(None, now + Duration::from_secs(86400)));
        assert!(!window_open(Some(now), now));
        assert!(window_open(Some(now + Duration::from_secs(1)), now));
    }

    #[test]
    fn transient_feedback_recovers_without_motor_writes() {
        let mut attempts = 0;
        let mut io = duck_control::io::FakeIo::at(DEFAULT_POSITION);
        let sample = retry_feedback(Instant::now(), || {
            attempts += 1;
            if attempts < 3 {
                Err(IoError::IncompleteFeedback("test missing frame".into()))
            } else {
                io.read()
            }
        })
        .unwrap();
        assert_eq!(sample.positions, DEFAULT_POSITION);
        assert_eq!(attempts, 3);
        assert_eq!(io.writes, 0);
        assert_eq!(io.torque_writes, 0);
    }

    #[test]
    fn feedback_loss_is_bounded_and_faults_are_not_retried() {
        let mut attempts = 0;
        let result: Result<()> = retry_feedback(Instant::now(), || {
            attempts += 1;
            Err(IoError::IncompleteFeedback("test disconnected".into()))
        });
        assert!(result.unwrap_err().to_string().contains("deadline"));
        assert!(attempts > 0);
        attempts = 0;
        let result: Result<()> = retry_feedback(Instant::now(), || {
            attempts += 1;
            Err(IoError::Bus("test device alarm".into()))
        });
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("test device alarm"));
        assert_eq!(attempts, 1);
    }
    #[test]
    fn normal_hold_does_not_relax_reenable_or_resend_a_goal() {
        let limits = Limits {
            joint_names: duck_control::model::JOINT_NAMES
                .iter()
                .map(|n| n.to_string())
                .collect(),
            ranges: [[-3., 3.]; NUM_JOINTS],
        };
        let mut io = duck_control::io::FakeIo::at(DEFAULT_POSITION);
        io.set_torque(true).unwrap();
        assert!(!finish_output(&mut io, true, &limits).unwrap());
        assert_eq!(io.torque, Some(true));
        assert_eq!(io.torque_writes, 1);
        assert_eq!(io.writes, 0);
    }

    #[test]
    fn failed_hold_validation_remains_a_failure_and_can_relax() {
        let limits = Limits {
            joint_names: vec![],
            ranges: [[-3., 3.]; NUM_JOINTS],
        };
        let mut io = duck_control::io::FakeIo::at(DEFAULT_POSITION);
        io.set_torque(true).unwrap();
        io.fail_next_read = true;
        assert!(finish_output(&mut io, true, &limits).is_err());
        assert!(finish_output(&mut io, false, &limits).unwrap());
        assert_eq!(io.torque, Some(false));
        assert_eq!(io.torque_writes, 2);
        assert_eq!(io.writes, 0);
    }

    #[test]
    fn delayed_read_does_not_burst_catchup_commands() {
        let due = Instant::now();
        assert_eq!(
            next_tick(due, due + Duration::from_millis(8)),
            due + Duration::from_millis(20)
        );
        assert_eq!(
            next_tick(due, due + Duration::from_millis(43)),
            due + Duration::from_millis(63)
        );
    }

    #[test]
    fn diagnostic_window_is_finite() {
        assert_eq!(duration_arg("30").unwrap(), 30);
        for value in ["0", "121", "-1", "nan", "1.5"] {
            assert!(duration_arg(value).is_err());
        }
    }
    #[test]
    fn suspended_mode_only_omits_tilt_cutoff() {
        let tilted = [0., 0.8, -0.6];
        assert!(check_attitude(&tilted, false).is_ok());
        assert!(check_attitude(&tilted, true).is_err());
        assert!(check_attitude(&[0., f64::NAN, -1.], false).is_err());
        assert!(check_attitude(&[0., 0., -1.], true).is_ok());
    }
    #[test]
    fn all_targets_are_validated_before_io() {
        let limits = Limits {
            joint_names: duck_control::model::JOINT_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ranges: [[-1., 1.]; NUM_JOINTS],
        };
        assert!(valid(&[0.; NUM_JOINTS], &limits).is_ok());
        for q in [f64::NAN, f64::INFINITY, -1.001, 1.001] {
            let mut target = [0.; NUM_JOINTS];
            target[14] = q;
            assert!(valid(&target, &limits).is_err());
        }
    }
    #[test]
    fn missed_deadline_does_not_continue_control() {
        assert!(check_stop(Instant::now()).is_ok());
        assert!(check_stop(Instant::now() - Duration::from_millis(101)).is_err());
    }

    #[test]
    fn replay_rejects_invalid_contract_ranges_and_discontinuities() {
        let names: Vec<String> = duck_control::model::JOINT_NAMES
            .iter()
            .map(|s| s.to_string())
            .collect();
        let limits = Limits {
            joint_names: names.clone(),
            ranges: [[-1., 1.]; NUM_JOINTS],
        };
        let mut trajectory = Trajectory {
            schema: 1,
            source: "supported_identification_reference".into(),
            joint_names: names,
            period_ms: 20,
            frames: vec![DEFAULT_POSITION; 3],
        };
        assert!(validate_trajectory(&trajectory, &limits).is_ok());
        for value in [f64::NAN, 1.01, 0.1] {
            trajectory.frames[1][0] = value;
            assert!(validate_trajectory(&trajectory, &limits).is_err());
        }
        trajectory.frames = vec![DEFAULT_POSITION; 3];
        trajectory.frames[2][0] = 0.001;
        assert!(validate_trajectory(&trajectory, &limits).is_err());
        trajectory.frames = vec![DEFAULT_POSITION; 3];
        trajectory.period_ms = 0;
        assert!(validate_trajectory(&trajectory, &limits).is_err());
        trajectory.period_ms = 20;
        trajectory.joint_names.swap(0, 1);
        assert!(validate_trajectory(&trajectory, &limits).is_err());
    }
}
