//! Native attitude monitoring without inventing joint observations or opening a serial port.
use super::*;
use duck_control::Bno08x;
use duck_control::imu::ImuData;
use serde::Deserialize;

fn open(path: &str) -> Result<Bno08x, Box<dyn std::error::Error>> {
    let cfg: duck_control::feetech::Config = serde_json::from_slice(&std::fs::read(path)?)?;
    if cfg.allow_motion { return Err("IMU-only configuration must disable motion".into()); }
    #[derive(Deserialize)]
    struct Mount { imu_mount_wxyz: [f64; 4] }
    let installation = PathBuf::from(path).parent().unwrap().join(cfg.installation);
    let mount: Mount = serde_json::from_slice(&std::fs::read(installation)?)?;
    Ok(Bno08x::open(&cfg.imu_bus, cfg.imu_address, mount.imu_mount_wxyz)?)
}

fn frame(t: f64, imu: ImuData, age: f64, hz: f64) -> proto::RobotState {
    proto::RobotState {
        t,
        movement: proto::MoveState { requested: [0.; 3], applied: [0.; 3], limited_by: vec!["imu_only".into()] },
        head: [0.; 4], policy: "imu_only".into(),
        safety: proto::SafetyState { fallen: false, limp: false, gravity: imu.gravity, gain: None },
        control_loop: proto::LoopState { hz, missed: 0 },
        joints: vec![], targets: vec![], target_written: Some(false), inference: None,
        feedback: Some(serde_json::json!({
            "backend":"bno085_native", "control_valid":false, "imu_valid":true,
            "joint_valid":false, "imu_age_s":age, "joint_age_s":null,
            "imu":{"quat":imu.quat, "gyro":imu.gyro, "gravity":imu.gravity},
            "positions":[], "states":[], "joints":[], "enabled_ids":[],
            "motion_available":false, "error":null, "odom_valid":false
        })),
        odom: proto::OdomState::default(), theremin: None, chorale: None,
    }
}

pub(super) fn publish_pending(imu: &Bno08x, state: &RobotState) {
    let Ok((sample, at)) = imu.snapshot() else { return; };
    let mut message = frame(state.started.elapsed().as_secs_f64(), sample,
        at.elapsed().as_secs_f64(), 0.);
    message.policy = "hardware_wait".into();
    let feedback = message.feedback.as_mut().unwrap();
    feedback["error"] = serde_json::json!(state.startup_error.load_full().as_deref());
    feedback["policy_enabled"] = serde_json::json!(false);
    feedback["imu_transport"] = imu.report_stats().unwrap_or_default();
    // Live attitude is not a control tick or evidence of motor readiness.
    let _ = state.state_tx.send(message);
}

pub(super) async fn run(path: &str, state: Arc<RobotState>, period: Duration) {
    let imu = match open(path) {
        Ok(imu) => imu,
        Err(e) => { tracing::error!(error = %e, "native IMU open failed"); return; }
    };
    let mut timer = tokio::time::interval(period);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut errors = 0u64;
    let mut window = Instant::now();
    let mut samples = 0u64;
    while !state.shutdown.load(Ordering::Relaxed) {
        timer.tick().await;
        match imu.snapshot() {
            Ok((sample, at)) => {
                errors = 0;
                state.imu_ready.store(true, Ordering::Relaxed);
                state.imu_stale_run.store(0, Ordering::Relaxed);
                state.ticks.fetch_add(1, Ordering::Relaxed);
                state.last_tick_us.store(state.started.elapsed().as_micros() as u64, Ordering::Relaxed);
                samples += 1;
                if window.elapsed() >= Duration::from_secs(1) {
                    state.achieved_hz.store((samples as f64 / window.elapsed().as_secs_f64()).to_bits(), Ordering::Relaxed);
                    samples = 0;
                    window = Instant::now();
                }
                let mut message = frame(state.started.elapsed().as_secs_f64(), sample,
                    at.elapsed().as_secs_f64(), f64::from_bits(state.achieved_hz.load(Ordering::Relaxed)));
                if let Ok(stats) = imu.report_stats() {
                    message.feedback.as_mut().unwrap()["imu_transport"] = stats;
                }
                let _ = state.state_tx.send(message);
            }
            Err(e) => {
                errors += 1;
                state.imu_ready.store(false, Ordering::Relaxed);
                state.imu_stale_run.fetch_add(1, Ordering::Relaxed);
                state.imu_stale_blocks.fetch_add(1, Ordering::Relaxed);
                if errors == 1 || errors.is_multiple_of(250) {
                    tracing::warn!(error = %e, "waiting for native IMU; no joint IO");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imu_only_never_invents_joints_or_accepts_motion() {
        let f = frame(1., ImuData::default(), 0.01, 50.);
        assert!(f.joints.is_empty() && f.targets.is_empty());
        assert_eq!(f.target_written, Some(false));
        assert_eq!(f.feedback.as_ref().unwrap()["control_valid"], false);
        let mut params = Params::default();
        params.bus.port = "imu:unused.json".into();
        params.policy.enabled = false;
        let state = RobotState::new(&params, std::path::Path::new("unused"), false, false);
        let intents = Intents::new();
        let response = dispatch(&state, &intents, proto::Id::Number(1), &proto::Call::RobotInit);
        let value = serde_json::to_value(response).unwrap();
        assert_eq!(value["result"]["accepted"], false);
        assert!(!intents.enabled());
    }
}
