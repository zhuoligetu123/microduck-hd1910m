//! Real sensors -> native observation -> ONNX. No enable or target writes.
//! Reference HD1910 policy: raw previous action, scale 1, one-tick velocity delay.
use duck_control::model::DEFAULT_POSITION;
use duck_control::obs::{Command, Observation};
use duck_control::policy::{Net, Policy, PolicyPaths};
use duck_control::{
    RobotIo,
    feetech::{Config, FeetechIo},
};
use serde_json::json;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: feetech_policy_probe READONLY_CONFIG POLICY.onnx SAMPLES".into());
    }
    let cfg: Config = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if cfg.allow_motion {
        return Err("probe requires allow_motion=false".into());
    }
    let count: usize = args[3].parse()?;
    if !(1..=3000).contains(&count) {
        return Err("invalid sample count".into());
    }
    let mut policy = Policy::load(
        &PolicyPaths {
            walk: args[2].clone().into(),
            ..Default::default()
        },
        0.05,
    )?;
    let mut io = FeetechIo::open(&args[1])?;
    let warmup = Instant::now();
    while !io.imu_ready() {
        if warmup.elapsed() > Duration::from_secs(10) {
            return Err(format!("IMU startup: {:?}", io.read()).into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut previous_velocity = None;
    let mut previous_action = [0.; 14];
    let mut maximum_ms = 0f64;
    let started = Instant::now();
    for seq in 0..count {
        let begin = Instant::now();
        let sensors = io.read()?;
        let velocity = previous_velocity.unwrap_or(sensors.velocities);
        previous_velocity = Some(sensors.velocities);
        let obs = Observation::build(
            &sensors.imu,
            &sensors.positions,
            &velocity,
            &DEFAULT_POSITION,
            &previous_action,
            &Command::default(),
        );
        let action = policy.infer(&obs, Net::Walk)?;
        let delta = Observation::scatter_action(&action);
        let targets: [f64; 15] = std::array::from_fn(|i| DEFAULT_POSITION[i] + delta[i]);
        previous_action = action;
        let elapsed_ms = begin.elapsed().as_secs_f64() * 1000.;
        maximum_ms = maximum_ms.max(elapsed_ms);
        println!(
            "{}",
            json!({"seq":seq,"positions":sensors.positions,
            "observation":obs.as_slice(),"action":action,"targets":targets,
            "compute_ms":elapsed_ms,"motion_commands_sent":0})
        );
        std::thread::sleep(Duration::from_millis(20).saturating_sub(begin.elapsed()));
    }
    println!(
        "{}",
        json!({"success":true,"samples":count,"max_compute_ms":maximum_ms,
        "actual_hz":count as f64 / started.elapsed().as_secs_f64(),
        "hardware_tested":true,"rl_inference_tested":true,"motion_commands_sent":0})
    );
    Ok(())
}
