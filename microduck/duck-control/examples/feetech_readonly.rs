//! Bounded real hardware test: no torque, position, gain or EEPROM writes.
use duck_control::{RobotIo, feetech::ThreadedFeetechIo as FeetechIo};
use serde_json::json;
use std::time::{Duration, Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: feetech_readonly CONFIG.json [samples]")?;
    let count = std::env::args()
        .nth(2)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(500);
    if !(1..=10000).contains(&count) {
        return Err("invalid sample count".into());
    }
    let mut io = FeetechIo::open(&path)?;
    println!(
        "{}",
        json!({"phase":"initial_torque_readback",
        "enabled_ids":io.torque_enabled_ids()?,"motion_commands_sent":0})
    );
    let warmup = Instant::now();
    while !io.imu_ready() {
        if warmup.elapsed() > Duration::from_secs(10) {
            return Err(format!("IMU startup timeout; current read: {:?}", io.read()).into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let start = Instant::now();
    for seq in 0..count {
        let begin = Instant::now();
        let s = io.read()?;
        println!(
            "{}",
            json!({"seq":seq,"positions":s.positions,"velocities":s.velocities,
            "currents_ma":s.currents_ma,"imu":{"gyro":s.imu.gyro,"quat":s.imu.quat,"gravity":s.imu.gravity},
            "read_ms":begin.elapsed().as_secs_f64()*1000.,"feedback":io.feedback_stats(),"motion_commands_sent":0})
        );
        std::thread::sleep(Duration::from_millis(20).saturating_sub(begin.elapsed()));
    }
    println!(
        "{}",
        json!({"success":true,"samples":count,"hz":count as f64/start.elapsed().as_secs_f64(),
            "enabled_ids":io.torque_enabled_ids()?,"motion_commands_sent":0})
    );
    Ok(())
}
