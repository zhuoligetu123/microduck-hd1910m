//! The robot control core: everything between reading the bus and writing it.
//!
//! Deliberately not a daemon. There is no tokio here, no socket, no systemd — `robotd`
//! owns all of that. The boundary is enforced by the compiler rather than by discipline,
//! which is what stops process concerns leaking into the code that drives motors.
//!
//! The control path it holds — model, bus, [`io::RobotIo`], observations, policy, safety — is
//! designed in `docs/design/robotd-design.md` §2.

pub mod bus;
pub mod fall;
pub mod imu;
pub mod io;
pub mod model;
#[cfg(unix)]
pub mod hd1910;
#[cfg(target_os = "linux")]
pub mod feetech;
#[cfg(target_os = "linux")]
mod bno08x;
#[cfg(target_os = "linux")]
pub use bno08x::Bno08x;
pub mod obs;
pub mod policy;
pub mod safety;

pub use imu::ImuData;
pub use io::{FakeIo, IoError, JointTargets, RobotIo, Sensors, SlowSensors};
pub mod sim;
pub use model::{
    BATTERY_EMPTY_V, BATTERY_FULL_V, DEFAULT_POSITION, JOINT_IDS, JOINT_NAMES, NUM_JOINTS,
    battery_percent,
};
pub use obs::{ACTION_LEN, Command, OBS_LEN, Observation};
