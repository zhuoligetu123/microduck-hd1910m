//! Native observation -> ONNX parity fixture. No RobotIo, serial, or network access.
use duck_control::imu::ImuData;
use duck_control::model::DEFAULT_POSITION;
use duck_control::obs::{Command, Observation};
use duck_control::policy::{Net, Policy, PolicyPaths};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: policy_offline POLICY.onnx")?;
    let mut policy = Policy::load(&PolicyPaths { walk: path.into(), ..Default::default() }, 0.05)?;
    let imu = ImuData { gyro: [0.;3], gravity: [0.,0.,-1.], quat: [1.,0.,0.,0.] };
    let mut history = [0.;14];
    for seq in 0..250 {
        let twist = [[0.,0.,0.],[0.1,0.,0.],[0.,0.,0.4],[-0.1,0.,0.],[0.,0.,-0.4]][seq/50];
        let obs = Observation::build(&imu, &DEFAULT_POSITION, &[0.;15], &DEFAULT_POSITION,
                                    &history, &Command { twist, ..Default::default() });
        let action = policy.infer(&obs, Net::Walk)?;
        if action.iter().any(|v| !v.is_finite()) { return Err("nonfinite policy output".into()); }
        println!("{}", json!({"seq":seq,"observation":obs.as_slice(),"action":action,
                             "hardware_opened":false,"fixture":"fixed_home_with_action_history"}));
        history = action;
    }
    Ok(())
}
