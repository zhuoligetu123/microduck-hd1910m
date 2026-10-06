//! HD1910M board-local adapter. The Python bridge owns FT-SCS and BNO08X.
//! No Dynamixel register/gain conversion is assumed. Only non-RL acceptance is enabled.
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};
use serde_json::{Value, json};
use crate::io::{IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::imu::ImuData;

pub struct Hd1910Io {
    stream: BufReader<UnixStream>,
    sequence: u64,
    pending: Vec<u8>,
}

fn error(e: impl std::fmt::Display) -> IoError { IoError::Bus(format!("HD1910: {e}")) }

fn array<const N: usize>(value: &Value) -> Result<[f64; N]> {
    let values = value.as_array().ok_or_else(|| error("missing numeric array"))?;
    if values.len() != N { return Err(error("wrong array length")); }
    let mut output = [0.0; N];
    for (out, value) in output.iter_mut().zip(values) {
        *out = value.as_f64().filter(|x| x.is_finite()).ok_or_else(|| error("nonfinite sample"))?;
    }
    Ok(output)
}

impl Hd1910Io {
    pub fn open(path: &str) -> Result<Self> {
        let stream = UnixStream::connect(path).map_err(error)?;
        stream.set_read_timeout(Some(Duration::from_millis(400))).map_err(error)?;
        stream.set_write_timeout(Some(Duration::from_millis(400))).map_err(error)?;
        Ok(Self { stream: BufReader::new(stream), sequence: 0, pending: Vec::new() })
    }

    fn call(&mut self, operation: &str, data: Value) -> Result<Value> {
        // The bridge individually verifies all 15 enable/disable acknowledgements.
        let timeout_ms = if operation == "torque" { 1500 } else { 400 };
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        self.sequence += 1;
        let request = json!({"seq":self.sequence,"op":operation,"data":data});
        writeln!(self.stream.get_mut(), "{request}").map_err(error)?;
        loop {
            let remaining = deadline.checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero()).ok_or_else(|| error("bridge response timeout"))?;
            self.stream.get_mut().set_read_timeout(Some(remaining)).map_err(error)?;
            // Retain partial frames on timeout. Late replies belong to old requests;
            // discard them within this deadline, never replay a motor command.
            let limit = 65537usize.saturating_sub(self.pending.len());
            Read::by_ref(&mut self.stream).take(limit as u64)
                .read_until(b'\n', &mut self.pending).map_err(error)?;
            if self.pending.len() > 65536 || !self.pending.ends_with(b"\n") {
                return Err(error("invalid bridge frame"));
            }
            let line = std::mem::take(&mut self.pending);
            let reply: Value = serde_json::from_slice(&line).map_err(error)?;
            match reply["seq"].as_u64() {
                Some(seq) if seq < self.sequence => continue,
                Some(seq) if seq == self.sequence => {},
                _ => return Err(error("mismatched reply")),
            }
            if reply["ok"] != true { return Err(error(reply["error"].as_str().unwrap_or("bridge rejected command"))); }
            return Ok(reply["data"].clone());
        }
    }
}

impl RobotIo for Hd1910Io {
    fn read(&mut self) -> Result<Sensors> {
        let v = self.call("read", Value::Null)?;
        Ok(Sensors { positions: array(&v["positions"])?, velocities: array(&v["velocities"])?,
            currents_ma: array(&v["currents_ma"])?, imu: ImuData {
                gyro: array(&v["imu"]["gyro"])?, gravity: array(&v["imu"]["gravity"])?,
                quat: array(&v["imu"]["quat"])? } })
    }
    fn write(&mut self, target: &JointTargets) -> Result<()> {
        self.call("write", json!(target.positions))?; Ok(())
    }
    fn set_gain(&mut self, kp: u16) -> Result<()> {
        self.call("gain", json!(kp))?; Ok(())
    }
    fn set_torque(&mut self, on: bool) -> Result<()> {
        self.call("torque", json!(on))?; Ok(())
    }
    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        let v = self.call("slow", Value::Null)?;
        let temps: [f64; 15] = array(&v["temps_c"])?;
        Ok(SlowSensors { volts:v["volts"].as_f64().filter(|v| v.is_finite()).ok_or_else(|| error("bad voltage"))?,
            temps_c:temps })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_sample_shape() {
        assert_eq!(array::<3>(&json!([1.,2.,3.])).unwrap(), [1.,2.,3.]);
        assert!(array::<3>(&json!([1.,2.])).is_err());
        assert!(array::<3>(&json!([1.,null,3.])).is_err());
    }

    #[test]
    fn delayed_partial_reply_does_not_poison_next_request() {
        let (client, server) = UnixStream::pair().unwrap();
        let worker = std::thread::spawn(move || {
            let mut server = BufReader::new(server);
            let mut line = String::new();
            server.read_line(&mut line).unwrap();
            let first: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(first["op"], "write");
            server.get_mut().write_all(b"{\"seq\":1,").unwrap();
            std::thread::sleep(Duration::from_millis(650));
            server.get_mut().write_all(b"\"ok\":true,\"data\":null}\n").unwrap();
            line.clear();
            server.read_line(&mut line).unwrap();
            let second: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(second["op"], "read");
            assert_eq!(second["seq"], 2);
            server.get_mut().write_all(b"{\"seq\":2,\"ok\":true,\"data\":\"fresh\"}\n").unwrap();
        });
        let mut io = Hd1910Io { stream: BufReader::new(client), sequence: 0, pending: Vec::new() };
        assert!(io.call("write", Value::Null).is_err());
        assert_eq!(io.call("read", Value::Null).unwrap(), json!("fresh"));
        worker.join().unwrap();
    }
}
