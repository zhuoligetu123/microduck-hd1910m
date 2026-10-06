//! Local stdin intents only. Authenticated UDP is handled by native_udp.py.
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Intent {
    pub ready: bool,
    pub exit: bool,
    twist: Option<[f64; 3]>,
    received: Option<Instant>,
}

impl Intent {
    pub fn command(&self, now: Instant) -> Option<[f64; 3]> {
        if !self.ready
            || self.exit
            || self
                .received
                .is_none_or(|t| now.saturating_duration_since(t) >= Duration::from_millis(400))
        {
            None
        } else {
            self.twist
        }
    }

    pub fn apply(&mut self, op: &str, now: Instant) -> Result<(), &'static str> {
        match op {
            "stop" | "disable" | "rl_hold" => self.twist = None,
            "relax" => {
                self.twist = None;
                self.exit = true;
            }
            _ => {
                let twist = match op {
                    "rl_idle" => [0., 0., 0.],
                    "rl_forward" => [0.1, 0., 0.],
                    "rl_backward" => [-0.1, 0., 0.],
                    "rl_left" => [0., 0., 0.4],
                    "rl_right" => [0., 0., -0.4],
                    _ => return Err("unsupported bench operation"),
                };
                if !self.ready || self.exit {
                    return Err("bench is not ready");
                }
                self.twist = Some(twist);
            }
        }
        self.received = Some(now);
        Ok(())
    }
}

pub fn start(stop: &'static std::sync::atomic::AtomicBool) -> Arc<Mutex<Intent>> {
    let shared = Arc::new(Mutex::new(Intent::default()));
    let input = shared.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else {
                break;
            };
            if line.len() > 8192 {
                break;
            }
            let Ok(packet) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let result = input
                .lock()
                .unwrap()
                .apply(packet["op"].as_str().unwrap_or(""), Instant::now());
            if input.lock().unwrap().exit {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let _ = writeln!(
                std::io::stdout().lock(),
                "{}",
                json!({"phase":"control_ack",
                "id":packet["id"],"accepted":result.is_ok(),"reason":result.err()})
            );
        }
        let mut state = input.lock().unwrap();
        state.twist = None;
        state.received = None;
    });
    shared
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readiness_expiry_stop_and_explicit_restart() {
        let mut state = Intent::default();
        let now = Instant::now();
        assert!(state.apply("rl_forward", now).is_err());
        state.ready = true;
        state.apply("rl_forward", now).unwrap();
        assert_eq!(state.command(now), Some([0.1, 0., 0.]));
        assert_eq!(state.command(now + Duration::from_millis(400)), None);
        state.apply("rl_left", now).unwrap();
        assert_eq!(state.command(now), Some([0., 0., 0.4]));
        state.apply("stop", now).unwrap();
        assert_eq!(state.command(now), None);
        assert!(state.apply("init", now).is_err());
        state.apply("relax", now).unwrap();
        assert!(state.exit);
        assert!(state.apply("rl_idle", now).is_err());
    }
}
