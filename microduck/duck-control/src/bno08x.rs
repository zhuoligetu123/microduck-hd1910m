//! Native BNO085 I2C SH-2 gyro/game-rotation input. Host receipt timestamps,
//! not synchronized acquisition times. Never returns an invented upright pose.
use crate::imu::{ImuData, mul, rotate, rotate_inverse};
use crate::io::{IoError, Result};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

fn err(e: impl std::fmt::Display) -> IoError {
    IoError::Bus(format!("BNO085: {e}"))
}

#[derive(Default)]
struct Reports {
    reference_filter: bool,
    gyro: Option<([f64; 3], u8, Instant)>,
    quat: Option<([f64; 4], u8, Instant)>,
    gyro_count: u64,
    quat_count: u64,
    rejected_packets: u64,
    reconnects: u64,
    read_errors: u64,
}

fn normalized(mut q: [f64; 4]) -> Result<[f64; 4]> {
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !n.is_finite() || !(0.9..=1.1).contains(&n) {
        return Err(err("invalid quaternion"));
    }
    for v in &mut q {
        *v /= n;
    }
    Ok(q)
}

fn fresh(previous: Option<u8>, next: u8) -> bool {
    previous.is_none_or(|p| (1..128).contains(&next.wrapping_sub(p)))
}

impl Reports {
    fn invalidate(&mut self) {
        self.gyro = None;
        self.quat = None;
    }
    fn receive(&mut self, bytes: &[u8], now: Instant) {
        if let Err(error) = self.parse(bytes, now) {
            self.rejected_packets += 1;
            if self.rejected_packets == 1 || self.rejected_packets % 100 == 0 {
                tracing::warn!(%error, rejected_packets = self.rejected_packets,
                    "discarding malformed IMU report; waiting for fresh data");
            }
        }
    }

    fn parse(&mut self, mut bytes: &[u8], now: Instant) -> Result<()> {
        while !bytes.is_empty() {
            let length = match bytes[0] {
                0x02 => 10,
                0x08 => 12,
                0xfb | 0xfa => 5,
                _ => return Err(err(format!("unexpected report {:02x}", bytes[0]))),
            };
            if bytes.len() < length {
                return Err(err("truncated report"));
            }
            let word = |i| i16::from_le_bytes([bytes[i], bytes[i + 1]]) as f64;
            match bytes[0] {
                2 if bytes[2] & 3 != 0 && fresh(self.gyro.map(|v| v.1), bytes[1]) => {
                    let mut g = [word(4) / 512., word(6) / 512., word(8) / 512.];
                    if self.reference_filter && let Some((old, _, at)) = self.gyro {
                        let weight = 0.5f64.powf(now.duration_since(at).as_secs_f64() / 0.01);
                        for i in 0..3 { g[i] = weight * old[i] + (1.0-weight) * g[i]; }
                    }
                    self.gyro = Some((
                        g,
                        bytes[1],
                        now,
                    ));
                    self.gyro_count += 1;
                }
                8 if bytes[2] & 3 != 0 && fresh(self.quat.map(|v| v.1), bytes[1]) => {
                    let q = normalized([
                        word(10) / 16384.,
                        word(4) / 16384.,
                        word(6) / 16384.,
                        word(8) / 16384.,
                    ])?;
                    self.quat = Some((q, bytes[1], now));
                    self.quat_count += 1;
                }
                _ => {}
            }
            bytes = &bytes[length..];
        }
        Ok(())
    }

    fn sample(&self, mount: [f64; 4], now: Instant) -> Result<ImuData> {
        let (g, _, gt) = self.gyro.ok_or_else(|| err("no gyro yet"))?;
        let (q, _, qt) = self.quat.ok_or_else(|| err("no quaternion yet"))?;
        if now.duration_since(gt) > Duration::from_millis(150)
            || now.duration_since(qt) > Duration::from_millis(150)
            || gt.max(qt).duration_since(gt.min(qt)) > Duration::from_millis(30)
        {
            return Err(err("stale/skewed reports"));
        }
        let trunk = mul(q, [mount[0], -mount[1], -mount[2], -mount[3]]);
        Ok(ImuData {
            gyro: rotate(mount, g),
            quat: trunk,
            gravity: rotate_inverse(trunk, [0., 0., -1.]),
        })
    }
}

#[test]
fn reference_gyro_ema_is_applied_once_per_fresh_report() {
    let now=Instant::now();
    let mut reports=Reports {reference_filter:true,..Default::default()};
    let mut packet=[2,1,3,0,0,2,0,0,0,0];
    reports.parse(&packet,now).unwrap();
    assert_eq!(reports.gyro.unwrap().0,[1.,0.,0.]);
    packet[1]=2;packet[5]=0;
    reports.parse(&packet,now+Duration::from_millis(10)).unwrap();
    assert_eq!(reports.gyro.unwrap().0,[0.5,0.,0.]);
    reports.parse(&packet,now+Duration::from_millis(20)).unwrap();
    assert_eq!(reports.gyro.unwrap().0,[0.5,0.,0.]);
    packet[1]=3;
    reports.parse(&packet,now+Duration::from_millis(30)).unwrap();
    assert_eq!(reports.gyro.unwrap().0,[0.125,0.,0.]);
}

pub struct Bno08x {
    state: Arc<Mutex<(Reports, Option<String>)>>,
    stop: Arc<AtomicBool>,
    reconnect: Arc<AtomicBool>,
    mount: [f64; 4],
    worker: Option<thread::JoinHandle<()>>,
}

impl Bno08x {
    pub fn set_reference_filter(&self, enabled: bool) {
        self.state.lock().unwrap().0.reference_filter = enabled;
    }
    #[cfg(test)]
    pub(crate) fn fixture_unavailable() -> Self {
        let imu = Self::fixture();
        imu.state.lock().unwrap().0.invalidate();
        imu
    }
    #[cfg(test)]
    pub(crate) fn fixture() -> Self {
        let t = Instant::now();
        Self {
            state: Arc::new(Mutex::new((
                Reports {
                    gyro: Some(([0.; 3], 1, t)),
                    quat: Some(([1., 0., 0., 0.], 1, t)),
                    ..Reports::default()
                },
                None,
            ))),
            stop: Arc::new(AtomicBool::new(false)),
            reconnect: Arc::new(AtomicBool::new(false)),
            mount: [1., 0., 0., 0.],
            worker: None,
        }
    }
    pub fn open(path: &str, address: u16, mount: [f64; 4]) -> Result<Self> {
        if ![0x4a, 0x4b].contains(&address) {
            return Err(err("invalid address"));
        }
        let mount = normalized(mount)?;
        let bus = open_bus(path, address)?;
        let state = Arc::new(Mutex::new((Reports::default(), None)));
        let stop = Arc::new(AtomicBool::new(false));
        let reconnect = Arc::new(AtomicBool::new(false));
        let requested = reconnect.clone();
        let shared = state.clone();
        let done = stop.clone();
        let path = path.to_owned();
        let worker = thread::Builder::new()
            .name("bno085".into())
            .spawn(move || {
                supervise(&path, address, Some(bus), &shared, &done, &requested);
            })
            .map_err(err)?;
        Ok(Self {
            state,
            stop,
            reconnect,
            mount,
            worker: Some(worker),
        })
    }

    pub fn read(&self) -> Result<ImuData> {
        Ok(self.snapshot()?.0)
    }

    /// Called only after an explicit reset has verified every servo torque off.
    pub fn reconnect(&self) -> Result<()> {
        let Some(worker) = &self.worker else { return self.read().map(|_| ()); };
        {
            let mut state = self.state.lock().map_err(err)?;
            state.0.invalidate();
            state.1 = Some("manual IMU reconnect pending".into());
        }
        self.reconnect.store(true, Ordering::Release);
        worker.thread().unpark();
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(3) {
            if !self.reconnect.load(Ordering::Acquire) && self.read().is_ok() { return Ok(()); }
            thread::sleep(Duration::from_millis(10));
        }
        Err(IoError::ImuUnavailable("manual reconnect failed; check IMU and retry fault reset".into()))
    }

    /// Counts accepted sensor reports, not control ticks or IPC publications.
    pub fn report_stats(&self) -> Result<serde_json::Value> {
        let state = self.state.lock().map_err(err)?;
        Ok(serde_json::json!({"requested_hz":100,
            "gyro_reports":state.0.gyro_count, "quaternion_reports":state.0.quat_count,
            "rejected_packets":state.0.rejected_packets,
            "reconnects":state.0.reconnects, "last_error":state.1,
            "read_errors":state.0.read_errors,
            "reconnect_mode":"manual",
            "timestamp_source":"host_receipt"}))
    }

    pub fn snapshot(&self) -> Result<(ImuData, Instant)> {
        let state = self.state.lock().map_err(err)?;
        if let Some(error) = &state.1 {
            return Err(IoError::ImuUnavailable(error.clone()));
        }
        // IMU recovery must not become a latched motor alarm in the shared reader.
        let data = state.0.sample(self.mount, Instant::now())
            .map_err(|error| IoError::ImuUnavailable(error.to_string()))?;
        Ok((data, state.0.gyro.unwrap().2.min(state.0.quat.unwrap().2)))
    }
}

impl Drop for Bno08x {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

fn open_bus(path: &str, address: u16) -> Result<File> {
    let bus = OpenOptions::new().read(true).write(true).open(path).map_err(err)?;
    // Shared with the protocol test tools: only one reader may own this device.
    bus.try_lock().map_err(err)?;
    if unsafe { libc::ioctl(bus.as_raw_fd(), 0x0703, address as libc::c_ulong) } < 0 {
        return Err(err(std::io::Error::last_os_error()));
    }
    Ok(bus)
}

fn supervise(path: &str, address: u16, mut bus: Option<File>,
    state: &Mutex<(Reports, Option<String>)>, stop: &AtomicBool, reconnect: &AtomicBool) {
    while !stop.load(Ordering::Relaxed) {
        let result = bus.take().map(Ok).unwrap_or_else(|| open_bus(path, address))
            .and_then(|bus| run(bus, state, stop, reconnect));
        if stop.load(Ordering::Relaxed) { break; }
        if let Err(error) = result {
            let mut shared = state.lock().unwrap();
            shared.0.invalidate();
            shared.1 = Some(error.to_string());
            tracing::warn!(%error, "IMU disconnected; waiting for explicit torque-off fault reset");
        }
        // Stay alive without polling or resetting the sensor during motion.
        while !stop.load(Ordering::Relaxed) && !reconnect.swap(false, Ordering::AcqRel) {
            thread::park();
        }
        if stop.load(Ordering::Relaxed) { break; }
        let mut shared = state.lock().unwrap();
        shared.0.invalidate();
        shared.0.reconnects += 1;
        shared.1 = Some("manual IMU reconnect pending".into());
    }
}

// I2C header probing consumes a transfer, not the cargo. CEVA's rxAssemble
// permits the remaining cargo on the same channel with a consecutive sequence.
// Some transports repeat the header unchanged instead. Never mix other cargo.
fn read_packet(bus: &mut impl Read) -> Result<Option<(u8, Vec<u8>)>> {
    let mut header = [0u8; 4];
    if bus.read(&mut header).map_err(err)? != 4 {
        return Err(err("short I2C header"));
    }
    let raw = u16::from_le_bytes([header[0], header[1]]);
    let length = usize::from(raw & 0x7fff);
    if raw == 0xffff || length == 0 || length == 4 {
        return Ok(None);
    }
    if !(4..=4096).contains(&length) || header[2] >= 8 {
        return Err(err("invalid SHTP length/channel"));
    }
    let channel = header[2];
    let mut remaining = length - 4;
    let mut payload = Vec::with_capacity(remaining);
    for _ in 0..16 {
        let mut packet = vec![0; remaining + 4];
        let received = bus.read(&mut packet).map_err(err)?;
        if received < 4 {
            return Err(err("short SHTP fragment"));
        }
        let next = u16::from_le_bytes([packet[0], packet[1]]);
        let repeat = packet[..4] == header;
        let continuation = next & 0x8000 != 0 && packet[3] == header[3].wrapping_add(1);
        if packet[2] != channel
            || usize::from(next & 0x7fff) != remaining + 4
            || (!repeat && !continuation)
            || received == 4
        {
            return Err(err("interrupted/invalid SHTP continuation"));
        }
        payload.extend_from_slice(&packet[4..received]);
        remaining -= received - 4;
        if remaining == 0 {
            return Ok(Some((channel, payload)));
        }
        header.copy_from_slice(&packet[..4]);
    }
    Err(err("too many SHTP fragments"))
}

fn run(mut bus: File, state: &Mutex<(Reports, Option<String>)>, stop: &AtomicBool, reconnect: &AtomicBool) -> Result<()> {
    // Soft reset discards reports from a previous owner. No calibration/NVM write.
    if bus.write(&[5, 0, 1, 0, 1]).map_err(err)? != 5 {
        return Err(err("short reset write"));
    }
    thread::sleep(Duration::from_millis(300));
    let boot = Instant::now();
    while boot.elapsed() < Duration::from_millis(700) {
        if stop.load(Ordering::Relaxed) || reconnect.load(Ordering::Acquire) {
            return Ok(());
        }
        let _ = read_packet(&mut bus)?;
        thread::sleep(Duration::from_millis(1));
    }
    // Volatile set-feature requests only.
    for (sequence, id) in [2, 8].into_iter().enumerate() {
        let mut packet = [0u8; 21];
        packet[..6].copy_from_slice(&[21, 0, 2, sequence as u8, 0xfd, id]);
        packet[9..13].copy_from_slice(&10_000u32.to_le_bytes());
        if bus.write(&packet).map_err(err)? != packet.len() {
            return Err(err("short feature write"));
        }
    }
    let mut last_valid = Instant::now();
    let mut read_failure_since: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) && !reconnect.load(Ordering::Acquire) {
        if last_valid.elapsed() > Duration::from_secs(2) {
            return Err(err("no fresh gyro/quaternion; re-subscribe required"));
        }
        let packet = match read_packet(&mut bus) {
            Ok(packet) => {
                read_failure_since = None;
                packet
            }
            Err(error) => {
                state.lock().map_err(err)?.0.read_errors += 1;
                let since = read_failure_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= Duration::from_millis(250) { return Err(error); }
                // A transient NACK need not reset the sensor and lose its feature reports.
                // snapshot() still rejects data older than 150 ms during this retry.
                thread::sleep(Duration::from_millis(5));
                continue;
            }
        };
        let Some((channel, payload)) = packet else {
            thread::sleep(Duration::from_millis(1));
            continue;
        };
        match channel {
            3 => {
                let mut shared = state.lock().map_err(err)?;
                let now = Instant::now();
                shared.0.receive(&payload, now);
                if shared.0.sample([1., 0., 0., 0.], now).is_ok() {
                    last_valid = now;
                    if shared.1.take().is_some() {
                        tracing::info!("IMU reconnected; fresh gyro and quaternion received");
                    }
                }
            }
            1 => {
                return Err(err("sensor reset; re-subscribe required"));
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_imu_is_recoverable_feedback_not_a_motor_fault() {
        let imu = Bno08x::fixture();
        imu.state.lock().unwrap().1 = Some("I2C NACK".into());
        assert!(matches!(imu.snapshot(), Err(IoError::ImuUnavailable(_))));
        {
            let mut state = imu.state.lock().unwrap();
            state.1 = None;
            state.0.invalidate();
        }
        assert!(matches!(imu.snapshot(), Err(IoError::ImuUnavailable(_))));
        imu.state.lock().unwrap().0.receive(&[2,0,3,0,0,0,0,0,0,0,8,0,3,0,0,0,0,0,0,0,0,64], Instant::now());
        assert!(imu.snapshot().is_ok());
    }
    #[test]
    fn missing_bus_waits_until_explicit_reconnect() {
        let state = Arc::new(Mutex::new((Reports::default(), None)));
        let stop = Arc::new(AtomicBool::new(false));
        let shared = state.clone();
        let done = stop.clone();
        let reconnect = Arc::new(AtomicBool::new(false));
        let requested = reconnect.clone();
        let worker = thread::spawn(move || supervise("/no-such-microduck-i2c", 0x4b, None, &shared, &done, &requested));
        thread::sleep(Duration::from_millis(1100));
        let automatic = state.lock().unwrap().0.reconnects;
        reconnect.store(true, Ordering::Release);
        worker.thread().unpark();
        let deadline = Instant::now();
        while state.lock().unwrap().0.reconnects < 1 && deadline.elapsed() < Duration::from_secs(3) {
            thread::sleep(Duration::from_millis(20));
        }
        let attempts = state.lock().unwrap().0.reconnects;
        let alive = !worker.is_finished();
        stop.store(true, Ordering::Relaxed);
        worker.thread().unpark();
        worker.join().unwrap();
        assert_eq!(automatic, 0);
        assert_eq!(attempts, 1);
        assert!(alive, "I2C failures must not terminate the worker");
    }
    #[test]
    fn reconnect_discards_old_samples_and_accepts_reset_sequence() {
        let now = Instant::now();
        let mut reports = Reports {gyro: Some(([0.;3], 100, now)),
            quat: Some(([1.,0.,0.,0.],100,now)),..Reports::default()};
        reports.invalidate();
        assert!(reports.sample([1.,0.,0.,0.],now).is_err());
        reports.receive(&[2,0,3,0,0,0,0,0,0,0],now);
        assert!(reports.sample([1.,0.,0.,0.],now).is_err());
        reports.receive(&[8,0,3,0,0,0,0,0,0,0,0,64],now);
        assert!(reports.sample([1.,0.,0.,0.],now).is_ok());
    }
    #[test]
    fn malformed_report_does_not_kill_reception_or_refresh_quaternion() {
        let t = Instant::now();
        let mut reports = Reports::default();
        let valid = [2, 1, 3, 0, 0, 0, 0, 0, 0, 0,
            8, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 64];
        reports.receive(&valid, t);
        let later = t + Duration::from_millis(200);
        reports.receive(&[8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], later);
        assert_eq!(reports.rejected_packets, 1);
        assert_eq!(reports.quat_count, 1);
        assert!(reports.sample([1., 0., 0., 0.], later).is_err());
        let mut recovered = valid;
        recovered[1] = 3;
        recovered[11] = 3;
        reports.receive(&recovered, later);
        assert_eq!(reports.quat_count, 2);
        assert!(reports.sample([1., 0., 0., 0.], later).is_ok());
    }
    struct Transfers(std::collections::VecDeque<Vec<u8>>);
    impl Read for Transfers {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let bytes = self.0.pop_front().expect("unexpected extra bus read");
            assert!(bytes.len() <= buf.len());
            buf[..bytes.len()].copy_from_slice(&bytes);
            Ok(bytes.len())
        }
    }
    fn packets(rows: &[&[u8]]) -> Transfers {
        Transfers(rows.iter().map(|v| v.to_vec()).collect())
    }
    #[test]
    fn i2c_probe_accepts_repeat_and_consecutive_continuation() {
        for header in [[8, 0, 3, 255], [8, 128, 3, 0]] {
            let mut full = header.to_vec();
            full.extend([1, 2, 3, 4]);
            let mut bus = packets(&[&[8, 0, 3, 255], &full]);
            assert_eq!(read_packet(&mut bus).unwrap(), Some((3, vec![1, 2, 3, 4])));
        }
    }
    #[test]
    fn shtp_fragment_lengths_and_sequence_are_checked() {
        let mut bus = packets(&[
            &[10, 0, 3, 1],
            &[10, 128, 3, 2, 1, 2],
            &[8, 128, 3, 3, 3, 4, 5, 6],
        ]);
        assert_eq!(
            read_packet(&mut bus).unwrap(),
            Some((3, vec![1, 2, 3, 4, 5, 6]))
        );
        for bad in [
            vec![8, 128, 3, 5, 1, 2, 3, 4],
            vec![8, 128, 4, 2, 1, 2, 3, 4],
            vec![7, 128, 3, 2, 1, 2, 3],
            vec![8, 128, 3, 2],
        ] {
            assert!(read_packet(&mut packets(&[&[8, 0, 3, 1], &bad])).is_err());
        }
        assert!(read_packet(&mut packets(&[&[1, 32, 3, 1]])).is_err());
        assert!(read_packet(&mut packets(&[&[8, 0, 9, 1]])).is_err());
        assert!(read_packet(&mut packets(&[&[1, 0]])).is_err());
        assert!(read_packet(&mut packets(&[&[3, 0, 3, 1]])).is_err());
        for header in [[0, 0, 0, 0], [255, 255, 255, 255], [4, 0, 3, 1]] {
            assert_eq!(read_packet(&mut packets(&[&header])).unwrap(), None);
        }
    }
    #[test]
    fn units_age_rotation_and_duplicate_sequences() {
        let t = Instant::now();
        let mut r = Reports::default();
        r.parse(
            &[
                2, 255, 3, 0, 0, 2, 0, 0, 0, 0, 8, 255, 3, 0, 0, 0, 0, 0, 0, 0, 0, 64,
            ],
            t,
        )
        .unwrap();
        let s = r.sample([1., 0., 0., 0.], t).unwrap();
        assert_eq!(s.gyro, [1., 0., 0.]);
        assert_eq!(s.gravity, [0., 0., -1.]);
        assert!(
            r.sample([1., 0., 0., 0.], t + Duration::from_millis(151))
                .is_err()
        );
        assert!(!fresh(Some(255), 255));
        assert!(fresh(Some(255), 0));
        assert!(!fresh(Some(3), 2));
        assert!(r.parse(&[8, 1, 3], t).is_err());
        assert!(normalized([0.; 4]).is_err());
    }
    #[test]
    fn unreliable_reports_do_not_refresh_valid_samples() {
        let t = Instant::now();
        let mut r = Reports {
            gyro: Some(([0.; 3], 1, t)),
            quat: Some(([1., 0., 0., 0.], 1, t)),
            ..Reports::default()
        };
        let stale = t + Duration::from_millis(151);
        r.parse(&[2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
            8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64], stale).unwrap();
        assert!(r.sample([1., 0., 0., 0.], stale).is_err());
    }
    #[test]
    fn mount_transforms_gyro_and_gravity_together() {
        let t = Instant::now();
        let r = Reports {
            gyro: Some(([1., 0., 0.], 2, t)),
            quat: Some(([1., 0., 0., 0.], 2, t)),
            ..Reports::default()
        };
        let s = r.sample([0.5, -0.5, 0.5, -0.5], t).unwrap();
        assert_eq!(s.gyro, [0., -1., 0.]);
        assert_eq!(s.gravity, [-1., 0., 0.]);
        let skewed = Reports {
            gyro: r.gyro,
            quat: Some(([1., 0., 0., 0.], 2, t + Duration::from_millis(31))),
            ..Reports::default()
        };
        assert!(
            skewed
                .sample([1., 0., 0., 0.], t + Duration::from_millis(31))
                .is_err()
        );
    }

    #[test]
    fn report_counts_ignore_duplicate_and_unreliable_reports() {
        let mut r = Reports::default();
        let t = Instant::now();
        let packet = [2, 1, 3, 0, 0, 0, 0, 0, 0, 0];
        r.parse(&packet, t).unwrap();
        r.parse(&packet, t).unwrap();
        r.parse(&[2, 2, 0, 0, 0, 0, 0, 0, 0, 0], t).unwrap();
        assert_eq!(r.gyro_count, 1);
        assert_eq!(r.quat_count, 0);
    }
}
