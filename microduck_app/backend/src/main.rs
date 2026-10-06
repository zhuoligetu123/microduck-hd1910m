use axum::{Router, extract::{State, WebSocketUpgrade, ws::{Message, WebSocket}},
    http::{HeaderMap, StatusCode}, response::IntoResponse, routing::get};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::{collections::HashMap, env, net::SocketAddr, path::PathBuf, sync::{Arc, atomic::{AtomicU64, Ordering}}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use tokio::{io::{AsyncBufReadExt, AsyncWriteExt, BufReader}, net::{TcpListener, UdpSocket, UnixStream}, sync::{broadcast, mpsc, watch, Mutex}};
use tower_http::services::ServeDir;

#[derive(Clone)]
struct App {
    state: watch::Sender<Value>,
    commands: mpsc::Sender<Value>,
    events: broadcast::Sender<Value>,
    last_move: Arc<Mutex<Option<Instant>>>,
    owner: Arc<AtomicU64>,
    clients: Arc<AtomicU64>,
    epoch: Arc<AtomicU64>,
    received: Arc<Mutex<Instant>>,
    token: Option<String>,
}

fn now_ms() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64 }

fn valid(cmd: &Value) -> bool {
    match cmd["type"].as_str() {
        Some("move") => [("vx",0.30),("vy",0.04),("vyaw",0.8)].iter().all(|(k,limit)|
            cmd.get(k).and_then(Value::as_f64).is_some_and(|v| v.is_finite() && v.abs() <= *limit)),
        Some("mouth") => cmd["open"].as_f64().is_some_and(|v| (0.0..=1.0).contains(&v)),
        Some("head") => [("neck_pitch",1.1),("head_pitch",1.1),("head_yaw",1.4),("head_roll",0.31)]
            .iter().all(|(k,limit)| cmd.get(k).and_then(Value::as_f64)
                .is_some_and(|v| v.is_finite() && v.abs() <= *limit)),
        Some("skill") => cmd["name"].as_str().is_some_and(|s| !s.is_empty() && s.len() <= 40),
        Some("stop") => true,
        Some("enable") => true,
        Some("clear_fault") => true,
        Some("claim_control") => true,
        _ => false,
    }
}

fn motion_limits(state: &Value) -> Value {
    // Published Reference weights need a larger command range than the legacy gait.
    let reference = state.pointer("/data/feedback/reference_native") == Some(&json!(true));
    json!({"vx":if reference {0.30} else {0.15}, "vy":0.04,
           "vyaw":if reference {0.8} else {0.5}})
}

#[test]
fn reference_command_range_matches_advertisement_without_expanding_legacy() {
    let reference = json!({"data":{"feedback":{"reference_native":true}}});
    let legacy = json!({});
    for (vx,wz) in [(0.3,0.0),(-0.3,0.0),(0.0,0.8),(0.0,-0.8)] {
        let cmd = json!({"type":"move","vx":vx,"vy":0,"vyaw":wz});
        assert!(valid_for_state(&cmd,&reference));
        assert!(!valid_for_state(&cmd,&legacy));
    }
    for vx in [-0.301,0.301] {
        assert!(!valid_for_state(&json!({"type":"move","vx":vx,"vy":0,"vyaw":0}),&reference));
    }
    assert!(valid_for_state(&json!({"type":"stop"}),&reference));
    assert!(valid_for_state(&json!({"type":"move","vx":0,"vy":0,"vyaw":0}),&legacy));
}

fn valid_for_state(cmd: &Value, state: &Value) -> bool {
    if !valid(cmd) { return false; }
    let limits = motion_limits(state);
    cmd["type"] != "move" || ["vx","vy","vyaw"].iter().all(|k|
        cmd[k].as_f64().unwrap().abs() <= limits[k].as_f64().unwrap())
}

fn current(cmd: &Value, app: &App) -> bool {
    cmd["_epoch"].as_u64() == Some(app.epoch.load(Ordering::Relaxed)) &&
    (cmd["type"] == "stop" || cmd["_client"].as_u64() == Some(app.owner.load(Ordering::SeqCst))) &&
    cmd["_time"].as_u64().is_some_and(|t| now_ms().saturating_sub(t) < 300)
}

fn client_state(app: &App, id: u64) -> String {
    let mut state = app.state.borrow().clone();
    state["control"] = json!({"client":id,"owner":app.owner.load(Ordering::SeqCst)});
    state["motion_limits"] = motion_limits(&state);
    state.to_string()
}

fn reply(app: &App, cmd: &Value, accepted: bool, reason: &str) {
    let _ = app.events.send(json!({"type":"reply", "client":cmd["_client"], "id":cmd["id"],
        "accepted":accepted,"reason":reason}));
}

async fn stop(app: &App) {
    let epoch = app.epoch.fetch_add(1, Ordering::Relaxed) + 1;
    let _ = app.commands.send(json!({"type":"stop","_epoch":epoch,"_time":now_ms()})).await;
}

fn allowed(headers: &HeaderMap, app: &App) -> bool {
    app.token.as_ref().is_none_or(|t| headers.get("x-duck-token")
        .and_then(|h| h.to_str().ok()).is_some_and(|h| h == t))
}

async fn health(State(app): State<App>, headers: HeaderMap) -> impl IntoResponse {
    if !allowed(&headers, &app) { return (StatusCode::UNAUTHORIZED, "token required").into_response(); }
    let online = app.received.lock().await.elapsed() < Duration::from_secs(1);
    let state = app.state.borrow().clone();
    axum::Json(json!({"ok": true, "online":online, "protocol": 1, "source":state["source"],
        "robotd_connected":state["robotd_connected"], "hardware_health":state["hardware_health"],
        "motion_error":state["motion_error"]})).into_response()
}

fn hardware_status(mut state: Value, health: &Value, fresh: bool) -> Value {
    state["robotd_connected"] = json!(true);
    state["hardware_health"] = health.clone();
    state["online"] = json!(fresh);
    if !fresh {
        state["motion_available"] = json!(false);
        state["head_control_available"] = json!(false);
        state["motion_error"] = json!(health["reason"].as_str().unwrap_or_else(|| {
            if health.pointer("/imu/ready") == Some(&json!(false)) {
                "IMU not ready; waiting for hardware feedback"
            } else { "Waiting for fresh hardware feedback" }
        }));
    }
    state
}

async fn discover(State(app): State<App>, headers: HeaderMap) -> impl IntoResponse {
    if !allowed(&headers,&app) { return StatusCode::UNAUTHORIZED.into_response(); }
    let Ok(socket) = UdpSocket::bind("0.0.0.0:0").await else { return StatusCode::SERVICE_UNAVAILABLE.into_response(); };
    let _ = socket.set_broadcast(true);
    for addr in ["255.255.255.255:38882","127.0.0.1:38882"] {
        let _ = socket.send_to(b"MICRODUCK_DISCOVER_V1",addr).await;
    }
    let until = tokio::time::Instant::now()+Duration::from_secs(2);
    let mut devices = HashMap::new();
    let mut buf=[0;2048];
    while let Ok(Ok((n,peer))) = tokio::time::timeout_at(until,socket.recv_from(&mut buf)).await {
        if let Ok(mut value) = serde_json::from_slice::<Value>(&buf[..n]) {
            if value["protocol"] == "microduck-app-v1" {
                value["host"]=json!(peer.ip().to_string());devices.insert(peer.ip(),value);
            }
        }
    }
    axum::Json(devices.into_values().collect::<Vec<_>>()).into_response()
}

async fn discovery_server(app: App, addr: SocketAddr) -> std::io::Result<()> {
    let socket = UdpSocket::bind(SocketAddr::new(addr.ip(),38882)).await?;
    let mut buf=[0;256];
    loop {
        let (n,peer)=socket.recv_from(&mut buf).await?;
        if &buf[..n]!=b"MICRODUCK_DISCOVER_V1" { continue; }
        let message=json!({"protocol":"microduck-app-v1","name":env::var("MICRODUCK_APP_NAME").unwrap_or("MicroDuck".into()),
            "port":addr.port(),"auth_required":app.token.is_some(),
            "online":app.received.lock().await.elapsed()<Duration::from_secs(1)});
        socket.send_to(message.to_string().as_bytes(),peer).await?;
    }
}

async fn socket(upgrade: WebSocketUpgrade, State(app): State<App>, headers: HeaderMap) -> impl IntoResponse {
    if app.token.is_none() {
        if let Some(origin)=headers.get("origin").and_then(|s|s.to_str().ok()) {
            let host=headers.get("host").and_then(|s|s.to_str().ok()).unwrap_or("");
            if origin!=format!("http://{host}") && origin!="http://app.microduck.local"
                && origin!="http://127.0.0.1:4173" && origin!="http://localhost:4173" {
                return StatusCode::FORBIDDEN.into_response();
            }
        }
    }
    // WebSocket cannot set custom headers in a browser; token is supplied as a subprotocol.
    let token_ok = app.token.as_ref().is_none_or(|t| headers.get("sec-websocket-protocol")
        .and_then(|h| h.to_str().ok()).is_some_and(|h| h.split(',').any(|p| p.trim() == t)));
    if !token_ok { return StatusCode::UNAUTHORIZED.into_response(); }
    let upgrade = if let Some(token) = &app.token { upgrade.protocols([token.clone()]) } else { upgrade };
    upgrade.max_message_size(4096).on_upgrade(move |ws| client(ws, app)).into_response()
}

async fn client(ws: WebSocket, app: App) {
    let id = app.clients.fetch_add(1, Ordering::Relaxed) + 1;
    let (mut out, mut input) = ws.split();
    let mut states = app.state.subscribe();
    let mut events = app.events.subscribe();
    let mut last_seq = 0u64;
    let initial = client_state(&app,id);
    let _ = out.send(Message::Text(initial.into())).await;
    loop {
        tokio::select! {
            changed = states.changed() => {
                if changed.is_err() { break; }
                states.borrow_and_update();
                let frame = client_state(&app,id);
                if out.send(Message::Text(frame.into())).await.is_err() { break; }
            }
            event = events.recv() => {
                if let Ok(value) = event {
                    if value["client"].as_u64().is_some_and(|c| c != id) { continue; }
                    if out.send(Message::Text(value.to_string().into())).await.is_err() { break; }
                }
            }
            frame = input.next() => {
                let raw = match frame { Some(Ok(Message::Text(raw))) => raw,
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue, _ => break };
                let Ok(mut cmd) = serde_json::from_str::<Value>(&raw) else { continue; };
                if !cmd.is_object() { continue; }
                cmd["_client"] = json!(id);
                if !valid_for_state(&cmd,&app.state.borrow()) { reply(&app,&cmd,false,"invalid command or outside policy command range"); continue; }
                let kind = cmd["type"].as_str().unwrap().to_owned();
                let manual_reset = kind == "clear_fault" && app.state.borrow()["robotd_connected"] == true;
                if kind != "stop" && !manual_reset && app.received.lock().await.elapsed() > Duration::from_secs(1) {
                    reply(&app,&cmd,false,"feedback unavailable"); continue;
                }
                if kind == "claim_control" {
                    *app.last_move.lock().await = None;
                    app.owner.store(id,Ordering::SeqCst);
                    stop(&app).await;
                    reply(&app,&cmd,true,"");
                    continue;
                }
                if !matches!(kind.as_str(), "stop" | "clear_fault") && app.state.borrow()["motion_available"] == false {
                    let reason=app.state.borrow()["motion_error"].as_str()
                        .unwrap_or("hardware motion unavailable").to_owned();
                    reply(&app,&cmd,false,&reason); continue;
                }
                if kind == "clear_fault" && app.state.borrow()["source"] != "robotd" {
                    reply(&app,&cmd,false,"fault reset requires native robotd"); continue;
                }
                if kind == "enable" && (app.state.borrow()["source"] != "robotd"
                    || app.state.borrow()["policy_unavailable"].is_string()) {
                    reply(&app,&cmd,false,"skill policy not loaded"); continue;
                }
                if !matches!(kind.as_str(), "stop" | "enable" | "clear_fault") && app.state.borrow()["source"] == "robotd"
                    && app.state.borrow().pointer("/data/feedback/policy_enabled") == Some(&json!(false)) {
                    reply(&app,&cmd,false,"enable the robot first"); continue;
                }
                if kind == "skill" && !app.state.borrow()["skills"].as_array()
                    .is_some_and(|skills| skills.contains(&cmd["name"])) {
                    reply(&app,&cmd,false,"skill policy not loaded"); continue;
                }
                if kind == "head" && app.state.borrow()["head_control_available"] != true {
                    reply(&app,&cmd,false,"head control unavailable in this mode"); continue;
                }
                if kind != "stop" {
                    let owner = app.owner.compare_exchange(0,id,Ordering::SeqCst,Ordering::SeqCst).unwrap_or_else(|v|v);
                    if owner != 0 && owner != id { reply(&app,&cmd,false,"another client owns control"); continue; }
                }
                if kind == "move" {
                    let seq = cmd.get("seq").and_then(Value::as_u64).unwrap_or(0);
                    if seq <= last_seq { continue; }
                    last_seq = seq;
                    *app.last_move.lock().await = Some(Instant::now());
                }
                if matches!(kind.as_str(), "skill" | "stop" | "enable" | "clear_fault") {
                    *app.last_move.lock().await = None;
                    // Invalidate queued joystick packets before the new action.
                    app.epoch.fetch_add(1,Ordering::Relaxed);
                }
                if kind == "mouth" && !cmd.get("open").and_then(Value::as_f64)
                    .is_some_and(|v| (0.0..=1.0).contains(&v)) { continue; }
                if kind == "skill" && cmd.get("name").and_then(Value::as_str)
                    .is_none_or(|s| s.len() > 40 || s.is_empty()) { continue; }
                cmd["_epoch"] = json!(app.epoch.load(Ordering::Relaxed));
                cmd["_time"] = json!(now_ms());
                if let Err(err) = app.commands.try_send(cmd) { reply(&app,&err.into_inner(),false,"command queue full"); }
            }
        }
    }
    if app.owner.compare_exchange(id,0,Ordering::SeqCst,Ordering::SeqCst).is_ok() { stop(&app).await; }
}

async fn sim_source(app: App, mut commands: mpsc::Receiver<Value>) -> std::io::Result<()> {
    let telemetry = UdpSocket::bind("127.0.0.1:38890").await?;
    let sender = UdpSocket::bind("127.0.0.1:0").await?;
    let mut buf = vec![0; 65536];
    loop {
        tokio::select! {
            received = telemetry.recv_from(&mut buf) => {
                let (n, _) = received?;
                if let Ok(value) = serde_json::from_slice::<Value>(&buf[..n]) {
                    if value.get("type") == Some(&json!("state")) {
                        *app.received.lock().await = Instant::now();
                        app.state.send_replace(value);
                    } else if value.get("type") == Some(&json!("reply")) { let _ = app.events.send(value); }
                }
            }
            Some(cmd) = commands.recv() => {
                if !current(&cmd,&app) { reply(&app,&cmd,false,"command expired"); continue; }
                sender.send_to(cmd.to_string().as_bytes(), "127.0.0.1:38891").await?;
            }
        }
    }
}

async fn robotd_source(app: App, mut commands: mpsc::Receiver<Value>, path: PathBuf) {
    loop {
        match UnixStream::connect(&path).await {
            Ok(stream) => {
                app.state.send_replace(json!({"type":"state","source":"robotd","online":false,
                    "robotd_connected":true,"motion_available":false,"skills":[],
                    "motion_error":"Waiting for fresh hardware feedback"}));
                let (reader, mut writer) = stream.into_split();
                let mut lines = BufReader::new(reader).lines();
                for hello in [json!({"jsonrpc":"2.0","id":1,"method":"hello","params":{"api_version":23}}),
                              json!({"jsonrpc":"2.0","id":2,"method":"robot.subscribe","params":{"hz":20}}),
                              json!({"jsonrpc":"2.0","id":3,"method":"robot.health","params":{}})] {
                    if writer.write_all(format!("{hello}\n").as_bytes()).await.is_err() { break; }
                }
                let mut id = 3u64;
                let mut pending: HashMap<u64, Value> = HashMap::new();
                let mut policy_unavailable = Value::Null;
                let mut catalogue = tokio::time::interval(Duration::from_secs(2));
                let mut last_reply = Instant::now();
                catalogue.tick().await;
                loop {
                    tokio::select! {
                        line = lines.next_line() => {
                            let Ok(Some(line)) = line else { break; };
                            let Ok(value) = serde_json::from_str::<Value>(&line) else { continue; };
                            last_reply = Instant::now();
                            if value.get("method") == Some(&json!("robot.state")) {
                                if let Some(payload) = value.get("params") {
                                    *app.received.lock().await = Instant::now();
                                    let skills = app.state.borrow().get("skills").cloned().unwrap_or(json!([]));
                                    let fault = payload.pointer("/feedback/error").cloned().unwrap_or(Value::Null);
                                    let motion = payload.pointer("/feedback/motion_available").and_then(Value::as_bool).unwrap_or(true)
                                        && !fault.is_string();
                                    app.state.send_replace(json!({"type":"state", "online":true, "robotd_connected":true, "source":"robotd", "data":payload,
                                        "skills":skills,"motion_available":motion,"head_control_available":motion
                                            && !matches!(payload["policy"].as_str(), Some("step" | "recovery" | "ground_pick" | "roulade")),
                                        "motion_error":fault,
                                        "experimental_skills":if payload.pointer("/feedback/reference_native") == Some(&json!(true)) {
                                            json!(["recovery","ground_pick","roulade"])
                                        } else if payload.pointer("/feedback/supported_m6") == Some(&json!(true)) { json!(["step"]) } else {json!([])},
                                        "policy_unavailable":policy_unavailable}));
                                }
                            } else if value.get("id") == Some(&json!(2)) {
                                let mut skills = value.pointer("/result/skills").and_then(Value::as_array).cloned().unwrap_or_default();
                                if value.pointer("/result/ground_pick").is_some_and(Value::is_string) { skills.push(json!("ground_pick")); }
                                if value.pointer("/result/walk").is_some_and(Value::is_string) { skills.extend([json!("stand"),json!("walk")]); }
                                policy_unavailable = value.pointer("/result/unavailable").cloned().unwrap_or(Value::Null);
                                if policy_unavailable.is_string() { skills.clear(); }
                                let mut state = app.state.borrow().clone();
                                state["skills"] = json!(skills);
                                state["policy_unavailable"] = policy_unavailable.clone();
                                app.state.send_replace(state);
                            } else if value.get("id") == Some(&json!(3)) {
                                let fresh = app.received.lock().await.elapsed() < Duration::from_secs(1);
                                let state = app.state.borrow().clone();
                                app.state.send_replace(hardware_status(state, &value["result"], fresh));
                            } else if value.get("id").is_some() {
                                if let Some(cmd) = value["id"].as_u64().and_then(|id|pending.remove(&id)) {
                                    let accepted = value.pointer("/result/accepted").and_then(Value::as_bool).unwrap_or(false);
                                    let reason = value.pointer("/result/reason").or_else(||value.pointer("/error/message")).and_then(Value::as_str).unwrap_or("");
                                    reply(&app,&cmd,accepted,reason);
                                }
                            }
                        }
                        _ = catalogue.tick() => {
                            if last_reply.elapsed() > Duration::from_secs(6) { break; }
                            let subscribe = json!({"jsonrpc":"2.0","id":2,"method":"robot.subscribe","params":{"hz":20}});
                            let health = json!({"jsonrpc":"2.0","id":3,"method":"robot.health","params":{}});
                            if !matches!(tokio::time::timeout(Duration::from_millis(200), writer.write_all(format!("{subscribe}\n{health}\n").as_bytes())).await, Ok(Ok(()))) { break; }
                        }
                        Some(cmd) = commands.recv() => {
                            if !current(&cmd,&app) { reply(&app,&cmd,false,"command expired"); continue; }
                            let kind = cmd["type"].as_str().unwrap_or("");
                            let (method, params, request) = match kind {
                                "enable" => ("robot.enable", json!({"on":true}), true),
                                "clear_fault" => ("robot.clear_fault", json!({}), true),
                                "move" => ("robot.move", json!({"vx":cmd["vx"],"vy":cmd["vy"],"vyaw":cmd["vyaw"]}), false),
                                "stop" => ("robot.stop", json!({}), true),
                                "skill" if cmd["name"] == "stand" => ("robot.stop", json!({}), true),
                                "skill" if cmd["name"] == "walk" => ("robot.stop", json!({}), true),
                                "skill" => ("robot.do", json!({"skill":cmd["name"]}), true),
                                "mouth" => ("robot.mouth", json!({"open":cmd["open"]}), true),
                                "head" => ("robot.head", json!({"neck_pitch":cmd["neck_pitch"],"head_pitch":cmd["head_pitch"],
                                    "head_yaw":cmd["head_yaw"],"head_roll":cmd["head_roll"]}), true),
                                _ => continue,
                            };
                            if pending.len() > 32 { reply(&app,&cmd,false,"robotd not acknowledging"); break; }
                            let msg = if request { id += 1; pending.insert(id,cmd.clone()); json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}) }
                                      else { json!({"jsonrpc":"2.0","method":method,"params":params}) };
                            if !matches!(tokio::time::timeout(Duration::from_millis(200),writer.write_all(format!("{msg}\n").as_bytes())).await,Ok(Ok(()))) { break; }
                        }
                    }
                }
            }
            Err(err) => eprintln!("robotd unavailable: {err}"),
        }
        app.state.send_replace(json!({"type":"state","source":"robotd","online":false,"robotd_connected":false,
            "motion_available":false,"motion_error":"Control service disconnected; reconnecting","skills":[]}));
        *app.received.lock().await = Instant::now()-Duration::from_secs(60);
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bind = env::var("MICRODUCK_APP_BIND").unwrap_or("127.0.0.1:38880".into());
    let token = env::var("MICRODUCK_APP_TOKEN").ok().filter(|s| !s.is_empty());
    let addr: SocketAddr = bind.parse()?;
    if !addr.ip().is_loopback() && token.is_none() {
        if env::var("MICRODUCK_APP_ALLOW_OPEN_LAN").as_deref() != Ok("1") {
            return Err("non-loopback bind without a token requires MICRODUCK_APP_ALLOW_OPEN_LAN=1".into());
        }
        eprintln!("WARNING: unauthenticated LAN control is enabled");
    }
    let (state, _) = watch::channel(json!({"type":"state","online":false,"source":"sim","skills":[]}));
    let (commands, receiver) = mpsc::channel(8);
    let (events, _) = broadcast::channel(64);
    let app = App { state, commands, events, last_move: Arc::new(Mutex::new(None)),
                    owner: Arc::new(AtomicU64::new(0)), clients: Arc::new(AtomicU64::new(0)), epoch: Arc::new(AtomicU64::new(0)),
                    received: Arc::new(Mutex::new(Instant::now()-Duration::from_secs(60))), token };
    let source = app.clone();
    if let Ok(path) = env::var("MICRODUCK_ROBOTD_SOCKET") {
        tokio::spawn(robotd_source(source, receiver, path.into()));
    } else {
        tokio::spawn(async move { if let Err(err) = sim_source(source, receiver).await { eprintln!("sim source: {err}"); } });
    }
    let watchdog = app.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            interval.tick().await;
            let mut last = watchdog.last_move.lock().await;
            if last.is_some_and(|t| t.elapsed() > Duration::from_millis(300)) {
                *last = None;
                stop(&watchdog).await;
            }
        }
    });
    let static_dir = env::var("MICRODUCK_APP_WEB").unwrap_or("../web/dist".into());
    let discovery=app.clone();
    tokio::spawn(async move { if let Err(err)=discovery_server(discovery,addr).await { eprintln!("discovery: {err}"); } });
    let router = Router::new().route("/ws", get(socket)).route("/api/health", get(health))
        .route("/api/discover",get(discover))
        .fallback_service(ServeDir::new(static_dir).append_index_html_on_directories(true))
        .with_state(app);
    let listener = TcpListener::bind(addr).await?;
    println!("MicroDuck app: http://{addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn health_reply_does_not_make_missing_telemetry_ready() {
        let state = hardware_status(json!({"online":true,"motion_available":true}),
            &json!({"imu":{"ready":false},"reason":"startup failed"}), false);
        assert_eq!(state["robotd_connected"], true);
        assert_eq!(state["online"], false);
        assert_eq!(state["motion_available"], false);
        assert_eq!(state["motion_error"], "startup failed");
    }
    #[test]
    fn health_reply_preserves_live_motion_fault() {
        let state = hardware_status(json!({"motion_available":false,"motion_error":"motor alarm"}),
            &json!({"healthy":true}), true);
        assert_eq!(state["motion_error"], "motor alarm");
        assert_eq!(state["motion_available"], false);
    }
    #[test]
    fn startup_motor_alarm_is_not_replaced_by_generic_imu_error() {
        let state = hardware_status(json!({}),
            &json!({"imu":{"ready":false},"reason":"Feetech: ID 9: alarm 0x03"}), false);
        assert_eq!(state["motion_error"], "Feetech: ID 9: alarm 0x03");
        assert_eq!(state["motion_available"], false);
    }
    #[test]
    fn input_matches_trained_velocity_range() {
        assert!(valid(&json!({"type":"move","vx":0.15,"vy":-0.04,"vyaw":0.5})));
        assert!(!valid(&json!({"type":"move","vx":0.31,"vy":0,"vyaw":0})));
        assert!(!valid(&json!({"type":"move","vx":"NaN","vy":0,"vyaw":0})));
        assert!(!valid(&json!({"type":"move","vx":0,"vy":0})));
        assert!(!valid(&json!({"type":"mouth","open":-1})));
        assert!(valid(&json!({"type":"mouth","open":1})));
        assert!(!valid(&json!({"type":"skill","name":""})));
    }
    #[test]
    fn viewer_cannot_release_owner() {
        let owner=AtomicU64::new(7);
        assert!(owner.compare_exchange(8,0,Ordering::SeqCst,Ordering::SeqCst).is_err());
        assert_eq!(owner.load(Ordering::SeqCst),7);
        assert!(owner.compare_exchange(7,0,Ordering::SeqCst,Ordering::SeqCst).is_ok());
    }
    #[test]
    fn head_requires_complete_finite_radian_command() {
        let mut cmd=json!({"type":"head","neck_pitch":0,"head_pitch":0.3,"head_yaw":-0.5,"head_roll":0});
        assert!(valid(&cmd));
        cmd["head_yaw"]=json!(1.41); assert!(!valid(&cmd));
        cmd["head_yaw"]=json!("0"); assert!(!valid(&cmd));
        cmd["head_yaw"]=Value::Null; assert!(!valid(&cmd));
        assert!(!valid(&json!({"type":"head","head_yaw":0.2})));
    }
}
