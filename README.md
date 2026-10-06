# MicroDuck HD1910M

Radxa ZERO 3W + 飞特 HD1910M ×15 + BNO08x。包含当前 Luwu 四模型适配、Rust 后端/底层、RL 训练与 MuJoCo 仿真源码；客户端只发布 APK，不包含 App 源码。

## 1. 接线

![接线总览](docs/images/01_overview.png)

| 连接 | 接法 |
| --- | --- |
| IMU | Radxa 物理针脚 1→VCC、6→GND、27→SDA、28→SCL；3.3V、I²C4 |
| 舵机通信 | USB-C2 HOST → URT-2 Type-C；下排 SCS/TTL 三针口依次连接 15 路舵机 |
| 舵机供电 | 外部 7.4～8.4V → TTL-BUS 的 V1+/G−；不是上方 RS485 的 V2/G |
| 主板供电 | 外部电源经降压模块输出稳定 5V → 物理针脚 4，GND → 9；不要并联另一 USB 供电 |
| 摄像头 | 匹配的 22Pin/0.5mm 排线 → CSI；断电核对触点朝向 |

[实物 IMU/电源接线](docs/images/04_photo_imu_power.png) · [实物舵机/摄像头接线](docs/images/05_photo_usb_servo_camera.png)

原 `exec-baff1ccd-ea24-47f1-a2a6-6d1fc63a12fa.png` 初稿存在端子连线错误，已用上面的校正版替代。电源并联供给各舵机，不是 15 台电源串联；接线以实物丝印为准。

## 2. 舵机编号

![舵机编号爆炸图](docs/images/gif_final_frame.png)

| ID | 关节 |
| --- | --- |
| 1 / 2 / 3 / 4 / 5 | 右踝 / 右膝 / 右髋俯仰 / 右髋侧倾 / 右髋偏航 |
| 6 / 7 / 8 / 9 / 10 | 左踝 / 左膝 / 左髋俯仰 / 左髋侧倾 / 左髋偏航 |
| 11 / 12 | 上颈 `head_pitch` / 下颈 `neck_pitch` |
| 13 / 14 / 15 | 侧头 `head_roll` / 转头 `head_yaw` / 嘴 |

以关节名称映射模型输出，不能把模型数组下标当作电机 ID。[展开动画下载](https://github.com/zhuoligetu123/microduck-hd1910m/releases/latest/download/microduck_exploded.gif)。

## 3. URDF 零位标定

![URDF 零位三视图](docs/images/URDF零位三视图.png)

按图对齐连杆和机身，再核对每个关节正方向。电机零位与模型 `q=0` 对齐，不能用“外壳长边水平/垂直”替代关节几何核对。

`q = direction × (ticks − zero_ticks) × 2π/4096`。示例映射在 [radxa/installation.json](radxa/installation.json)，仅供对照；部署使用本机实际标定文件，不覆盖已有校准，不自动写 EEPROM。嘴闭合为 0，张开为正。

## 4. HOME 站姿

![HOME 三视图](docs/images/HOME三视图.png)

**HOME 是模型站姿，不是舵机编码器零位。** 每个 ONNX 的 `default_joint_pos` 给出各自 HOME；输入关节位置使用 `q−HOME`，输出经本地 ID/方向/零位映射写给舵机。

App“使能”→平滑进入 HOME→静止保持；随后再选择行走/任务。停止保持当前姿态，卸力/清除告警另行触发。不得把 HOME 再校成硬件零位。

## 5. 演示与 APK

[![App 演示](docs/images/app_final.png)](https://github.com/zhuoligetu123/microduck-hd1910m/releases/latest/download/app_preview.mp4)
[![MuJoCo 演示](docs/images/mujoco_00210.jpg)](https://github.com/zhuoligetu123/microduck-hd1910m/releases/latest/download/mujoco_preview.mp4)

[下载 APK](https://github.com/zhuoligetu123/microduck-hd1910m/releases/latest/download/microduck.apk) · [App 视频](https://github.com/zhuoligetu123/microduck-hd1910m/releases/latest/download/app_preview.mp4) · [MuJoCo 视频](https://github.com/zhuoligetu123/microduck-hd1910m/releases/latest/download/mujoco_preview.mp4)

APK 为当前 `com.microduck.control 0.1.1` 内部测试签名包；[ARM64 运行包](https://github.com/zhuoligetu123/microduck-hd1910m/releases/latest/download/microduck-hd1910m-arm64.tar.gz)包含编译后的底层、后端、模型与部署脚本。

两个视频均为 **1920×1080、30fps、180秒**。这是仿真展示，不是实机验收：方向控制已验证；仍有直行偏航，拾取回稳、起身和翻滚落地未全部成功。嘴部在当前 MuJoCo 中为显示动画，非接触动力学。[视频验证记录](docs/video_validation.json)。

## 6. 编译、仿真与部署

Linux 环境：Rust ≥1.89、C/C++ 编译器、pkg-config；训练/仿真使用 Python 3.12 和 uv。硬件运行使用板端 Python venv 提供 ONNX Runtime 动态库。

```bash
git clone https://github.com/zhuoligetu123/microduck-hd1910m.git
cd microduck-hd1910m
python3 scripts/fetch_models.py                 # 固定来源版本并校验四个模型 SHA256
bash scripts/build.sh native                   # robotd、robotctl、Rust App 后端
cd microduck_rl
uv sync --locked
uv pip install websockets==15.0.1
cd ..
microduck_rl/.venv/bin/python scripts/build_sim.py
microduck_rl/.venv/bin/python scripts/run_sim.py --viewer --bind 0.0.0.0:38880
```

安装 APK，连接电脑的 `http://电脑IP:38880`，即可用 App 操作 MuJoCo。只有 API 后端，无浏览器版 App 源码。默认无令牌，**仅用于可信局域网，不要暴露到公网**。

Radxa 部署：

```bash
# Ubuntu Linux 上先安装交叉工具链
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu
bash scripts/build.sh arm64
python3 scripts/deploy.py --ip 192.168.43.3 --user robot
ssh robot@192.168.43.3
cd ~/workspace/microduck-hd1910m
python3 -m venv .venv
.venv/bin/pip install onnxruntime==1.24.4
# 必须使用该设备已核对的关节及 IMU 标定；不填 --allow-motion 为只读模式
python3 scripts/configure.py --calibration /绝对路径/installation.json --allow-motion
bash scripts/run_hardware.sh
```

先启用板端 I²C4，并确认运行用户有串口/I²C 权限。USB 串口使用稳定别名 `/dev/hd1910-servo`，也可用 `configure.py --port /dev/serial/by-id/...`。部署脚本只向新目录传输，不覆盖旧目录或标定；再次部署用 `--directory ~/workspace/新版本目录`。运行前显式停止旧控制服务，避免抢占硬件。本次 ARM64 包要求 glibc ≥2.35，适用于对应 Ubuntu 22.04/24.04 系统。

需要开机运行时，在旧服务停止后执行 `sudo bash scripts/install_service.sh`，再 `sudo systemctl start microduck-hd1910`；日志 `journalctl -u microduck-hd1910 -f`。开机启动程序不等于自动使能。

## 7. 模型与源码

模型目录：`radxa/references/luwu_runtime_20261005/`。原权重固定于 Luwu 提交 `8cdbbd84710d856581982c9eaf0d5e2970666232`，不混入旧 M6 基线。

本次仓库按私有方式托管，四个基线权重随私有仓库保存；公开发布前请先确认第三方权重的再分发授权。

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| `xgoduck_walk.onnx` | 前后行走、侧移、转向、头部控制 | 当前默认行走基线 |
| `xgoduck_pick.onnx` | 拾取动作及嘴部阶段联动 | 实验功能，非抓取成功保证 |
| `xgoduck_getup.onnx` | 倒地起身 | 实验功能，需逐姿态验证 |
| `xgoduck_roulade.onnx` | 翻滚 | 实验功能，落地未稳定 |

嘴部为独立第 15 路，不在 14 维 RL 动作内。此四模型不包含独立坐站、踢腿或踏步权重。

| 路径 | 内容 |
| --- | --- |
| `microduck/duck-control` | HD1910/BNO08x 接入、观测、ONNX 推理与底层 I/O |
| `microduck/robotd` | HOME/保持/行走/任务执行、JSON-RPC |
| `microduck_app/backend` | Rust HTTP/WebSocket、设备发现、App 控制适配 |
| `microduck_rl` | 训练环境、BAM M6、机器人几何、导出与回放源码 |
| `robot_description` | 三视图对应 URDF、网格和舵机编号数据，不含前端程序 |
| `radxa` | 本地 ID/零位适配、模型协议参考及部署配置 |
| `scripts` | 构建、获取模型、仿真和 SSH 部署 |

训练入口见 [RL 说明](docs/training.md)，许可和模型来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。训练中间产物、设备凭据、App/Android 源码均不上传。
