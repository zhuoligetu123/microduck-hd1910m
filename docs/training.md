# 训练与验证

发布默认模型是未修改的 Reference 四个 ONNX，不是本机 PPO 续训检查点；本仓库未包含优化器、训练日志、W&B 数据或历史候选权重。ONNX 不能直接当 PPO 优化器检查点续训。

本机研究训练环境保留在 `microduck_rl/src/mjlab_microduck/tasks`。先验证环境和动作语义，再训练；不要把下列冒烟命令视为复现 Reference 原权重的配方。

```bash
cd microduck_rl
uv sync --locked
uv run list-envs
uv run train Mjlab-Velocity-Flat-MicroDuck-HD1910-XgoBam-P6-Slew \
  --env.scene.num-envs 64 --agent.max_iterations 5
```

训练环境与部署必须逐项核对：关节名称顺序、61维观测、14维动作、各模型 HOME、动作历史、EMA、目标限幅及50Hz时序。舵机 ID/方向/零位和 IMU 安装变换属于实机配置，不是 BAM `q_offset`。

协议回归（无硬件访问）：

```bash
cd microduck
cargo test --locked -p duck-control -p robotd-params -p robotd
cargo build --locked -p robotd
cd ../microduck_app/backend
cargo build --locked
cd ../..
PYTHONPATH=radxa microduck_rl/.venv/bin/python -m unittest discover -s radxa -p test_reference_policy.py
microduck_rl/.venv/bin/python scripts/build_sim.py
microduck_rl/.venv/bin/python tests/test_reference_native.py
```

最后一项使用固定基座 MuJoCo 验证 HOME、嘴、行走指令、停止与技能切换，不代表自由站立或实机步态通过。自由仿真用 `scripts/run_sim.py --viewer`，不要加 `--supported`。

部分历史研究测试引用未发布的训练结果/外部基准数据，不能把整个研究测试目录当作本发布的全部验收入口。发布验收范围见 `docs/release_validation.json`。
