# MicroDuck HD1910M RL

基于 MuJoCo / mjlab / PPO 的训练与回放代码。当前发布使用 4 个固定 ONNX，不是本地续训检查点；安装零位、模型 HOME 和 BAM 拟合偏置是不同概念。

## 使用

在仓库根目录完成模型校验和原生编译：

```bash
python3 scripts/fetch_models.py
bash scripts/build.sh native
cd microduck_rl
uv sync --locked
uv pip install websockets==15.0.1
cd ..
microduck_rl/.venv/bin/python scripts/build_sim.py
microduck_rl/.venv/bin/python scripts/run_sim.py --viewer
```

App 使用 APK，连接本机 IP 的 38880 端口。仿真不会连接物理串口。

## 训练与接口

- 环境：`src/mjlab_microduck/tasks/hd1910_bam.py`。
- 执行器：`src/mjlab_microduck/actuator/cpu_hd1910_bam.py`，BAM M6 / P6。
- 观测 61 维，策略输出 14 维，嘴部独立控制；关节以名称映射设备 ID。
- 输入位置为 `q - HOME`；动作历史使用网络原始输出，EMA 与部署一致。
- 控制频率 50 Hz。延迟、限幅、头部命令与接触状态必须一起验证。
- 原生策略适配：`../radxa/reference_policy.py` 和 Rust 控制器。

[训练及测试命令](../docs/training.md) · [完整接线、标定及部署](../README.md) · [来源与许可](../THIRD_PARTY.md)。研究数据、训练日志和未发布候选模型不在本包内。

## 验收边界

本轮自由仿真录到前侧起身及头顶翻滚成功样本，翻滚后是动态步态保持，不是静态双脚锁定。其他初态仍有失败，不代表实机或全工况验收。详见 [本轮动作记录](../docs/skill_validation.json)。
