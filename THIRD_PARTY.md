# 来源与许可

本仓库是本地集成快照，不代表上游原作者发布，也不将第三方文件统一改授为本项目许可。

| 部分 | 来源 | 许可/边界 |
| --- | --- | --- |
| Rust 基础工程 | https://github.com/pollen-robotics/microduck | 保留 `microduck/LICENSE` Apache-2.0；本地 HD1910/BNO/App 适配包含修改 |
| RL 基础工程及几何 | https://github.com/pollen-robotics/microduck_rl | 保留 `microduck_rl/LICENSE` Apache-2.0 及源文件声明 |
| BAM 执行器实现 | https://github.com/Rhoban/bam | 通过 RL 的锁文件安装，不将第三方实现另行改授许可 |
| 当前四个 ONNX | https://github.com/LuwuDynamics/xgoduck_runtime_arduino/tree/8cdbbd84710d856581982c9eaf0d5e2970666232/python | 固定来源、未修改；所核对版本未附仓库 LICENSE。公开再分发/商用授权需另行确认，下载脚本不授予额外权利 |
| M6 参数参考 | https://github.com/LuwuDynamics/xgoduck_rl | 外部参考参数，不等于本机已完成执行器辨识 |
| APK、演示与照片 | 本地工作区及用户提供素材 | 不包含 APK 对应前端源码，不声明第三方板卡照片的额外授权 |

导出的 Rust 与 RL 源码来自含本地未提交修改的工作区，不能仅凭上游 HEAD 重建本地修改。发布提交及文件校验清单才是本次集成快照的依据。
