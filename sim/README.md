# 仿真运行目录

`scripts/build_sim.py` 从发布的机器人几何与 BAM 参数生成 MuJoCo 3.10.0 场景和文件校验值；`scripts/run_sim.py` 生成当前机器路径的配置。

MJB、复制的策略权重、临时 socket 及本机绝对路径不进入 Git。当前仿真为外部参考执行器模型，并非本机电气和摩擦辨识结果。
