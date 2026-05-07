# Roby

Franka FR3 的遥操作、数据回放、以及 DP3 在线推理客户端。

## 快速目录

1. 环境准备
2. DP3 在线推理（177 服务端 + 机器人端客户端）
3. 机器人复位
4. Parquet 轨迹回放
5. 常见问题

## 1. 环境准备

在机器人这台机器（例如 111）执行：

```bash
cd /home/server/franka/ckf_roby
conda activate roby
```

如果遇到 websocket 相关报错，可安装：

```bash
conda run -n roby python -m pip install websockets
```

## 2. DP3 在线推理

### 2.1 在 177 启动 DP3 服务端

```bash
cd /home/server/franka/ckf_roby
python deployment/serve_policy_dp3.py \
	--ckpt /hard_data1/user/chenkuifan/DemoGen/outputs/dp3_ba20260427_demogen_d1_posgen300_e120/checkpoints/99.ckpt \
	--host 0.0.0.0 \
	--port 3333 \
	--device cuda:0
```

看到以下日志说明服务已正常启动：

- `[dp3-server] start`
- `[dp3-server] host=0.0.0.0 port=3333 ...`

### 2.2 在机器人端做联通测试

```bash
cd /home/server/franka/ckf_roby
conda run -n roby python deployment/smoke_test_dp3_ws.py --host 10.184.17.177 --port 3333 --loops 1
```

输出包含 `[smoke] PASS` 表示联通正常。

### 2.3 启动在线客户端（机器人端）

```bash
cd /home/server/franka/ckf_roby
conda run -n roby python algo/dp3_client_pc.py \
	--server-host 10.184.17.177 \
	--server-port 3333 \
	--robot-ip 172.16.0.2 \
	--seconds 600 \
	--rot-order current_delta \
	--robot-quat-format xyzw
```

说明：

- 默认会在启动时自动 `home`（不想 home 可以加 `--no-home`）。
- 若姿态仍有轻微反向，可尝试 `--invert-rot` 或 `--rot-order delta_current`。

## 3. 机器人复位

只复位不跑模型：

```bash
cd /home/server/franka/ckf_roby
conda run -n roby python -c "from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig; print('[home] connecting...', flush=True); cfg=FR3RobotConfig(id='fr3', robot_ip='172.16.0.2', load_gripper=False, relative_dynamics_factor=0.05, buffer_size=10); r=FR3Robot(cfg); r.connect(); print('[home] homing...', flush=True); r.home(); print('[home] disconnect...', flush=True); r.disconnect(); print('[home] done', flush=True)"
```

## 4. Parquet 轨迹回放

### 4.1 Dry-run（不发机器人动作）

```bash
cd /home/server/franka/ckf_roby
conda run -n roby python algo/replay_parquet_on_robot.py \
	--parquet /home/server/franka/ckf_roby/algo/episode_00000.parquet \
	--dry-run \
	--parquet-quat-format xyzw
```

### 4.2 真机回放（基础）

```bash
cd /home/server/franka/ckf_roby
conda run -n roby python algo/replay_parquet_on_robot.py \
	--parquet /home/server/franka/ckf_roby/algo/episode_00000.parquet \
	--robot-ip 172.16.0.2 \
	--start 0 --steps 270 \
	--hz 3 \
	--pos-clip 0.002 \
	--rot-clip 0.03 \
	--home \
	--parquet-quat-format xyzw \
	--robot-quat-format xyzw
```

### 4.3 真机回放（含夹爪离散控制）

```bash
cd /home/server/franka/ckf_roby
conda run -n roby python algo/replay_parquet_on_robot.py \
	--parquet /home/server/franka/ckf_roby/algo/episode_00000.parquet \
	--robot-ip 172.16.0.2 \
	--start 0 --steps 270 \
	--hz 3 \
	--pos-clip 0.002 \
	--rot-clip 0.03 \
	--home \
	--parquet-quat-format xyzw \
	--robot-quat-format xyzw \
	--load-gripper \
	--send-gripper \
	--gripper-open-th 0.06 \
	--gripper-close-th 0.04 \
	--gripper-min-interval-steps 12 \
	--gripper-delay-steps 15
```

## 5. 常见问题

### Q1: `ModuleNotFoundError: No module named 'algo'`

请在仓库根目录执行命令：

```bash
cd /home/server/franka/ckf_roby
```

### Q2: `ModuleNotFoundError: No module named 'websockets'`

在 roby 环境安装：

```bash
conda run -n roby python -m pip install websockets
```

### Q3: 路径里有括号，bash 报语法错误

把路径用双引号包起来，例如：

```bash
--parquet "/path/episode_00000(2).parquet"
```

### Q4: base 环境能读文件，roby 环境报错

请统一使用 roby 环境运行机器人相关命令，避免环境不一致：

```bash
conda run -n roby python ...
```

### Q5: 在线模型姿态翻转

优先使用：

- `--robot-quat-format xyzw`
- `--rot-order current_delta`

若仍有偏差，再试：

- `--invert-rot`
- `--rot-order delta_current`

## 补充：Spacemouse 遥操作

```bash
conda activate roby
streamlit run example/app.py
```
