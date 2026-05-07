#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# add project root to PYTHONPATH
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig, FR3RobotAction, FR3ActionMode
from algo.utils.websocket_client_policy import WebsocketClientPolicy


@dataclass
class FR3Config:
    # robot
    robot_id: str = "fr3"
    robot_ip: str = "172.16.0.2"
    load_gripper: bool = True
    relative_dynamics_factor: float = 0.05
    buffer_size: int = 10
    home: bool = True

    # camera (single scene camera)
    scene_camera_id: Optional[str] = "938422072347"
    fps: int = 15
    width: int = 640
    height: int = 480
    camera_buffer: int = 5

    # intrinsics for depth->point cloud (640x480 stream)
    fx: float = 603.340087890625
    fy: float = 603.390869140625
    cx: float = 329.23199462890625
    cy: float = 249.3147430419922


@dataclass
class TaskConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 3333

    # model IO
    obs_horizon: int = 2
    n_points: int = 1024

    # action scaling for real robot (tune conservatively)
    control_hz: int = 6
    delta_pos_scale: float = 1.0
    delta_pos_clip: float = 0.01
    delta_rot_scale: float = 1.0
    delta_rot_clip: float = 0.15  # radians on rotvec norm
    rot_order: str = "current_delta"  # current_delta | delta_current
    invert_rot: bool = False
    robot_quat_format: str = "xyzw"  # xyzw | wxyz

    # gripper mapping
    gripper_open_positive: bool = True


class DP3PointCloudClient:
    def __init__(self, robot_cfg: FR3Config, task_cfg: TaskConfig):
        self.robot_cfg = robot_cfg
        self.task_cfg = task_cfg

        self.policy_client = WebsocketClientPolicy(task_cfg.server_host, task_cfg.server_port)

        self.robot: Optional[FR3Robot] = None
        self.scene_camera: Optional[RealSenseCamera] = None

        self.obs_buf_pc = []
        self.obs_buf_state = []

        self._connect_hardware()

    def _connect_hardware(self):
        robot_conf = FR3RobotConfig(
            id=self.robot_cfg.robot_id,
            robot_ip=self.robot_cfg.robot_ip,
            load_gripper=self.robot_cfg.load_gripper,
            relative_dynamics_factor=self.robot_cfg.relative_dynamics_factor,
            buffer_size=self.robot_cfg.buffer_size,
            initial_joint=None,
            initial_end_pose=[0.4707543519985666, -0.02776023399946076, 0.29950666236991815, 1.0, 0.0, 0.0, 0.0],
        )
        self.robot = FR3Robot(robot_conf)
        self.robot.connect()
        self.robot.read_state()
        self.robot._start_read_thread()

        if self.robot_cfg.home:
            self.robot.home()
            self.robot.gripper.open(0.1)

        cam_cfg = RealSenseCameraConfig(
            fps=self.robot_cfg.fps,
            width=self.robot_cfg.width,
            height=self.robot_cfg.height,
            buffer_size=self.robot_cfg.camera_buffer,
            serial_number_or_name=self.robot_cfg.scene_camera_id,
            use_depth=True,
        )
        self.scene_camera = RealSenseCamera(cam_cfg)
        self.scene_camera.connect()
        self.scene_camera._start_read_thread()

    def _depth_to_point_cloud(self, depth_mm: np.ndarray, n_points: int) -> np.ndarray:
        depth = depth_mm.astype(np.float32) / 1000.0
        h, w = depth.shape

        valid = depth > 0
        if not np.any(valid):
            return np.zeros((n_points, 3), dtype=np.float32)

        yy, xx = np.where(valid)
        z = depth[yy, xx]
        x = (xx.astype(np.float32) - self.robot_cfg.cx) / self.robot_cfg.fx * z
        y = (yy.astype(np.float32) - self.robot_cfg.cy) / self.robot_cfg.fy * z

        pts = np.stack([x, y, z], axis=1).astype(np.float32)

        if pts.shape[0] >= n_points:
            idx = np.random.choice(pts.shape[0], size=n_points, replace=False)
        else:
            idx = np.random.choice(pts.shape[0], size=n_points, replace=True)
        return pts[idx]

    def _build_observation(self):
        assert self.scene_camera is not None
        assert self.robot is not None

        if self.scene_camera.frame_buffer.empty():
            return None

        frame = self.scene_camera.frame_buffer.queue[-1]
        if frame is None or frame.depth is None:
            return None

        state = self.robot.read_state()
        if state is None or state.end_effector_position is None:
            return None

        ee = np.asarray(state.end_effector_position, dtype=np.float32)
        if ee.shape[0] < 7:
            return None
        gripper_width = float(state.gripper_width) if state.gripper_width is not None else 0.0
        agent_pos = np.concatenate([ee[:7], np.asarray([gripper_width], dtype=np.float32)], axis=0).astype(np.float32)

        point_cloud = self._depth_to_point_cloud(frame.depth, self.task_cfg.n_points)

        self.obs_buf_pc.append(point_cloud)
        self.obs_buf_state.append(agent_pos)

        H = int(self.task_cfg.obs_horizon)
        if len(self.obs_buf_pc) > H:
            self.obs_buf_pc.pop(0)
            self.obs_buf_state.pop(0)

        if len(self.obs_buf_pc) < H:
            return None

        obs = {
            "observation": {
                "point_cloud": np.asarray(self.obs_buf_pc, dtype=np.float32)[None, ...],  # (1,To,N,3)
                "agent_pos": np.asarray(self.obs_buf_state, dtype=np.float32)[None, ...],  # (1,To,8)
            }
        }
        return obs

    def _dp3_action_to_robot_absolute(self, a7: np.ndarray) -> FR3RobotAction:
        assert self.robot is not None

        def quat_to_xyzw(q_in: np.ndarray, quat_format: str) -> np.ndarray:
            q = np.asarray(q_in, dtype=np.float64).reshape(4)
            if quat_format == "xyzw":
                return q
            if quat_format == "wxyz":
                return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
            raise ValueError(f"unsupported robot_quat_format={quat_format}")

        a = np.asarray(a7, dtype=np.float32).reshape(-1)
        if a.shape[0] < 7:
            raise ValueError(f"expected action dim>=7, got {a.shape[0]}")

        s = self.robot.read_state()
        ee = np.asarray(s.end_effector_position, dtype=np.float64)  # [x,y,z,q0,q1,q2,q3]

        # delta position
        dpos = np.clip(a[:3] * float(self.task_cfg.delta_pos_scale), -self.task_cfg.delta_pos_clip, self.task_cfg.delta_pos_clip)
        pos_abs = ee[:3] + dpos

        # delta rotation from rotvec (radians)
        drot = np.asarray(a[3:6], dtype=np.float64) * float(self.task_cfg.delta_rot_scale)
        if self.task_cfg.invert_rot:
            drot = -drot
        n = np.linalg.norm(drot)
        if n > float(self.task_cfg.delta_rot_clip) and n > 1e-8:
            drot = drot * (float(self.task_cfg.delta_rot_clip) / n)

        cur_q_xyzw = quat_to_xyzw(ee[3:7], self.task_cfg.robot_quat_format)
        q_delta = R.from_rotvec(drot)
        if self.task_cfg.rot_order == "current_delta":
            q_new = R.from_quat(cur_q_xyzw) * q_delta
        elif self.task_cfg.rot_order == "delta_current":
            q_new = q_delta * R.from_quat(cur_q_xyzw)
        else:
            raise ValueError(f"unsupported rot_order={self.task_cfg.rot_order}")
        q_new_xyzw = q_new.as_quat()

        # gripper command expected by robot.send_action: -1 open, +1 close
        g = float(a[6])
        if self.task_cfg.gripper_open_positive:
            grip_cmd = -1 if g >= 0 else 1
        else:
            grip_cmd = -1 if g <= 0 else 1

        # FR3Robot.send_action expects cartesian_positions = [x,y,z,qx,qy,qz,qw,gripper]
        cart = np.asarray(
            [
                pos_abs[0],
                pos_abs[1],
                pos_abs[2],
                q_new_xyzw[0],
                q_new_xyzw[1],
                q_new_xyzw[2],
                q_new_xyzw[3],
                grip_cmd,
            ],
            dtype=np.float64,
        )
        return FR3RobotAction(cartesian_positions=cart.tolist(), action_mode=FR3ActionMode.ABSOLUTE)

    def run(self, seconds: Optional[float] = None):
        assert self.robot is not None

        start = time.time()
        period = 1.0 / max(1, int(self.task_cfg.control_hz))
        next_t = time.time()

        try:
            while True:
                if seconds is not None and time.time() - start > seconds:
                    break

                obs = self._build_observation()
                if obs is not None:
                    try:
                        out = self.policy_client.infer(obs)
                        actions = np.asarray(out["actions"], dtype=np.float32)
                        # use first action in first batch for low latency closed-loop
                        a = actions[0, 0]
                        robot_action = self._dp3_action_to_robot_absolute(a)
                        self.robot.send_action(robot_action, asynchronous=False)
                    except Exception as e:
                        print(f"[dp3-client] infer/apply failed: {e}")

                next_t += period
                dt = next_t - time.time()
                if dt > 0:
                    time.sleep(min(dt, 0.01))
                else:
                    next_t = time.time()
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt...")
        finally:
            try:
                if self.scene_camera is not None:
                    self.scene_camera.disconnect()
            except Exception:
                pass
            try:
                self.robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=3333)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--control-hz", type=int, default=6)
    parser.add_argument("--delta-pos-clip", type=float, default=0.01)
    parser.add_argument("--delta-rot-clip", type=float, default=0.15)
    parser.add_argument("--rot-order", choices=["current_delta", "delta_current"], default="current_delta")
    parser.add_argument("--invert-rot", action="store_true")
    parser.add_argument("--robot-quat-format", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--gripper-open-positive", action="store_true", default=True)
    parser.add_argument("--gripper-open-negative", action="store_true")
    args = parser.parse_args()

    robot_cfg = FR3Config(
        robot_ip=args.robot_ip,
        home=not args.no_home,
    )
    task_cfg = TaskConfig(
        server_host=args.server_host,
        server_port=args.server_port,
        control_hz=args.control_hz,
        delta_pos_clip=args.delta_pos_clip,
        delta_rot_clip=args.delta_rot_clip,
        rot_order=args.rot_order,
        invert_rot=args.invert_rot,
        robot_quat_format=args.robot_quat_format,
        gripper_open_positive=(not args.gripper_open_negative),
    )

    cli = DP3PointCloudClient(robot_cfg, task_cfg)
    cli.run(seconds=args.seconds)
