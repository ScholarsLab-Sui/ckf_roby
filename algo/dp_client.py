import os
from pathlib import Path
import sys
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 找到项目根目录 (假设 algo/ 在项目根目录下，所以是 .parent.parent)
project_root = current_file.parent.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(project_root))
import cv2
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Optional, List
from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig, FR3RobotAction, FR3ActionMode
from utils import websocket_client_policy
from PIL import Image
from algo.utils.action_mapping import (
    delta_joint_action_mapping,
    absolute_position_action_mapping,
    absolute_joint_action_mapping,
    delta_absolute_position_action_mapping,
)

# --- Configuration from test_client.py ---
OBS_HORIZON = 2

@dataclass
class FR3Config:
    # --- robot setting ---
    robot_id: str = "fr3"
    robot_ip: str = "172.16.0.2"
    load_gripper: bool = True
    relative_dynamics_factor: float = 0.05
    buffer_size: int = 10
    home: bool = True                 

    # --- camera setting ---
    scene_camera_id: Optional[str] = "938422072347" # left camera oringinal
    # scene_camera_id: Optional[str] = "112322077048" # left camera new
    wrist_camera_id: Optional[str] = "112322074840"
    fps: int = 15
    width: int = 640
    height: int = 480
    camera_buffer: int = 5

    # --- control mode ---
    action_mode: str = "POSITION_DELTA"  # ["POSITION_DELTA", "JOINT_DELTA", "POSITION_ABSOLUTE", "JOINT_ABSOLUTE"]
    img_update_rate: int = 15            
    asynchronous: bool = False           # move asynchronous


@dataclass
class TaskConfig:
    algorithm: str = "diffusion_policy"
    is_online: bool = True
    server_host: str = "10.184.17.133"
    server_port: int = 3333 # Port from test_client.py / serve_policy.py default
    object_to_manipulate: str = "banana"  # banana, carrot, oranges
    # [CLIENT-SMALL-1] Keep legacy path default; enable tiny DP3 path by setting backend="dp3_small"
    backend: str = "legacy"  # legacy | dp3_small
    dp3_obs_horizon: int = 2
    dp3_n_points: int = 1024

class RobotClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(cfg.server_host, cfg.server_port)

    def infer(self, obs):
        return self.policy_client.infer(obs)

class shw_franka:
    def __init__(self, robot_cfg, task_cfg):
        self.current_gripper_state = 0  # -1: open, 1: closed (跟踪当前夹爪实际状态)
        self.target_gripper_state = 0   # 0: open, 1: closed (策略要求的目标状态)
        self.cfg = robot_cfg
        self.action_mode = robot_cfg.action_mode
        self.delta_position_action_mapping = delta_absolute_position_action_mapping.__get__(self)
        self.delta_joint_action_mapping = delta_joint_action_mapping.__get__(self)
        self.absolute_position_action_mapping = absolute_position_action_mapping.__get__(self)
        self.absolute_joint_action_mapping = absolute_joint_action_mapping.__get__(self)
        self.action_mapping = {
            "POSITION_DELTA": self.delta_position_action_mapping,
            "JOINT_DELTA": self.delta_joint_action_mapping,
            "POSITION_ABSOLUTE": self.absolute_position_action_mapping,
            "JOINT_ABSOLUTE": self.absolute_joint_action_mapping,
        }
        self.robot = None
        self.client = RobotClient(task_cfg)
        self.object_to_manipulate = task_cfg.object_to_manipulate
        if self.object_to_manipulate == "banana":
            self.language_instruction = "place banana into red bowl"
        elif self.object_to_manipulate == "carrot":
            self.language_instruction = "pick up carrot and place it into red bowl"
        elif self.object_to_manipulate == "orange":
            self.language_instruction = "grasp orange and put it into red bowl"
        else:
            raise ValueError("Unsupported object to manipulate")
        
        self.scene_camera = None
        self.wrist_camera = None
        self.scene_camera_id = robot_cfg.scene_camera_id
        self.wrist_camera_id = robot_cfg.wrist_camera_id
        self.scene_camera_image = None
        self.scene_camera_depth = None  # [CLIENT-SMALL-2] optional depth cache for dp3_small
        self.wrist_camera_image = None
        self._img_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads = []
        
        # Sliding window buffer from test_client.py
        self.obs_buffer = {
            "rgb": [],
            "state": []
        }
        # [CLIENT-SMALL-3] dedicated tiny buffers for DP3 input protocol
        self.dp3_obs_buffer = {
            "point_cloud": [],
            "agent_pos": [],
        }
        
        self.devices = self.setup_hardwares()

    def setup_hardwares(self):
        robot_conf = FR3RobotConfig(
            id=self.cfg.robot_id,
            robot_ip=self.cfg.robot_ip,
            load_gripper=self.cfg.load_gripper,
            relative_dynamics_factor=self.cfg.relative_dynamics_factor,
            buffer_size=self.cfg.buffer_size,
            initial_joint = None,
            initial_end_pose=[0.4707543519985666, -0.02776023399946076, 0.29950666236991815, 1.0, 0.0, 0.0, 0.0]
        )
        self.robot = FR3Robot(robot_conf)
        self.robot.connect()
        self.robot.read_state()
        self.robot._start_read_thread()
        if self.cfg.home:
            self.robot.home()
            self.robot.gripper.open(0.1)
            self.current_gripper_state = 0  # 初始状态为开
        
        def create_camera(name, serial):
            if serial is None:
                return None
            cam_cfg = RealSenseCameraConfig(
                fps=self.cfg.fps,
                width=self.cfg.width,
                height=self.cfg.height,
                buffer_size=self.cfg.camera_buffer,
                serial_number_or_name=serial,
            )
            cam = RealSenseCamera(cam_cfg)
            cam.connect()
            cam._start_read_thread()
            record_devices[name] = cam
            return cam
        
        record_devices = {"fr3": self.robot}
        self.scene_camera = create_camera("scene_camera", self.cfg.scene_camera_id)
        self.wrist_camera = create_camera("wrist_camera", self.cfg.wrist_camera_id)
        return record_devices

    def move(self, action):
        action = self.action_mapping[self.action_mode](action)
        self.robot.send_action(action, asynchronous=self.cfg.asynchronous)

    def update_images(self):
        period = 1.0 / max(1, int(self.cfg.img_update_rate))
        next_t = time.time()
        while not self._stop_event.is_set():
            with self._img_lock:
                if self.scene_camera:
                    if not self.scene_camera.frame_buffer.empty():
                        frame = self.scene_camera.frame_buffer.queue[-1]
                        img = frame.color
                        self.scene_camera_image = cv2.resize(img, (1280, 720))
                        self.scene_camera_depth = getattr(frame, "depth", None)
                if self.wrist_camera:
                    if not self.wrist_camera.frame_buffer.empty():
                        img = self.wrist_camera.frame_buffer.queue[-1].color
                        self.wrist_camera_image = cv2.resize(img, (1280, 720))
            next_t += period
            dt = next_t - time.time()
            time.sleep(max(dt, 0.001))

    # [CLIENT-SMALL-4] tiny depth->point_cloud util for dp3_small backend only
    def _depth_to_point_cloud(self, depth_mm: np.ndarray, n_points: int = 1024):
        if depth_mm is None:
            return np.zeros((n_points, 3), dtype=np.float32)
        depth = depth_mm.astype(np.float32) / 1000.0
        valid = depth > 0
        if not np.any(valid):
            return np.zeros((n_points, 3), dtype=np.float32)
        yy, xx = np.where(valid)
        z = depth[yy, xx]
        fx, fy = 603.340087890625, 603.390869140625
        cx, cy = 329.23199462890625, 249.3147430419922
        x = (xx.astype(np.float32) - cx) / fx * z
        y = (yy.astype(np.float32) - cy) / fy * z
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        if pts.shape[0] >= n_points:
            idx = np.random.choice(pts.shape[0], size=n_points, replace=False)
        else:
            idx = np.random.choice(pts.shape[0], size=n_points, replace=True)
        return pts[idx]
    
    def _to_numpy_uint8_rgb(self, data: np.ndarray) -> np.ndarray:
        """Convert raw image to HxWx3 uint8 RGB."""
        resize_to = (256, 256)

        def _center_crop_pil(pil_img: Image.Image) -> Image.Image:
            w, h = pil_img.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            return pil_img.crop((left, top, left + s, top + s))

        # 处理numpy array
        if data.ndim == 3:
            # data_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(data, 'RGB')
        
        if pil is None:
            return np.zeros((resize_to[0], resize_to[1], 3), dtype=np.uint8)

        # 中心裁剪和缩放
        pil = _center_crop_pil(pil)
        resample = getattr(Image, 'Resampling', Image).BILINEAR
        pil = pil.resize(resize_to, resample)
        pil = pil.convert('RGB')
        
        img = np.asarray(pil, dtype=np.uint8)
        assert img.ndim == 3 and img.shape[2] == 3, f"Shape mismatch: {img.shape}"
        return img
    
    def get_franka_image(self, images):
        """Extracts third-person image from observations and preprocesses it."""
        img = images["scene_image"]
        img = self._to_numpy_uint8_rgb(img)
        return img

    def get_franka_wrist_image(self, images):
        """Extracts wrist camera image from observations and preprocesses it."""
        img = images["wrist_image"]
        img = self._to_numpy_uint8_rgb(img)
        return img
            
    def normalize_gripper_action(self, action: np.ndarray, binarize: bool = True) -> np.ndarray:
        """Convert gripper action from [0,1] to binary [0,1] range"""
        action = np.array(action)
        normalized_action = action.copy()
        
        if binarize:
            # 二值化：>=0.5为关(1)，<0.5为开(0)
            normalized_action[:, -1] = np.where(normalized_action[:, -1] >= 0.9, 1, 0)
        
        return normalized_action

    def prepare_observation(self, images, state):
        """Prepare observation for policy input."""
        img_resized = self.get_franka_image(images)
        wrist_img_resized = self.get_franka_wrist_image(images)
        
        # Concatenate images: base_camera, hand_camera
        # (256, 256, 3) -> (256, 256, 6)
        curr_rgb = np.concatenate([img_resized, wrist_img_resized], axis=-1)
        
        # Construct state
        # Try to get qpos and qvel from state
        try:
            # Assuming state has q and dq attributes for joint positions and velocities
            qpos = np.array(state.joint_positions)
            qvel = np.array(state.joint_velocities)
            gripper_width = np.array([state.gripper_width])
            curr_state = np.concatenate([qpos, gripper_width,gripper_width, qvel]).astype(np.float32)
        except AttributeError:
            # Fallback if q/dq not available, though this might not match training data
            print("Warning: state.joint_positions or state.joint_velocities not found. finished")
           
        return curr_rgb, curr_state

    def process_action(self, action):
        """Process action before sending to environment."""
        action = self.normalize_gripper_action(action, binarize=True)
        action[:, :3] = action[:, :3] / 100
        action[:, 3:-1] =[1e-3, 1e-3, 1e-3]
        return action
    def merge(self, actions_chunk):
        merged_pos = np.sum(actions_chunk[:, :3], axis=0)
        merged_gripper = actions_chunk[-1, 6]
        merged_euler = [1e-3, 1e-3, 1e-3]
        merged_action = np.concatenate([merged_pos, merged_euler, [merged_gripper]])
        return merged_action
    
    def merge_action_sequence(self, action_list, merge_count):
        flag = action_list.shape[0] // merge_count
        merged_actions = []
        for i in range(flag):
            start_idx = i * merge_count
            end_idx = start_idx + merge_count
            action_chunk = action_list[start_idx:end_idx]
            merged_action = self.merge(action_chunk)
            print("merged action:", merged_action)
            merged_actions.append(merged_action)
        return np.stack(merged_actions, axis=0)
    
    def get_action(self, control_hz: int = 10):
        period = 1.0 / max(1, control_hz)
        next_t = time.time()

        while not self._stop_event.is_set():
            with self._img_lock:
                scene_img = self.scene_camera_image.copy() 
                wrist_img = self.wrist_camera_image.copy()
                scene_depth = None if self.scene_camera_depth is None else self.scene_camera_depth.copy()

            if all(x is None for x in [scene_img, wrist_img]):
                time.sleep(0.001)
                continue
            
            images = {
                'scene_image': scene_img,
                "wrist_image": wrist_img,
            }
            
            state = self.robot.read_state()
            curr_rgb, curr_state = self.prepare_observation(images, state)
            
            if getattr(self.client.cfg, "backend", "legacy") == "dp3_small":
                # [CLIENT-SMALL-5] Minimal DP3 protocol: observation.point_cloud + observation.agent_pos
                ee = np.asarray(state.end_effector_position, dtype=np.float32)
                if ee.shape[0] < 7:
                    time.sleep(0.001)
                    continue
                gripper = float(state.gripper_width)
                agent_pos = np.concatenate([ee[:7], np.asarray([gripper], dtype=np.float32)], axis=0).astype(np.float32)
                point_cloud = self._depth_to_point_cloud(scene_depth, int(getattr(self.client.cfg, "dp3_n_points", 1024)))

                self.dp3_obs_buffer["point_cloud"].append(point_cloud)
                self.dp3_obs_buffer["agent_pos"].append(agent_pos)
                H = int(getattr(self.client.cfg, "dp3_obs_horizon", 2))
                if len(self.dp3_obs_buffer["point_cloud"]) > H:
                    self.dp3_obs_buffer["point_cloud"].pop(0)
                    self.dp3_obs_buffer["agent_pos"].pop(0)
                ready = len(self.dp3_obs_buffer["point_cloud"]) == H
                if not ready:
                    next_t += period
                    dt = next_t - time.time()
                    if dt > 0:
                        time.sleep(min(dt, 0.005))
                    continue
                input_obs = {
                    "point_cloud": np.stack(self.dp3_obs_buffer["point_cloud"])[np.newaxis, ...],
                    "agent_pos": np.stack(self.dp3_obs_buffer["agent_pos"])[np.newaxis, ...],
                }
            else:
                # Legacy path unchanged
                self.obs_buffer["rgb"].append(curr_rgb)
                self.obs_buffer["state"].append(curr_state)
                if len(self.obs_buffer["rgb"]) > OBS_HORIZON:
                    self.obs_buffer["rgb"].pop(0)
                    self.obs_buffer["state"].pop(0)
                if len(self.obs_buffer["rgb"]) != OBS_HORIZON:
                    next_t += period
                    dt = next_t - time.time()
                    if dt > 0:
                        time.sleep(min(dt, 0.005))
                    continue
                input_obs = {
                    "rgb": np.stack(self.obs_buffer["rgb"])[np.newaxis, ...],
                    "state": np.stack(self.obs_buffer["state"])[np.newaxis, ...],
                }

            try:
                response = self.client.infer({"observation": input_obs})
                # Diffusion Policy returns actions in shape (Batch, Act_Horizon, Act_Dim)
                actions = response['actions']

                # Take the first batch
                action_batch = actions[0]

                print("raw_action", action_batch[0])
                if getattr(self.client.cfg, "backend", "legacy") == "dp3_small":
                    # [CLIENT-SMALL-6] keep DP3 raw deltas; only clip for safety
                    action_list = np.asarray(action_batch, dtype=np.float32).copy()
                    action_list[:, :3] = np.clip(action_list[:, :3], -0.01, 0.01)
                    action_list[:, 3:6] = np.clip(action_list[:, 3:6], -0.2, 0.2)
                else:
                    action_list = self.process_action(action_batch)
                print("processed action", action_list[0])
                processed_action_list = self.merge_action_sequence(action_list[:8], 4)

                # Execute actions
                # Note: Diffusion Policy predicts a sequence of actions.
                # We can execute the first one, or a few.
                # dp_client_ori.py executed 2 steps.
                # test_client.py just printed.
                # We'll stick to dp_client_ori.py's logic of executing a few steps.

                if action_list is None:
                    print("No action")
                    continue
                elif isinstance(action_list, (list, tuple, np.ndarray)):
                    # Execute first few actions? Or just one?
                    # dp_client_ori.py executed 2.
                    steps_to_execute = 2
                    for i in range(min(len(action_list), steps_to_execute)):
                        try:
                            self.move(list(processed_action_list[i]))
                            print(f"move step {i+1}")
                        except Exception as e:
                            print(f"[get_action] apply action failed: {e}")
                else:
                    try:
                        self.move(list(action_list))
                    except Exception as e:
                        print(f"[get_action] apply action failed: {e}")

            except Exception as e:
                print(f"Inference failed: {e}")
            
            next_t += period
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.005))
            else:
                next_t = time.time()

    def run(self, seconds: float = None):
        t_img = threading.Thread(target=self.update_images, name="t_update_images", daemon=True)
        t_ctl = threading.Thread(target=self.get_action, name="t_get_action", daemon=True)
        self._threads = [t_img, t_ctl]

        for t in self._threads:
            t.start()

        start = time.time()
        try:
            while True:
                if seconds is not None and time.time() - start > seconds:
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt...")
        finally:
            self._stop_event.set()
            for t in self._threads:
                t.join(timeout=1.0)
            for cam in [self.scene_camera, self.wrist_camera]:
                try:
                    if cam: cam.disconnect()
                except Exception:
                    pass
            try:
                if self.robot: self.robot.disconnect()
            except Exception:
                pass

if __name__ == "__main__":
    franka_client = shw_franka(FR3Config(), TaskConfig())
    franka_client.run(seconds=600)
