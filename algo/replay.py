import sys
from pathlib import Path
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 找到项目根目录 (假设 algo/ 在项目根目录下，所以是 .parent.parent)
project_root = current_file.parent.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(project_root))
import cv2
import time
import threading
import requests
import io
import signal
import pandas as pd
import numpy as np
import contextlib
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union
from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig, FR3RobotAction, FR3ActionMode
from utils import image_tools
from utils import websocket_client_policy
from PIL import Image
from termcolor import colored

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
    scene_camera_id: Optional[str] = "938422072347" # left camera
    right_camera_id: Optional[str] = None # front camera
    wrist_camera_id: Optional[str] = "112322074840" # wrist camera
    fps: int = 15
    width: int = 1280
    height: int = 720
    camera_buffer: int = 5

    # --- control mode ---
    action_mode: str = "POSITION_BASE_DELTA"  # ["POSITION_DELTA", "JOINT_DELTA", "POSITION_ABSOLUTE", "JOINT_ABSOLUTE", "POSITION_BASE_DELTA"]
    img_update_rate: int = 15            
    asynchronous: bool = False           # move asynchronous
    action_chunk: int = 5                # move step num


class shw_franka:
    def __init__(self, robot_cfg, action_list = None, state_list=None):
        self.gripper_state = 0  # 0: open, 1: closed
        self.cfg = robot_cfg
        self.action_mode = robot_cfg.action_mode
        self.action_mapping = {
            "POSITION_DELTA": self.delta_position_action_mapping,
            "JOINT_DELTA": self.delta_joint_action_mapping,
            "POSITION_ABSOLUTE": self.absolute_position_action_mapping,
            "JOINT_ABSOLUTE": self.absolute_joint_action_mapping,
            "POSITION_BASE_DELTA" : self.delta_absolute_position_action_mapping,

        }
        self.robot = None
        self.scene_camera = None
        self.right_camera = None
        self.wrist_camera = None
        self.scene_camera_id = robot_cfg.scene_camera_id
        self.right_camera_id = robot_cfg.right_camera_id
        self.wrist_camera_id = robot_cfg.wrist_camera_id
        self.scene_camera_image = None
        self.right_camera_image = None
        self.wrist_camera_image = None
        self._img_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads = []
        self.devices = self.setup_hardwares()
        defined_ids = [self.cfg.scene_camera_id, self.cfg.right_camera_id, self.cfg.wrist_camera_id]
        self.camera_count = sum(1 for cid in defined_ids if cid is not None)
        self.actions = action_list
        self.states = state_list


    def setup_hardwares(self):
        robot_conf = FR3RobotConfig(
            id=self.cfg.robot_id,
            robot_ip=self.cfg.robot_ip,
            load_gripper=self.cfg.load_gripper,
            relative_dynamics_factor=self.cfg.relative_dynamics_factor,
            buffer_size=self.cfg.buffer_size,
            initial_joint = None,
            # initial_end_pose=[0.4707543519985666, -0.02776023399946076, 0.29950666236991815, 1.0, 0.0, 0.0, 0.0] #### pick and place task 251212 by chenxinzhe
            initial_end_pose = [0.3207543519985666, 0.12776023399946076, 0.29950666236991815, -1.0, 0.0, 0.0, 0.0] #### long horizon task 251214 by chenxinzhe 
        )
        self.robot = FR3Robot(robot_conf)
        self.robot.connect()
        self.robot.read_state()
        self.robot._start_read_thread()
        if self.cfg.home:
            self.robot.home()
            self.robot.gripper.open(0.1)
            print(colored("Sucess move fr3 back to initial position", "green"))
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
        self.right_camera = create_camera("right_camera", self.cfg.right_camera_id)
        self.wrist_camera = create_camera("wrist_camera", self.cfg.wrist_camera_id)
        return record_devices

    def move(self, action, states):
        action = self.action_mapping[self.action_mode](action, states)
        self.robot.send_action(action, asynchronous=self.cfg.asynchronous)

    def update_images(self):
        period = 1.0 / max(1, int(self.cfg.img_update_rate))
        next_t = time.time()
        while not self._stop_event.is_set():
            with self._img_lock:
                if self.scene_camera:
                    if not self.scene_camera.frame_buffer.empty():
                        img = self.scene_camera.frame_buffer.queue[-1].color
                        self.scene_camera_image = cv2.resize(img, (1280, 720))
                if self.right_camera:
                    if not self.right_camera.frame_buffer.empty():
                        img = self.right_camera.frame_buffer.queue[-1].color
                        self.right_camera_image = cv2.resize(img, (320, 180))
                if self.wrist_camera:
                    if not self.wrist_camera.frame_buffer.empty():
                        img = self.wrist_camera.frame_buffer.queue[-1].color
                        self.wrist_camera_image = cv2.resize(img, (1280, 720))
            next_t += period
            dt = next_t - time.time()
            time.sleep(max(dt, 0.001))

    def preprocess_observation(self, images, state):
        observations = {}
        observations['scene_image'] = images["scene_image"]
        observations['hand_image'] = images["wrist_image"]
        observations['right_image'] = images["right_image"]
        # observations['end_effector'] = np.array(state.end_effector_position)
        observations['end_effector'] = np.array(state)
        # observations['joint'] = np.array(state.joint_positions)
        # observations['gripper'] = np.array([state.gripper_width])
        return observations

    def get_action(self, control_hz: int = 10):
        period = 1.0 / max(1, control_hz)
        next_t = time.time()
        while not self._stop_event.is_set():
            with self._img_lock:
                    scene_img = self.scene_camera_image.copy()
                    wrist_img = self.wrist_camera_image.copy()
                    right_img = self.wrist_camera_image.copy()

            if all(x is None for x in [scene_img, wrist_img, right_img]):
                time.sleep(0.001)
                continue
            images = {
                'scene_image': scene_img,
                "wrist_image": wrist_img,
                "right_image": right_img
            }
            state = self.robot.read_state()
            # state_list = self.states
            observations = self.preprocess_observation(images, state)
            action_list = self.actions
            print(action_list)
            # action_list = [[0,0,0.01,0,0,0,0]]
            if action_list is None:
                print("No action")
                continue
            elif isinstance(action_list[0], (list, tuple, np.ndarray)):
                for i in range(len(action_list)):
                    try:
                        print(list(action_list[i]))
                        self.move(list(action_list[i]), None)
                        print("move once")
                    except Exception as e:
                        print(f"[get_action 0] apply action failed: {e}")
            else:
                try:
                    self.move(list(action_list))
                except Exception as e:
                    print(f"[get_action 1] apply action failed: {e}")
            self.actions = None 
            print("All actions executed. Action list cleared.")
            print("tcp_state", np.array(state))
            time.sleep(1)
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
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt...")
        finally:
            self._stop_event.set()
            for t in self._threads:
                t.join(timeout=1.0)
            for cam in [self.scene_camera, self.right_camera, self.wrist_camera]:
                try:
                    if cam: cam.disconnect()
                except Exception:
                    pass
            try:
                if self.robot: self.robot.disconnect()
            except Exception:
                pass

    def delta_position_action_mapping(self, action):
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.DELTA)
        if len(action) != 7:
            raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
        translation = np.array(action[:3])
        rotation = np.array(action[3:-1])
        rotation = R.from_euler("xyz", rotation, degrees=False).as_quat()
        gripper = np.array(action[-1])
        positions = np.concatenate([translation, rotation, [gripper]]).tolist()
        return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.DELTA)

    def delta_joint_action_mapping(self, action):
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.DELTA)
        if len(action) != 8:
            raise ValueError(f"Expected action length 8, got {len(action)}")
        joint_position = np.array(action)
        return FR3RobotAction(joint_positions=joint_position, action_mode=FR3ActionMode.DELTA)
    
    def absolute_position_action_mapping(self, action):
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.ABSOLUTE)
        if len(action) != 7:
            raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
        translation = np.array(action[:3])
        rotation = np.array(action[3:-1])
        rotation = R.from_euler("xyz", rotation, degrees=False).as_quat()
        gripper = np.array(action[-1])
        positions = np.concatenate([translation, rotation, [gripper]]).tolist()
        return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)
    
    def absolute_joint_action_mapping(self, action):
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.ABSOLUTE)
        if len(action) != 8:
            raise ValueError(f"Expected action length 8, got {len(action)}")
        joint_position = np.array(action)
        return FR3RobotAction(joint_positions=joint_position, action_mode=FR3ActionMode.ABSOLUTE)
    
    def delta_absolute_position_action_mapping(self, action, state):
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.ABSOLUTE)
        if len(action) != 7:
            raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
        state = self.robot.read_state()
        # tcp_state = np.array(state)
        tcp_state = np.array(state.end_effector_position)
        print("tcp_state", tcp_state)
        print("actions", action)
        translation = np.array(action[:3]) + tcp_state[:3]
        # 1. 获取当前的姿态 (Rotation 对象)
        # 注意：Scipy 的 from_quat 需要 [x, y, z, w] 格式
        # 假设 tcp_state[[4, 5, 6, 3]] 已经是正确的 [x, y, z, w] 顺序
        current_rot = R.from_quat(tcp_state[[3, 4, 5, 6]])

        # 2. 将 action 中的欧拉角增量转换为 Rotation 对象
        # 假设 action 中的 rpy 是相对于当前 TCP 自身的旋转 (Local Frame)
        delta_rot = R.from_euler("xyz", action[3:-1], degrees=False)

        # 3. 执行旋转合成 (乘法)
        # 如果是局部坐标系旋转 (Local): new = current * delta
        # 如果是全局坐标系旋转 (Global): new = delta * current
        new_rot = current_rot * delta_rot

        # 4. 转回四元数并调整顺序 [w, x, y, z]
        # Scipy as_quat 返回 [x, y, z, w]，你需要 [3, 0, 1, 2] -> [w, x, y, z]
        rotation = new_rot.as_quat()[[3, 0, 1, 2]]
        # rotation = np.array(action[3:-1]) + R.from_quat(tcp_state[[4, 5, 6, 3]]).as_euler("xyz", degrees=False)
        # rotation = R.from_euler("xyz", rotation, degrees=False).as_quat()[[3, 0, 1, 2]]
        # rotation = R.from_euler("xyz", rotation, degrees=False).as_quat()
        gripper = np.array(action[-1])
        positions = np.concatenate([translation, np.array([1,0,0,0]), [gripper]]).tolist()
        print("positions", positions)
        return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)

def _clip_and_scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action

def clip_and_scale_action(action):
    pos_action = _clip_and_scale_action(
        action, 
        np.array([-0.1, -0.1, -0.1, 0 , 0, 0, 0]), 
        np.array([0.1, 0.1, 0.1, 0 , 0, 0, 0])
    )
    return pos_action
if __name__ == "__main__":
    import pickle
    data = np.load("/home/server/franka/hw_roby/test.npy", allow_pickle = True)
    collect_action = np.array(data)
    actions_list = clip_and_scale_action(collect_action).tolist()
    
    # data = np.load("/home/server/franka/roby/long_horizon_banana_numpy_251214/episode_00000/episode.npy", allow_pickle = True)
    # steps = data.item()['steps']
    # actions_list = []
    # state_list = []
    # for step in steps:
    #     act = step['action']
    #     act[:3] = act[:3] / 100.0
    #     act = act.tolist() 
    #     actions_list.append(act)
    #     state = step['end_effector_position']
    #     state = state.tolist()
    #     state_list.append(state)
    # has_triggered = False 

    # actions_list = []

    # for step in steps:
    #     # 1. 获取并复制数据（防止修改原数据）
    #     act = step['action']
    #     act = act.copy()
        
    #     # 2. 你的缩放操作
    #     act[:3] = act[:3] / 100.0
        
    #     # 3. 【核心修改】控制夹爪只出现一次 1
    #     if not has_triggered:
    #         # 如果还没触发过，设为 1，并标记为“已触发”
    #         if act[-1] == 1.0:
    #             act[-1] = 1.0
    #             has_triggered = True
    #     else:
    #         # 如果已经触发过了，剩下的全部设为 0
    #         act[-1] = 0.0
            
    #     actions_list.append(act.tolist())
    print(actions_list)
    franka_client = shw_franka(FR3Config(), action_list= actions_list, state_list = None)
    franka_client.run(seconds=600)