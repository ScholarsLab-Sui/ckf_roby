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
# @contextlib.contextmanager
# def prevent_keyboard_interrupt():
#     """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
#     interrupted = False
#     original_handler = signal.getsignal(signal.SIGINT)

#     def handler(signum, frame):
#         nonlocal interrupted
#         interrupted = True

#     signal.signal(signal.SIGINT, handler)
#     try:
#         yield
#     finally:
#         signal.signal(signal.SIGINT, original_handler)
#         if interrupted:
#             raise KeyboardInterrupt

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
    right_camera_id: Optional[str] = None
    wrist_camera_id: Optional[str] = "112322074840"
    fps: int = 15
    width: int = 640
    height: int = 480
    camera_buffer: int = 5

    # --- control mode ---
    action_mode: str = "POSITION_DELTA"  # ["POSITION_DELTA", "JOINT_DELTA", "POSITION_ABSOLUTE", "JOINT_ABSOLUTE"]
    img_update_rate: int = 15            
    asynchronous: bool = False           # move asynchronous
    action_chunk: int = 5                # move step num


@dataclass
class TaskConfig:
    task_prompt: str = "pick up the cube"
    algorithm: str = "act"
    is_online: bool = True
    is_fastapi: bool = False
    # server_host: str = "http://10.184.17.177"
    server_host: str = "10.184.17.177"
    server_port: int = 8059
    server_app: str = "act"

    def __repr__(self):
        return f"<TaskConfig {self.algorithm} | online={self.is_online}>"

class RobotClient:
    _action_registry: dict[str, callable] = {}

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.is_fastapi:
            self.server_url = cfg.server_host + ":" + str(cfg.server_port) + "/" + cfg.server_app
        else:
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(cfg.server_host, cfg.server_port)
    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._action_registry[name.lower()] = func
            return func
        return decorator



    def get_action(self, img_resized, wrist_img_resized, gripper_state):
        obs = {"left_image": img_resized, "wrist_image": wrist_img_resized, "qpos": gripper_state}
        actions = self.policy_client.infer(obs)["actions"]
        return actions

    # @register("mine_pi0")
    # def _get_mine_pi0_action(self, observations):
    #     scene_image = observations['scene_image']
    #     right_image = observations['right_image']
    #     hand_image = observations['hand_image']
    #     joint = observations['joint']
    #     gripper = observations['gripper']
    #     task_prompt = "Pick up apple"
    #     if self.cfg.is_online:
    #         action_list = self.send_two_observation(scene_image, hand_image, joint, gripper, task_prompt)
    #     else:
    #         raise NotImplementedError()
    #     return action_list

    # @register("pi0")
    def _get_pi0_action(self, observations):
        scene_image = observations['scene_image']
        right_image = observations['right_image']
        hand_image = observations['hand_image']
        joint = observations['joint']
        gripper = observations['gripper']
        task_prompt = "Pick up apple"
        if self.cfg.is_online:
            action_list = self.send_three_observation(scene_image, right_image, hand_image, joint, gripper, task_prompt)
        else:
            raise NotImplementedError()
        return action_list

    def send_three_observation(self, scene_image: np.ndarray, right_image: np.ndarray, hand_image: np.ndarray, joint: np.ndarray, gripper: np.ndarray, task_prompt):
        # cv2.imshow("scene_image", scene_image)
        # cv2.imshow("hand_image", hand_image)
        def ensure_bgr(image):
            if image.shape[-1] == 3 and image.dtype == np.uint8:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image[0,0,0] < 256 else image
            return image
        def normalize_to_uint8(image):
            if image.dtype == np.float32 or image.dtype == np.float64:
                # 若数据范围是0-1，则缩放至0-255
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                # 若数据范围是0-255的浮点数，直接转换
                else:
                    image = image.astype(np.uint8)
            return image

        # 处理后再显示
        scene_bgr = ensure_bgr(scene_image)
        hand_bgr = ensure_bgr(hand_image)
        scene_uint8 = normalize_to_uint8(scene_bgr)
        hand_uint8 = normalize_to_uint8(hand_bgr)

        hand_uint8 = cv2.flip(hand_uint8, flipCode=1) 
        
        combined_image = np.hstack([scene_uint8, hand_uint8])
        
        # 6. 显示拼接后的图像（单窗口管理）
        cv2.imshow("Three Observation Visualization (Scene | Hand)", combined_image)
        
        # 7. 窗口控制（按'q'退出，避免程序卡死）
        key = cv2.waitKey(1) & 0xFF  # 1ms刷新，兼容实时流
        request_data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(scene_uint8, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(hand_uint8, 224, 224),
            "observation/joint_position": joint,
            "observation/gripper_position": gripper,
            "prompt": task_prompt,
        }

        # with prevent_keyboard_interrupt():
        pred_action_chunk = self.policy_client.infer(request_data)["actions"]
        assert pred_action_chunk.shape == (10, 8)
        return pred_action_chunk

    def send_two_observation(self, scene_image: np.ndarray, hand_image: np.ndarray, joint: np.ndarray, gripper: np.ndarray, task_prompt):
        _, scene_image_encoded = cv2.imencode('.jpg', scene_image)
        scene_image_bytes = io.BytesIO(scene_image_encoded.tobytes())
        _, hand_image_encoded = cv2.imencode('.jpg', hand_image)
        hand_image_bytes = io.BytesIO(hand_image_encoded.tobytes())
        joint_bytes = io.BytesIO()
        np.save(joint_bytes, joint)
        joint_bytes.seek(0)
        gripper_bytes = io.BytesIO()
        np.save(gripper_bytes, gripper)
        gripper_bytes.seek(0)

        files = {
            "scene_image_file": ("scene.jpg", scene_image_bytes, "image/jpeg"),
            "hand_image_file": ("hand.jpg", hand_image_bytes, "image/jpeg"),
            "joint_file": joint_bytes,
            "gripper_file": gripper_bytes,
        }
        
        data = {
            "task_prompt": task_prompt
        }

        response = requests.post(self.server_url, files=files, data=data)

        if response.status_code == 200:
            action = response.json()
            return action['action']
        else:
            print("Failed to get a valid response from server. Status code:", response.status_code)


class shw_franka:
    def __init__(self, robot_cfg, task_cfg):
        self.gripper_state = 0  # 0: open, 1: closed
        self.cfg = robot_cfg
        self.action_mode = robot_cfg.action_mode
        self.action_mapping = {
            # "POSITION_DELTA": self.delta_position_action_mapping,
            "POSITION_DELTA": self.delta_absolute_position_action_mapping,
            "JOINT_DELTA": self.delta_joint_action_mapping,
            "POSITION_ABSOLUTE": self.absolute_position_action_mapping,
            "JOINT_ABSOLUTE": self.absolute_joint_action_mapping,
        }
        self.robot = None
        self.gripper_state = np.array([0])
        self.client = RobotClient(task_cfg)
        self.current_gripper_state = 0
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


    def setup_hardwares(self):
        robot_conf = FR3RobotConfig(
            id=self.cfg.robot_id,
            robot_ip=self.cfg.robot_ip,
            load_gripper=self.cfg.load_gripper,
            relative_dynamics_factor=self.cfg.relative_dynamics_factor,
            buffer_size=self.cfg.buffer_size,
            initial_joint = None,
            initial_end_pose=[0.4707543519985666, -0.02776023399946076, 0.29950666236991815, 1.0, 0.0, 0.0, 0.0] #### pick and place task 251212 by chenxinzhe
        )
        self.robot = FR3Robot(robot_conf)
        self.robot.connect()
        self.robot.read_state()
        self.robot._start_read_thread()
        if self.cfg.home:
            self.robot.home()
            self.robot.gripper.open(0.1)
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
                        img = self.scene_camera.frame_buffer.queue[-1].color
                        self.scene_camera_image = cv2.resize(img, (640, 480))
                if self.right_camera:
                    if not self.right_camera.frame_buffer.empty():
                        img = self.right_camera.frame_buffer.queue[-1].color
                        self.right_camera_image = cv2.resize(img, (320, 180))
                if self.wrist_camera:
                    if not self.wrist_camera.frame_buffer.empty():
                        img = self.wrist_camera.frame_buffer.queue[-1].color
                        self.wrist_camera_image = cv2.resize(img, (640, 480))
            next_t += period
            dt = next_t - time.time()
            time.sleep(max(dt, 0.001))
            
    # def _to_numpy_uint8_rgb(self, data: Any) -> Any:
    #     """Convert raw image to HxWx3 uint8 RGB.
    #     Steps:
    #     - Decode bytes (prefer OpenCV). If OpenCV used: BGR->RGB.
    #     - Center-crop to 720x720.
    #     - Resize to 256*256.
    #     - Return numpy uint8 RGB array.
    #     """
    #     resize_to = (256, 256)

    #     def _center_crop_pil(pil_img: Image.Image) -> Image.Image:
    #         w, h = pil_img.size
    #         s = min(w, h)
    #         left = (w - s) // 2
    #         top = (h - s) // 2
    #         return pil_img.crop((left, top, left + s, top + s))

    #     import cv2  # type: ignore
    #     rgb = data
    #     if rgb is not None:
    #         # if rgb.shape[-1] == 3 and rgb.dtype == np.uint8:
    #         #     rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb[0,0,0] < 256 else rgb
    #         pil = Image.fromarray(rgb, 'RGB')
    #         # pil.save('image.png')
    #         pil = _center_crop_pil(pil)
    #         resample = getattr(Image, 'Resampling', Image).BILINEAR
    #         pil = pil.resize(resize_to, resample)
    #         # pil.save("resized_image.png")
    #     img = np.asarray(pil, dtype=np.uint8)
           
    #     return img     

    def _to_numpy_uint8_rgb(self, data: Any) -> np.ndarray:
        """Convert raw image to HxWx3 uint8 RGB.
        Steps:
        - Handle Input (Bytes or Numpy).
        - Center-crop.
        - Resize to 256*256.
        - Return numpy uint8 RGB array with shape (H, W, 3).
        """
        resize_to = (256, 256)

        def _center_crop_pil(pil_img: Image.Image) -> Image.Image:
            w, h = pil_img.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            return pil_img.crop((left, top, left + s, top + s))

        # 1. 数据读取与预处理 (处理 Bytes 或 Numpy)
        pil = None
        if isinstance(data, bytes):
            # 如果是 bytes，优先使用 OpenCV 解码 (更快)
            img_array = np.frombuffer(data, np.uint8)
            img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # BGR
            if img_cv2 is not None:
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
        elif isinstance(data, np.ndarray):
            # 如果已经是 numpy array
            if data.ndim == 2: # 灰度图 (H, W)
                pil = Image.fromarray(data, 'L')
            elif data.ndim == 3: # (H, W, C)
                # 假设输入是 RGB，如果是 BGR 需要在这里转换，或者由外部保证
                pil = Image.fromarray(data, 'RGB')
        
        # 防止 data 为 None 或解码失败
        if pil is None:
            # 这里需要定义失败时的行为，例如返回全黑图或抛出异常
            # 此时返回一个全黑的 256x256x3
            return np.zeros((resize_to[0], resize_to[1], 3), dtype=np.uint8)

        # 2. 中心裁剪
        pil = _center_crop_pil(pil)

        # 3. 缩放
        resample = getattr(Image, 'Resampling', Image).BILINEAR
        pil = pil.resize(resize_to, resample)

        # 4. 【关键步骤】强制转换为 RGB 模式
        # 这步保证了无论之前是灰度('L')还是透明('RGBA')，结果都会变成 3 通道
        pil = pil.convert('RGB')

        # 5. 转为 Numpy
        img = np.asarray(pil, dtype=np.uint8)

        # 6. 【双重保险】断言形状 (可选，开发调试用)
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
        """
        Normalize gripper action from [0,1] to [-1,+1] range

        Args:
            action: Action array with gripper action in the last dimension
            binarize: Whether to binarize gripper action to 0 or 1

        Returns:
            np.ndarray: Action array with normalized gripper action
        """
        # Create a copy to avoid modifying the original
        action = np.array(action)
        normalized_action = action.copy()

        if binarize:
            # Binarize to 0 or 1
            # normalized_action[-1] = 1.0 if normalized_action[-1] >= 0.1 else 0.0
            normalized_action[:, -1] = np.where(normalized_action[:, -1] >= 0.95, 1, 0)

        return normalized_action
    

    def prepare_observation(self, images, state):
        """Prepare observation for policy input."""
        # Get preprocessed images
        img_resized = self.get_franka_image(images)
        wrist_img_resized = self.get_franka_wrist_image(images)
        # img_resized = img_resized[:, :, ::-1]
        # wrist_img_resized = wrist_img_resized[:, :, ::-1]
        # cv2.imwrite("cv2_image.jpeg", img_resized)
        # cv2.imwrite("cv2_wrist_image.jpeg", wrist_img_resized)
        # pil_img_resized = Image.fromarray(img_resized)
        # pil_img_resized.save("pil_img.jpeg")
        # pil_wrist_img = Image.fromarray(wrist_img_resized)
        # pil_wrist_img.save("pil_wrist_image.jpeg")
        state = np.array(state.joint_positions)
        print("state:", state)
        # Prepare observations dict

        return img_resized, wrist_img_resized, state # Return both processed observation and original image for replay

    def process_action(self, action):
        """Process action before sending to environment."""
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        action = self.normalize_gripper_action(action, binarize=True)
        print(f"action is {action}")
        action[:, :3] = action[:, :3] / 100
        action[:, 3:-1] =[1e-3, 1e-3, 1e-3]
        # action[..., -1] = 0.0
        return action

    # def preprocess_observation(self, images, state):
    #     observations = {}
    #     observations['scene_image'] = images["scene_image"]
    #     observations['hand_image'] = images["wrist_image"]
    #     observations['right_image'] = images["right_image"]
    #     observations['end_effector'] = np.array(state.end_effector_position)
    #     observations['joint'] = np.array(state.joint_positions)
    #     observations['gripper'] = np.array([state.gripper_width])
    #     return observations

    def get_action(self, control_hz: int = 10):
        period = 1.0 / max(1, control_hz)
        next_t = time.time()
        test = True
        while not self._stop_event.is_set():
            with self._img_lock:
                scene_img = self.scene_camera_image.copy() 
                wrist_img = self.wrist_camera_image.copy()

            if all(x is None for x in [scene_img, wrist_img]):
                time.sleep(0.001)
                continue
            images = {
                'scene_image': scene_img,
                "wrist_image": wrist_img,
            }
            state = self.robot.read_state()
            img_resized, wrist_img_resized, joint_state = self.prepare_observation(images, state)
            # img_resized = img_resized[:, :, ::-1]
            # wrist_img_resized = wrist_img_resized[:, :, ::-1]
            print("Hereee")
            # time.sleep(1)
            if test:
                self.client.get_action(img_resized, wrist_img_resized, joint_state)
                test = False
            action_list = self.client.get_action(img_resized, wrist_img_resized, joint_state)
            
            print("raw_action", action_list)
            action_list = self.process_action(action_list)
            print("processed action", action_list)
            # mode_key = str(self.action_mode).upper()
            # if mode_key in ("POSITION_DELTA", "POSITION_ABSOLUTE"):
            #     action_list = [0.0, -0.1, 0.0, 0.0, 0.0, 0.0,   0.0]
            # else:
            #     action_list = [0.0]*6 + [0.2] + [0.0]
            if action_list is None:
                print("No action")
                continue
            elif isinstance(action_list[0], (list, tuple, np.ndarray)):
                for i in range(2):
                    try:
                        self.move(list(action_list[i]))
                        print("move once")
                    except Exception as e:
                        print(f"[get_action] apply action failed: {e}")
            else:
                try:
                    self.move(list(action_list))
                except Exception as e:
                    print(f"[get_action] apply action failed: {e}")
            # time.sleep(0.5)
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
        rotation = R.from_euler("xyz", rotation, degrees=True).as_quat()
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
        rotation = R.from_euler("xyz", rotation, degrees=True).as_quat()
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

    def delta_absolute_position_action_mapping(self, action):
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 0], action_mode=FR3ActionMode.ABSOLUTE)
        if len(action) != 7:
            raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
        
        state = self.robot.read_state()
        tcp_state = np.array(state.end_effector_position)
        
        print("tcp_state", tcp_state)
        print("actions", action)
        
        # 计算新的位置
        translation = np.array(action[:3]) + tcp_state[:3]
        rotation = np.array([1,0,0,0])
        
        # 获取策略要求的夹爪目标状态（0或1）
        target_state = action[-1]
        
        # 只有当目标状态与当前状态不同时，才发送夹爪动作
        gripper_cmd = 0  # 默认保持当前状态
        
        if target_state != self.current_gripper_state:
            if target_state == 1:
                # 需要关闭夹爪（从开到关）
                gripper_cmd = 1
                self.current_gripper_state = 1
                print("Closing gripper (open -> closed)")
            elif target_state == 0:
                # 需要打开夹爪（从关到开）
                gripper_cmd = -1
                self.current_gripper_state = 0
                print("Opening gripper (closed -> open)")
        else:
            gripper_cmd = 0
        
        positions = np.concatenate([translation, rotation, [gripper_cmd]]).tolist()
        print("positions", positions)
        
        return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)
    
if __name__ == "__main__":
    franka_client = shw_franka(FR3Config(), TaskConfig())
    franka_client.run(seconds=600)