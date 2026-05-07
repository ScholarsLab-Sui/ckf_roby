
import os
from pathlib import Path
import sys
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 找到项目根目录 (假设 algo/ 在项目根目录下，所以是 .parent.parent)
project_root = current_file.parent.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(project_root))
import streamlit as st
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from collections import deque
from typing import Optional
from dataclasses import asdict, is_dataclass

import time
import numpy as np
import uuid
import pandas as pd
import threading
from scipy.spatial.transform import Rotation as R

import cv2
import plotly.graph_objects as go

from roby.hardware.teleoperators.spacemouse.teleop_spacemouse import SpacemouseTeleopConfig, SpacemouseTeleop, SpacemouseOutput
from roby.common.recorder import EpisodeRecorder
from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig, FR3RobotAction, FR3ActionMode

# =========================================================
# Streamlit app title
# =========================================================

# st.title("Roby: Robot Control and Visualization")
st.set_page_config("Roby", layout="wide")
USE_WRIST_CAMERA = True
USE_RIGHT_CAMERA = False
USE_LEFT_CAMERA = True
USE_FRONT_CAMERA = False
USE_DEPTH = True
# =========================================================
# Streamlit app configuration
# =========================================================

if "recording" not in st.session_state:
    st.session_state.recording = False
if "episode_id" not in st.session_state:
    st.session_state.episode_id = 0
if "recorder" not in st.session_state:
    st.session_state.recorder = None
if "fps_image" not in st.session_state:
    st.session_state.fps_image = 0.0
if "fps_state" not in st.session_state:
    st.session_state.fps_state = 0.0
if "last_timestep" not in st.session_state:
    st.session_state.last_timestep = 0
if "robot" not in st.session_state:
    st.session_state.robot = None
if "image_fps_fig" not in st.session_state:
    st.session_state.image_fps_fig = None
if "state_fps_fig" not in st.session_state:
    st.session_state.state_fps_fig = None
if "joint_states" not in st.session_state:
    st.session_state.joint_states = None
if "line_fig" not in st.session_state:
    st.session_state.line_fig = None  # Placeholder for line chart figure
if "left_image" not in st.session_state:
    st.session_state.left_image = None
if "front_image" not in st.session_state:
    st.session_state.front_image = None
if "wrist_image" not in st.session_state:
    st.session_state.wrist_image = None
if "right_image" not in st.session_state:
    st.session_state.right_image = None
if "left_image_depth" not in st.session_state:
    st.session_state.left_image_depth = None
if "front_image_depth" not in st.session_state:
    st.session_state.front_image_depth = None
if "wrist_image_depth" not in st.session_state:
    st.session_state.wrist_image_depth = None
if "right_image_depth" not in st.session_state:
    st.session_state.right_image_depth = None


# =========================================================
# Set up hardwares
# =========================================================
    
# def action_mapping(action: Optional[SpacemouseOutput], translation_factor: Optional[float] = 0.04, rotation_factor: Optional[float] = 10.0) -> FR3RobotAction:
#     """Maps Spacemouse actions to FR3RobotAction."""
#     if action is None:
#         return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.DELTA)
            
#     translation = np.array(action.offsets[:3]) * np.array([-1, 1, -1])
#     rotation = np.array(action.offsets[3:]) * np.array([-1, 1, -1])
#     if translation_factor is not None:
#         translation *= translation_factor
#     if rotation_factor is not None:
#         rotation *= rotation_factor
#     rotation = np.array([0,0,0]) # 251104 set 3-dof
#     rotation = R.from_euler("xyz", rotation, degrees=True).as_quat()
    
#     gripper = 0
#     if action.left_button and not action.right_button:
#         gripper = 1
#     elif action.right_button and not action.left_button:
#         gripper = -1
#     positions = np.concatenate([translation, rotation, [gripper]]).tolist()

#     return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.DELTA)

### keep rotation z, modified by chenxinzhe 251211
def action_mapping(action: Optional[SpacemouseOutput], translation_factor: Optional[float] = 0.04, rotation_factor: Optional[float] = 10.0) -> FR3RobotAction:
    """Maps Spacemouse actions to FR3RobotAction."""
    if action is None:
        return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.DELTA)
            
    # 1. 处理平移 (XYZ)
    translation = np.array(action.offsets[:3]) * np.array([-1, 1, -1])
    if translation_factor is not None:
        translation *= translation_factor

    # 2. 处理旋转 (只保留 Z 轴)
    # 获取原始输入并调整方向 (根据您之前的 np.array([-1, 1, -1]) 逻辑)
    raw_rotation = np.array(action.offsets[3:]) * np.array([-1, 1, -1])
    
    if rotation_factor is not None:
        raw_rotation *= rotation_factor

    # # raw_rotation 通常是 [rot_x, rot_y, rot_z]
    # # 强制将 y 设为 0，只保留x, z 的值
    # x_rotation = raw_rotation[0]
    # z_rotation = raw_rotation[2] 
    # rotation_euler = np.array([x_rotation, 0.0, z_rotation]) #

    # # 强制将 y,z设为 0，只保留x的值
    # z_rotation = raw_rotation[2] 
    # rotation_euler = np.array([0.0, 0.0, z_rotation]) #

    # # 强制将 x 和 z 设为 0，只保留 y 的值
    # y_rotation = raw_rotation[1] 
    # rotation_euler = np.array([0.0, y_rotation, 0.0]) # [x=0, y=controlled, z=0]

    # # 强制将 z 设为 0，只保留x, y 的值
    # x_rotation = raw_rotation[0]
    # y_rotation = raw_rotation[1]
    # rotation_euler = np.array([x_rotation, y_rotation, 0.0]) # [x=0, y=controlled, z=0]


    # 强制将 x 设为 0，只保留y, z 的值
    y_rotation = raw_rotation[1]
    z_rotation = raw_rotation[2]
    rotation_euler = np.array([0.0, y_rotation,z_rotation])

    # # 保留x,y,z 的值
    # rotation_euler = raw_rotation[:3] 

    # 3. 转为四元数
    # SciPy 的 from_euler 默认顺序通常是 "xyz" (如果是外旋) 或 "XYZ" (如果是内旋)
    # 这里假设是标准欧拉角顺序
    rotation_quat = R.from_euler("xyz", rotation_euler, degrees=True).as_quat()
    
    # 4. 处理夹爪
    gripper = 0
    if action.left_button and not action.right_button:
        gripper = 1
    elif action.right_button and not action.left_button:
        gripper = -1
        
    positions = np.concatenate([translation, rotation_quat, [gripper]]).tolist()

    return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.DELTA)

@st.cache_resource
def setup_hardwares():
    config = FR3RobotConfig(id="fr3", robot_ip="172.16.0.2", load_gripper=True, relative_dynamics_factor=0.05, buffer_size=10)
    robot = FR3Robot(config)
    robot.connect()
    robot.read_state()
    state = robot.read_state()
    print("FR3 Robot connected. Current joint positions:", state.joint_positions)
    robot._start_read_thread()
    robot.home()
    
    teleop_config = SpacemouseTeleopConfig()
    teleop = SpacemouseTeleop(teleop_config)
    teleop.connect()
    robot.start_teleoperation(teleoperator=teleop, action_mapping=action_mapping)
    #shw 112322077048
    #cxz 938422072347
    if USE_LEFT_CAMERA:
        left_camera_config = RealSenseCameraConfig(
            fps=15, width=640, height=480, buffer_size=5, serial_number_or_name="938422072347", use_depth=USE_DEPTH
        )
    if USE_FRONT_CAMERA:
        front_camera_config = RealSenseCameraConfig(
            fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="233522073398", use_depth=USE_DEPTH
        )
    # if USE_FRONT_CAMERA:
    #     front_camera_config = RealSenseCameraConfig(
    #         fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="112322077048", use_depth=USE_DEPTH
    #     )
    if USE_WRIST_CAMERA:
        wrist_camera_config = RealSenseCameraConfig(
            fps=15, width=640, height=480, buffer_size=5, serial_number_or_name="112322074840", use_depth=USE_DEPTH
        )
    if USE_RIGHT_CAMERA:
        right_camera_config = RealSenseCameraConfig(
            fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="112322077222", use_depth=USE_DEPTH
        )
    if USE_LEFT_CAMERA:
        left_camera = RealSenseCamera(left_camera_config)
    if USE_RIGHT_CAMERA:
        right_camera = RealSenseCamera(right_camera_config)
    if USE_FRONT_CAMERA:
        front_camera = RealSenseCamera(front_camera_config)
    if USE_WRIST_CAMERA:
        wrist_camera = RealSenseCamera(wrist_camera_config)
    if USE_LEFT_CAMERA:
        left_camera.connect()
    if USE_RIGHT_CAMERA:
        right_camera.connect()
    if USE_FRONT_CAMERA:
        front_camera.connect()
    if USE_WRIST_CAMERA:
        wrist_camera.connect()
    if USE_LEFT_CAMERA:
        left_camera._start_read_thread()
    if USE_RIGHT_CAMERA:
        right_camera._start_read_thread()
    if USE_FRONT_CAMERA:
        front_camera._start_read_thread()
    if USE_WRIST_CAMERA:
        wrist_camera._start_read_thread()

    record_devices = {
        "left_camera": left_camera if USE_LEFT_CAMERA else None,
        "front_camera": front_camera if USE_FRONT_CAMERA else None,
        "right_camera": right_camera if USE_RIGHT_CAMERA else None,
        "wrist_camera": wrist_camera if USE_WRIST_CAMERA else None,
        "fr3": robot,
        "teleop": teleop
    }

    return record_devices


# =========================================================
# Streamlit app sidebar
# =========================================================

st.sidebar.title(":rainbow[Episode Configuration]")

# save dir and episode id
save_dir = st.sidebar.text_input("Save Directory", value=".")
episode_id = st.sidebar.number_input("Episode ID", min_value=0, value=st.session_state.episode_id, step=1)

# Display FPS and frames
# image_fps = setup_hardwares()["left_camera"].reading_fps
# state_fps = setup_hardwares()["fr3"].reading_fps
fps_col1, fps_col2 = st.sidebar.columns(2)
fps_placeholder1 = fps_col1.empty()
fps_placeholder2 = fps_col2.empty()
# fps_col1.metric("Image FPS", value=image_fps, delta=None, border=True)
# fps_col2.metric("State FPS", value=state_fps, delta=None, border=True)

frames_placeholder = st.sidebar.empty()

# start/end episode toggle
def toggle_recording():
    if st.session_state.recording:
        if st.session_state.recorder is not None:
            st.session_state.recorder.save_episode()
            st.session_state.recorder.close()
            st.session_state.recorder = None
            try:
                del st.session_state.robot
            except Exception as e:
                st.toast(f"Failed to disconnect robot: {e}", icon="🚨")
        st.session_state.recording = not st.session_state.recording
        st.session_state.last_timestep = 0
        if st.session_state.joint_states is not None:
            st.session_state.joint_states = None
        st.session_state.episode_id = episode_id + 1
        st.toast(f"Episode {episode_id} saved successfully!", icon="✅")
    else:
        if os.path.exists(os.path.join(save_dir, f"episode_{episode_id:05d}.parquet")):
            st.toast("Episode already exists. Please choose a different ID.", icon="🚨")
        else:
            st.session_state.recording = not st.session_state.recording
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                st.toast(f"Created save directory: {save_dir}", icon="📂")
            record_devices = setup_hardwares()
            with st.spinner("Waiting for robot to be ready..."):
                record_devices["fr3"].stop_teleoperation()
                record_devices["fr3"].home()
                record_devices["fr3"].start_teleoperation(
                    teleoperator=record_devices["teleop"],
                    action_mapping=action_mapping
                )
            st.session_state.recorder = EpisodeRecorder(
                episode_id=episode_id,
                record_devices=record_devices,
                save_dir=save_dir,
                fps=30,
                tolerance_ms=300
            )
            st.session_state.recorder.new_episode()
            st.session_state.last_timestep = 0
            if st.session_state.joint_states is not None:
                st.session_state.joint_states = None
            st.toast(f"Started recording episode {episode_id}.")

def delete_episode():
    if st.session_state.recording:
        if st.session_state.recorder is not None:
            del st.session_state.recorder
            st.session_state.recorder = None
            try:
                del st.session_state.robot
            except Exception as e:
                st.toast(f"Failed to disconnect robot: {e}", icon="🚨")
        st.session_state.recording = not st.session_state.recording
        st.session_state.last_timestep = 0
        if st.session_state.joint_states is not None:
            st.session_state.joint_states = None
        st.toast(f"Episode {episode_id} deleted successfully!", icon="❌")
    else:
        if os.path.exists(os.path.join(save_dir, f"episode_{episode_id:05d}.parquet")):
            os.remove(os.path.join(save_dir, f"episode_{episode_id:05d}.parquet"))
            st.toast(f"Episode {episode_id} deleted successfully!", icon="❌")
        else:
            st.toast(f"Episode {episode_id} does not exist.", icon="🚨")

if st.session_state.recording:
    st.sidebar.button("Save Episode", on_click=toggle_recording, use_container_width=True, icon="⏹️")
else:
    st.sidebar.button("Start Episode", on_click=toggle_recording, use_container_width=True, icon="▶️")

st.sidebar.button("Delete Episode", on_click=delete_episode, use_container_width=True, icon="🗑️")


# =========================================================
# Monitering and visualization
# =========================================================
    
col1, col2 = st.columns(2)
image_placeholder_1 = col1.empty()
image_placeholder_3 = col2.empty()
col3, col4 = st.columns(2)
image_placeholder_2 = col3.empty()
# image_placeholder_4 = col4.empty()
figure_placeholder = col4.empty()

def update_images(image_placeholder_1, image_placeholder_2, image_placeholder_3, figure_placeholder):
    while True:
        if USE_LEFT_CAMERA:
            left_image = setup_hardwares()["left_camera"].frame_buffer.queue[-1].color if not setup_hardwares()["left_camera"].frame_buffer.empty() else None
            if USE_DEPTH:
                left_image_depth = setup_hardwares()["left_camera"].frame_buffer.queue[-1].depth if not setup_hardwares()["left_camera"].frame_buffer.empty() else None
            else:
                left_image_depth = None
        else:
            left_image = None
            left_image_depth = None
        if USE_FRONT_CAMERA:
            front_image = setup_hardwares()["front_camera"].frame_buffer.queue[-1].color if not setup_hardwares()["front_camera"].frame_buffer.empty() else None
            if USE_DEPTH:
                front_image_depth = setup_hardwares()["front_camera"].frame_buffer.queue[-1].depth if not setup_hardwares()["front_camera"].frame_buffer.empty() else None
            else:
                front_image_depth = None
        else:
            front_image = None
            front_image_depth = None
        if USE_WRIST_CAMERA:
            wrist_image = setup_hardwares()["wrist_camera"].frame_buffer.queue[-1].color if not setup_hardwares()["wrist_camera"].frame_buffer.empty() else None
            if USE_DEPTH:
                wrist_image_depth = setup_hardwares()["wrist_camera"].frame_buffer.queue[-1].depth if not setup_hardwares()["wrist_camera"].frame_buffer.empty() else None
            else:
                wrist_image_depth = None
        else:
            wrist_image = None
            wrist_image_depth = None
        if USE_RIGHT_CAMERA:
            right_image = setup_hardwares()["right_camera"].frame_buffer.queue[-1].color if not setup_hardwares()["right_camera"].frame_buffer.empty() else None
            if USE_DEPTH:
                right_image_depth = setup_hardwares()["right_camera"].frame_buffer.queue[-1].depth if not setup_hardwares()["right_camera"].frame_buffer.empty() else None
        else:
            right_image = None
            right_image_depth = None
        if left_image is not None:
            # left_image = cv2.resize(left_image, (320, 180))
            st.session_state.left_image = left_image
            st.session_state.left_image_depth = left_image_depth
        if front_image is not None:
            # front_image = cv2.resize(front_image, (320, 180))
            st.session_state.front_image = front_image
            st.session_state.front_image_depth = front_image_depth
        if wrist_image is not None:
            # wrist_image = cv2.resize(wrist_image, (320, 180))
            st.session_state.wrist_image = wrist_image
            st.session_state.wrist_image_depth = wrist_image_depth
        if right_image is not None:
            # right_image = cv2.resize(right_image, (320, 180))
            st.session_state.right_image = right_image
            st.session_state.right_image_depth = right_image_depth

        if st.session_state.left_image is not None:
            image_placeholder_1.image(st.session_state.left_image, caption="Left Camera", use_container_width=True)

        if st.session_state.right_image is not None:
            image_placeholder_2.image(st.session_state.right_image, caption="Right Camera", use_container_width=True)
        if st.session_state.wrist_image is not None:
            image_placeholder_3.image(st.session_state.wrist_image, caption="Wrist Camera", use_container_width=True)
        # if st.session_state.front_image is not None:
        #     image_placeholder_2.image(st.session_state.front_image, caption="Front Camera", use_container_width=True)
        # if st.session_state.right_image is not None:
        #     image_placeholder_3.image(st.session_state.right_image, caption="Right Camera", use_container_width=True)
        # if st.session_state.wrist_image is not None:
        #     figure_placeholder.image(st.session_state.wrist_image, caption="Wrist Camera", use_container_width=True)
        if st.session_state.front_image is not None:
            figure_placeholder.image(st.session_state.front_image, caption="Front Camera", use_container_width=True)

        time.sleep(1.0 / 15)  # Assuming a target FPS of 30
    
    # st.rerun()


# @st.fragment
# def _update_figures():
#     image_fps = setup_hardwares()["left_camera"].reading_fps
#     state_fps = setup_hardwares()["fr3"].reading_fps
#     if st.session_state.image_fps_fig is None:
#         st.session_state.image_fps_fig = fps_gauge(image_fps, title="Image FPS")
#     if st.session_state.state_fps_fig is None:
#         st.session_state.state_fps_fig = fps_gauge(state_fps, title="State FPS")
#     st.session_state.image_fps_fig.update_traces(value=image_fps)
#     st.session_state.state_fps_fig.update_traces(value=state_fps)
#     joint_state = setup_hardwares()["fr3"].state_buffer.queue[-1].joint_positions if not setup_hardwares()["fr3"].state_buffer.empty() else None
#     if joint_state is not None:
#         if st.session_state.joint_states is None:
#             st.session_state.joint_states = pd.DataFrame(np.array(joint_state).reshape(1, -1), columns=[f"joint_{i}" for i in range(len(joint_state))])
#         else:
#             st.session_state.joint_states = pd.concat([st.session_state.joint_states, pd.DataFrame(np.array(joint_state).reshape(1, -1), columns=[f"joint_{i}" for i in range(len(joint_state))])], ignore_index=True)
#     if st.session_state.joint_states is not None and len(st.session_state.joint_states) > 500:
#         st.session_state.joint_states = st.session_state.joint_states.iloc[-500:]
    
#     image_placeholder_4.plotly_chart(st.session_state.image_fps_fig, use_container_width=True, theme="streamlit", key=uuid.uuid4())
#     image_placeholder_5.plotly_chart(st.session_state.state_fps_fig, use_container_width=True, theme="streamlit", key=uuid.uuid4())
#     if st.session_state.joint_states is not None:
#         image_placeholder_6.line_chart(
#             st.session_state.joint_states,
#             y= st.session_state.joint_states.columns.tolist(),
#             use_container_width=True,
#             y_label="Joint Position (rad)",
#         )
#     st.rerun(scope="fragment")

# def update_figures(image_placeholder_4, image_placeholder_5, image_placeholder_6):
#     while True:
#         image_fps = setup_hardwares()["left_camera"].reading_fps
#         state_fps = setup_hardwares()["fr3"].reading_fps
#         if st.session_state.image_fps_fig is None:
#             st.session_state.image_fps_fig = fps_gauge(image_fps, title="Image FPS")
#         if st.session_state.state_fps_fig is None:
#             st.session_state.state_fps_fig = fps_gauge(state_fps, title="State FPS")
#         st.session_state.image_fps_fig.update_traces(value=image_fps)

#         st.session_state.state_fps_fig.update_traces(value=state_fps)
#         joint_state = setup_hardwares()["fr3"].state_buffer.queue[-1].joint_positions if not setup_hardwares()["fr3"].state_buffer.empty() else None
#         if joint_state is not None:
#             if st.session_state.joint_states is None:
#                 st.session_state.joint_states = pd.DataFrame(np.array(joint_state).reshape(1, -1), columns=[f"joint_{i}" for i in range(len(joint_state))])
#             else:
#                 st.session_state.joint_states = pd.concat([st.session_state.joint_states, pd.DataFrame(np.array(joint_state).reshape(1, -1), columns=[f"joint_{i}" for i in range(len(joint_state))])], ignore_index=True)
#         if st.session_state.joint_states is not None and len(st.session_state.joint_states) > 500:
#             st.session_state.joint_states = st.session_state.joint_states.iloc[-500:]
        
#         image_placeholder_4.plotly_chart(st.session_state.image_fps_fig, use_container_width=True, theme="streamlit", key=uuid.uuid4())
#         image_placeholder_5.plotly_chart(st.session_state.state_fps_fig, use_container_width=True, theme="streamlit", key=uuid.uuid4())
#         if st.session_state.joint_states is not None:
#             image_placeholder_6.line_chart(
#                 st.session_state.joint_states,
#                 y= st.session_state.joint_states.columns.tolist(),
#                 use_container_width=True,
#                 y_label="Joint Position (rad)",
#             )
#         time.sleep(1.0 / 5)  # Assuming a target FPS of 30
        

def update_figures(figure_placeholder):
    while True:
        joint_state = setup_hardwares()["fr3"].state_buffer.queue[-1].joint_positions if not setup_hardwares()["fr3"].state_buffer.empty() else None
        if joint_state is not None:
            joint_states = pd.DataFrame(np.array(joint_state).reshape(1, -1), columns=[f"joint_{i}" for i in range(len(joint_state))])
            if st.session_state.line_fig is None:
                st.session_state.line_fig = figure_placeholder.line_chart(
                    joint_states,
                    y=joint_states.columns.tolist(),
                    use_container_width=True,
                    y_label="Joint Position (rad)",
                )
            else:
                st.session_state.line_fig.add_rows(joint_states)
        
        time.sleep(1.0 / 5)  # Assuming a target FPS of 30


def update_metrics(fps_placeholder1, fps_placeholder2, frames_placeholder):
    while True:
        image_fps = setup_hardwares()["left_camera"].reading_fps
        state_fps = setup_hardwares()["fr3"].reading_fps
        if st.session_state.recorder is not None:
            frames = st.session_state.recorder.total_frames
        else:
            frames = 0
        fps_placeholder1.metric("Image FPS", value=round(image_fps, 1), delta=None, border=True)
        fps_placeholder2.metric("State FPS", value=round(state_fps, 1), delta=None, border=True)
        frames_placeholder.metric("Recorded Frames", value=frames, delta=None, border=True)
        time.sleep(1.0 / 30)  # Update every 5 seconds

    
# Start the threads to update images and figures
image_thread = threading.Thread(target=update_images, args=(image_placeholder_1, image_placeholder_2, image_placeholder_3, figure_placeholder), daemon=True)
add_script_run_ctx(image_thread, get_script_run_ctx())

# figure_thread = threading.Thread(target=update_figures, args=(figure_placeholder1,), daemon=True)
# add_script_run_ctx(figure_thread, get_script_run_ctx())

metric_thread = threading.Thread(target=update_metrics, args=(fps_placeholder1, fps_placeholder2, frames_placeholder), daemon=True)
add_script_run_ctx(metric_thread, get_script_run_ctx())

image_thread.start()
# figure_thread.start()
metric_thread.start()

image_thread.join()
# figure_thread.join()
metric_thread.join()