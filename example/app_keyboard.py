import streamlit as st
import os
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import logging
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

from roby.hardware.teleoperators.keyboard.teleop_keyboard import KeyboardTeleopConfig, KeyboardTeleop, KeyboardOutput
from roby.common.recorder import EpisodeRecorder
from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig, FR3RobotAction, FR3ActionMode


for name, l in logging.root.manager.loggerDict.items():
    if "streamlit" in name:
        l.disabled = True

# =========================================================
# Streamlit app title
# =========================================================

# st.title("Roby: Robot Control and Visualization")
st.set_page_config("Roby", layout="wide")


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
if "right_image" not in st.session_state:
    st.session_state.right_image = None
if "wrist_image" not in st.session_state:
    st.session_state.wrist_image = None


# =========================================================
# Set up hardwares
# =========================================================
    
def action_mapping(action: Optional[KeyboardOutput], velocity=0.5, angular_velocity=5.0) -> FR3RobotAction:
    translation = np.array([0.0, 0.0, 0.0])
    rotation = np.array([0.0, 0.0, 0.0])
    gripper = 0
    for key, _ in action.pressed_keys.items():
        """Maps keyboard actions to FR3RobotAction."""
        if key == "w":
            translation[0] += velocity # x
        elif key == "s":
            translation[0] -= velocity
        elif key == "a":
            translation[1] -= velocity # y
        elif key == "d":
            translation[1] += velocity
        elif key == "q":
            translation[2] -= velocity # z
        elif key == "e":
            translation[2] += velocity

        elif key == "i":
            rotation[2] += angular_velocity
        elif key == "k":
            rotation[2] -= angular_velocity
        elif key == "j":
            rotation[1] += angular_velocity
        elif key == "l":
            rotation[1] -= angular_velocity
        elif key == "u":
            rotation[0] += angular_velocity
        elif key == "o":
            rotation[0] -= angular_velocity

        elif key == "g":
            gripper = 1
        elif key == "h":
            gripper = -1

    translation = np.array(translation, dtype=np.float32)
    rotation = np.array(rotation, dtype=np.float32)
    rotation = R.from_euler("xyz", rotation, degrees=True).as_quat()
    positions = np.concatenate([translation, rotation, [gripper]]).tolist()
    return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.DELTA)


@st.cache_resource
def setup_hardwares():
    config = FR3RobotConfig(id="fr3", robot_ip="172.16.0.2", load_gripper=True, relative_dynamics_factor=0.05, buffer_size=10)
    robot = FR3Robot(config)
    robot.connect()
    robot.read_state()
    robot._start_read_thread()
    robot.home()
    
    teleop_config = KeyboardTeleopConfig()
    teleop = KeyboardTeleop(teleop_config)
    teleop.connect()
    robot.start_teleoperation(teleoperator=teleop, action_mapping=action_mapping)

    left_camera_config = RealSenseCameraConfig(
        fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="112322074336"
    )
    right_camera_config = RealSenseCameraConfig(
        fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="216322073340"
    )
    wrist_camera_config = RealSenseCameraConfig(
        fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="216322072028"
    )
    left_camera = RealSenseCamera(left_camera_config)
    right_camera = RealSenseCamera(right_camera_config)
    wrist_camera = RealSenseCamera(wrist_camera_config)
    left_camera.connect()
    right_camera.connect()
    wrist_camera.connect()
    left_camera._start_read_thread()
    right_camera._start_read_thread()
    wrist_camera._start_read_thread()

    record_devices = {
        "left_camera": left_camera,
        "right_camera": right_camera,
        "wrist_camera": wrist_camera,
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
image_placeholder_2 = col2.empty()

col3, col4 = st.columns(2)
image_placeholder_3 = col3.empty()
figure_placeholder = col4.empty()
# image_placeholder_3 = col4.empty()

def update_images(image_placeholder_1, image_placeholder_2, image_placeholder_3, figure_placeholder):
    while True:
        left_image = setup_hardwares()["left_camera"].frame_buffer.queue[-1].color if not setup_hardwares()["left_camera"].frame_buffer.empty() else None
        right_image = setup_hardwares()["right_camera"].frame_buffer.queue[-1].color if not setup_hardwares()["right_camera"].frame_buffer.empty() else None
        wrist_image = setup_hardwares()["wrist_camera"].frame_buffer.queue[-1].color if not setup_hardwares()["wrist_camera"].frame_buffer.empty() else None

        if left_image is not None:
            left_image = cv2.resize(left_image, (320, 180))
            st.session_state.left_image = left_image
        if right_image is not None:
            right_image = cv2.resize(right_image, (320, 180))
            st.session_state.right_image = right_image
        if wrist_image is not None:
            wrist_image = cv2.resize(wrist_image, (320, 180))
            st.session_state.wrist_image = wrist_image

        if st.session_state.left_image is not None:
            image_placeholder_1.image(st.session_state.left_image, caption="Left Camera", use_container_width=True)
        if st.session_state.right_image is not None:
            image_placeholder_2.image(st.session_state.right_image, caption="Right Camera", use_container_width=True)
        if st.session_state.wrist_image is not None:
            image_placeholder_3.image(st.session_state.wrist_image, caption="Wrist Camera", use_container_width=True)

        time.sleep(1.0 / 15)  # Assuming a target FPS of 30
            

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