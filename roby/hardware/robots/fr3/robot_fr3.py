import logging
import time
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Callable
from threading import Event, Lock, Thread

import numpy as np

try:
    import franky
except ImportError:
    raise ImportError(
        "The 'franky' package is required for the FR3 robot. "
    )
from roby.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from roby.common.buffer import FIFOBuffer
from roby.common.utils import WindowAverageMeter
from roby.hardware.modeling_outputs import BaseHardwareOutput

from roby.hardware.teleoperators.teleoperator import Teleoperator
from roby.hardware.robots.robot import Robot, RobotAction
from roby.hardware.robots.fr3.configuration_fr3 import FR3RobotConfig

logger = logging.getLogger(__name__)


class FR3ActionMode(str, Enum):
    """
    Enum for action modes supported by the FR3 robot.

    This enum defines the different modes in which actions can be sent to the FR3 robot.
    Each mode corresponds to a specific way of specifying actions, such as absolute or relative positions.
    """
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    DELTA = "delta"


@dataclass
class FR3RobotAction(RobotAction):
    """
    Action class for the FR3 robot.

    This class defines the actions that can be performed on the FR3 robot.
    It includes methods for sending actions to the robot and checking supported action modes.
    """

    cartesian_positions: Optional[np.ndarray] = None
    joint_positions: Optional[np.ndarray] = None
    action_mode: Optional[FR3ActionMode] = None

    @property
    def supported_actions(self) -> dict:
        """Returns a dictionary of supported actions for the FR3 robot."""
        return {
            "cartesian_positions": {
                "shape": (8,),
                "dtype": np.float32,
                "names": [
                    "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"
                ],
                "modes": [FR3ActionMode.ABSOLUTE, FR3ActionMode.RELATIVE, FR3ActionMode.DELTA]
            },
            "joint_positions": {
                "shape": (8,),
                "dtype": np.float32,
                "names": [
                    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"
                ],
                "modes": [FR3ActionMode.ABSOLUTE, FR3ActionMode.RELATIVE, FR3ActionMode.DELTA]
            },
        }
    

@dataclass
class FR3StateOutput(BaseHardwareOutput):
    """State output for the FR3 robot.

    This class extends BaseHardwareOutput to include specific state information for the FR3 robot.
    It can be used to encapsulate the state data returned by the robot during operation.
    
    Attributes:
        timestamp (Optional[float]): Timestamp of the output, if available.
    """
    
    end_effector_position: Optional[Tuple[float,...]] = None
    end_effector_velocity: Optional[Tuple[float,...]] = None
    joint_positions: Optional[Tuple[float,...]] = None
    joint_velocities: Optional[Tuple[float,...]] = None
    gripper_width: Optional[int] = None


class FR3Robot(Robot):
    """
    The FR3 robot class represents a specific type of robot in the roby framework.
    
    This class inherits from the base Robot class and implements methods specific to the FR3 robot.
    It includes functionality for connecting, configuring, and interacting with the FR3 robot.
    """

    config_class = FR3RobotConfig
    name = "fr3"
    # home_joint = [-0.04675282187699063, -0.041377512891862527, -0.005883186767752964, -2.5007440734645083, -0.05221588539595626, 2.496152230226616, 0.9172026069154627]
    # home_position = [0.4707543519985666, -0.02776023399946076, 0.19950666236991815, -0.9971261867839067, 0.07163508574300684, -0.019454377341563425, -0.015142962808831156]
    # home_position = [0.4707543519985666, -0.02776023399946076, 0.29950666236991815, -0.9971261867839067, 0.07163508574300684, -0.019454377341563425, -0.015142962808831156]
    home_position = [0.4707543519985666, -0.02776023399946076, 0.29950666236991815, -1.0, 0.0, 0.0, 0.0] #### pick and place task 251214 by chenxinzhe
    # home_joint = [-0.023454, -0.094028, -0.011200, -2.297059, 0.072552, 2.167216, 0.675963] #### pick and place task 251214 by chenxinzhe
    # home_position = [0.4707543519985666, -0.02776023399946076, 0.40950666236991815, -1.0, 0.0, 0.0, 0.0] #### new task 260114 by chenxinzhe

    # home_position = [0.3207543519985666, 0.12776023399946076, 0.29950666236991815, -1.0, 0.0, 0.0, 0.0] #### long horizon task 251214 by chenxinzhe 

    def __init__(self, config: FR3RobotConfig):
        super().__init__(config)
        self.robot_ip = config.robot_ip
        self.load_gripper = config.load_gripper
        self.relative_dynamics_factor = config.relative_dynamics_factor

        self.robot: Optional[franky.Robot] = None
        self.gripper: Optional[franky.Gripper] = None

        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None
        self.state_buffer: FIFOBuffer = FIFOBuffer(maxsize=config.buffer_size)
        self.new_state_event: Event = Event()

        self.teleoperator: Optional[Teleoperator] = None
        self.teleop_thread: Optional[Thread] = None
        self.teleop_stop_event: Optional[Event] = None

        self.fps_tracker = WindowAverageMeter()
        self.last_timestamp: Optional[float] = None

    @property
    def is_connected(self) -> bool:
        """Check if the FR3 robot is currently connected."""
        return self.robot is not None

    def connect(self) -> None:
        """
        Establish a connection to the FR3 robot.

        This method sets up the necessary communication with the robot, allowing for further operations.
        It may include steps like initializing hardware interfaces or establishing network connections.
        """
        self.robot = franky.Robot(self.robot_ip)
        self.robot.set_joint_impedance([1000]*7)
        if self.load_gripper:
            self.gripper = franky.Gripper(self.robot_ip)
        else:
            self.gripper = None

        if self.relative_dynamics_factor is not None:
            self.robot.relative_dynamics_factor = self.relative_dynamics_factor

    def configure(self) -> None:
        pass

    def home(self):
        # motion = franky.JointMotion(self.home_joint)
        # motion = franky.JointMotion(self.home_joint[:])
        motion = franky.CartesianMotion(franky.Affine(self.home_position[:3], self.home_position[3:]))
        try:
            self.robot.move(motion, asynchronous=False)
        except franky._franky.ControlException as e:
            self.robot.recover_from_errors()
        except RuntimeError as e:
            if "Motion Planner" in e.__str__():
                self.robot.recover_from_errors()
        except Exception as e:
            print(type(e), e)
        # self.robot.move(motion, asynchronous=False)

    def read_state(self) -> Any:
        """
        Read the current state of the FR3 robot.

        This method retrieves the robot's current state, including joint positions, velocities,
        end-effector position and velocity, and gripper state.

        Returns:
            FR3StateOutput: An instance containing the current state of the robot.
        """
        cartesian_state = self.robot.current_cartesian_state
        joint_state = self.robot.current_joint_state
        timestamp = time.time()*1e3

        end_effector_position = np.concatenate([cartesian_state.pose.end_effector_pose.translation, 
                                                cartesian_state.pose.end_effector_pose.quaternion]).tolist()
        end_effector_velocity = np.concatenate([cartesian_state.velocity.end_effector_twist.linear, 
                                                cartesian_state.velocity.end_effector_twist.angular]).tolist()
        joint_positions = joint_state.position.tolist()
        joint_velocities = joint_state.velocity.tolist()
        # end_effector_position = None
        # end_effector_velocity = None
        # joint_positions = None
        # joint_velocities = None

        if self.gripper is not None:
            gripper_width = self.gripper.width
            # gripper_width = None
        else:
            gripper_width = None

        return FR3StateOutput(
            timestamp=timestamp,  # Timestamp can be set as needed
            end_effector_position=end_effector_position,
            end_effector_velocity=end_effector_velocity,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            gripper_width=gripper_width
        )
    
    def _read_loop(self):
        """ Continuously reads the state of the FR3 robot in a background thread."""
        while not self.stop_event.is_set():
            try:
                state = self.read_state()
                # print("ori_load", state.end_effector_position, state.joint_positions)
                self.state_buffer.put(state)
                self.new_state_event.set()

                timestamp = time.perf_counter()
                if self.last_timestamp is not None:
                    elapsed_time = timestamp - self.last_timestamp
                    self.fps_tracker.update(elapsed_time)
                self.last_timestamp = timestamp
                time.sleep(1/30)

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read state if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> FR3StateOutput:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_state_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        state = self.state_buffer.queue[-1] if not self.state_buffer.empty() else None
        self.new_state_event.clear()

        if state is None:
            raise RuntimeError(f"Internal error: Event set but no state available for {self}.")

        return state
    
    @property
    def reading_fps(self) -> float:
        """
        Returns the average FPS of the camera based on the last read operations.

        This is calculated as the inverse of the average time taken to read a frame
        from the camera, based on the `fps_tracker`.

        Returns:
            float: The average FPS of the camera.
        """
        return 1. / self.fps_tracker.avg if self.fps_tracker.avg is not None else 0.0
    
    def send_action(self, action: FR3RobotAction, asynchronous=True) -> Any:
        """
        Send an action command to the FR3 robot.

        Args:
            action (FR3RobotAction): The action to be sent to the robot, which may include
                joint positions, cartesian positions, or gripper commands.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if action.cartesian_positions is None and action.joint_positions is None:
            return

        if action.cartesian_positions is not None and action.joint_positions is not None:
            raise ValueError("Cannot send both cartesian and joint positions at the same time.")
        
        if action.action_mode is None:
            raise ValueError("Action mode must be specified for the action.")
        
        if action.action_mode not in FR3ActionMode:
            raise ValueError(f"Unsupported action mode: {action.action_mode}. Supported modes are: {list(FR3ActionMode)}.")
        
        if action.action_mode == FR3ActionMode.ABSOLUTE:
            if action.cartesian_positions is not None:
                motion = franky.CartesianMotion(
                    franky.Affine(action.cartesian_positions[:3], action.cartesian_positions[3:7]), franky.ReferenceType.Absolute
                )
            elif action.joint_positions is not None:
                motion = franky.JointMotion(
                    action.joint_positions[:7], franky.ReferenceType.Absolute
                )
        elif action.action_mode == FR3ActionMode.RELATIVE:
            raise NotImplementedError("Relative action mode is not implemented for FR3 robot.")
        elif action.action_mode == FR3ActionMode.DELTA:
            if action.cartesian_positions is not None:
                motion = franky.CartesianMotion(
                    franky.Affine(action.cartesian_positions[:3], action.cartesian_positions[3:7]), franky.ReferenceType.Relative
                )
            elif action.joint_positions is not None:
                motion = franky.JointMotion(
                    action.joint_positions[:7], franky.ReferenceType.Relative
                )
        else:
            raise ValueError(f"Unsupported action mode: {action.action_mode}. Supported modes are: {list(FR3ActionMode)}.")
        
        try:
            self.robot.move(motion, asynchronous=asynchronous)
        except franky._franky.ControlException as e:
            self.robot.recover_from_errors()
        except RuntimeError as e:
            if "Motion Planner" in e.__str__():
                self.robot.recover_from_errors()
        except Exception as e:
            print(type(e), e)

        gripper_state = action.cartesian_positions[-1] if action.cartesian_positions is not None else action.joint_positions[-1]
        if gripper_state == -1:
            self.gripper.open(0.1)
        elif gripper_state == 1:
            print(1)
            # self.gripper.grasp(0.0, 0.1, 30, epsilon_outer=1.0)              
            self.gripper.grasp(0.0, 0.1, 30, epsilon_inner=0.0, epsilon_outer=0.1)       
    def _teleoperate_loop(self, action_mapping: Callable) -> None:
        """ Continuously reads teleoperator inputs and sends actions to the robot."""
        while not self.teleop_stop_event.is_set():
            try:
                if self.teleoperator is None:
                    raise DeviceNotConnectedError(f"{self} is not connected to a teleoperator.")

                action = self.teleoperator.get_action()
                if action is not None:
                    action = action_mapping(action)
                    self.send_action(action, asynchronous=True)
                time.sleep(1/30)
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error in teleoperation loop for {self}: {e}")

    def start_teleoperation(self, teleoperator: Teleoperator, action_mapping: Callable) -> None:
        """
        Start teleoperation with the specified teleoperator.

        Args:
            teleoperator (Teleoperator): The teleoperator to use for controlling the robot.
            action_mapping (Callable): A function that maps teleoperator actions to robot motions.
        """
        if self.teleop_thread is not None and self.teleop_thread.is_alive():
            self.stop_teleoperation()

        self.teleoperator = teleoperator
        self.teleop_stop_event = Event()
        self.teleop_thread = Thread(target=self._teleoperate_loop, args=(action_mapping,), name=f"{self}_teleop_loop")
        self.teleop_thread.daemon = True
        self.teleop_thread.start()

    def stop_teleoperation(self) -> None:
        """Stop the teleoperation thread and clean up resources."""
        if self.teleop_stop_event is not None:
            self.teleop_stop_event.set()

        if self.teleop_thread is not None and self.teleop_thread.is_alive():
            self.teleop_thread.join()

        self.teleop_thread = None
        self.teleop_stop_event = None
        self.teleoperator = None

    def disconnect(self) -> None:
        self.robot.stop()
        del self.robot


if __name__ == "__main__":

    from roby.hardware.teleoperators.spacemouse.teleop_spacemouse import SpacemouseTeleopConfig, SpacemouseTeleop, SpacemouseOutput
    from roby.common.recorder import EpisodeRecorder
    from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
    from scipy.spatial.transform import Rotation as R

    def action_mapping(action: Optional[SpacemouseOutput], translation_factor: Optional[float] = 0.05, rotation_factor: Optional[float] = 10.0) -> FR3RobotAction:
        """Maps Spacemouse actions to FR3RobotAction."""
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.DELTA)
        
        translation = np.array(action.offsets[:3]) * np.array([-1, 1, -1])
        rotation = np.array(action.offsets[3:]) * np.array([-1, 1, -1])
        if translation_factor is not None:
            translation *= translation_factor
        if rotation_factor is not None:
            rotation *= rotation_factor
        rotation = R.from_euler("xyz", rotation, degrees=True).as_quat()
        
        gripper = 0
        if action.left_button and not action.right_button:
            gripper = 1
        elif action.right_button and not action.left_button:
            gripper = -1

        positions = np.concatenate([translation, rotation, [gripper]]).tolist()
        return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.DELTA)
        
    config = FR3RobotConfig(id="fr3", robot_ip="172.16.0.2", load_gripper=True, relative_dynamics_factor=0.05, buffer_size=10)
    robot = FR3Robot(config)
    robot.connect()
    robot.read_state()
    robot._start_read_thread()
    
    teleop_config = SpacemouseTeleopConfig()
    teleop = SpacemouseTeleop(teleop_config)
    teleop.connect()
    robot.start_teleoperation(teleoperator=teleop, action_mapping=action_mapping)

    left_camera_config = RealSenseCameraConfig(
        fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="112322074336"
    )
    right_camera_config = RealSenseCameraConfig(
        fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="216322072028"
    )
    wrist_camera_config = RealSenseCameraConfig(
        fps=15, width=1280, height=720, buffer_size=5, serial_number_or_name="216322073340"
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

    recorder = EpisodeRecorder(
        episode_id=0,
        record_devices={
            "left_camera": left_camera,
            "right_camera": right_camera,
            "wrist_camera": wrist_camera,
            "fr3": robot
        },
        save_dir=".",
        fps=30,
        tolerance_ms=200
    )

    recorder.new_episode()

    for i in range(100):
        print(recorder.reading_fps)
        time.sleep(0.1)

    recorder.save_episode()
    recorder.close()
    
    # while True:
    #     state = robot.async_read(timeout_ms=200)
    #     print(robot.reading_fps)
    #     time.sleep(0.1)
    #     # print(1)
