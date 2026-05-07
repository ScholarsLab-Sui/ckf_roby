# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from pathlib import Path
from typing import Any, Type, Optional, Tuple, Dict
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import draccus

from roby.hardware.robots.configs import RobotConfig


@dataclass
class RobotAction(abc.ABC):
    """
    Base class for actions that can be sent to a robot.

    This class serves as a template for defining specific action types.
    It can be extended to include additional attributes or methods relevant to the action.
    """

    @property
    @abc.abstractmethod
    def supported_actions(self) -> Dict[str, Any]:
        """
        Returns a tuple of supported action names for this action type.
        This should match the keys in the action_features dictionary.
        """
        pass


# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """
    The base abstract class for all LeRobot-compatible robots.

    This class provides a standardized interface for interacting with physical robots.
    Subclasses must implement all abstract methods and properties to be usable.

    Attributes:
        config_class (RobotConfig): The expected configuration class for this robot.
        name (str): The unique robot name used to identify this robot type.
    """

    # Set these in ALL subclasses
    config_class: Type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the robot.

        Args:
            calibrate (bool): If True, automatically calibrate the robot after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        pass

    @abc.abstractmethod
    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    @abc.abstractmethod
    def read_state(self) -> Any:
        """
        Read the current state of the robot.

        Returns:
            dict[str, Any]: The current state of the robot, which may include sensor readings,
                motor states, and other relevant information.
        """
        pass

    @abc.abstractmethod
    def send_action(self, action: RobotAction) -> Any:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action. Its structure should match
                :pymeth:`action_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        pass