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
from typing import Any, Type, Dict

import draccus

from .configs import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """
    The base abstract class for all LeRobot-compatible teleoperation devices.

    This class provides a standardized interface for interacting with physical teleoperators.
    Subclasses must implement all abstract methods and properties to be usable.

    Attributes:
        config_class (RobotConfig): The expected configuration class for this teleoperator.
        name (str): The unique name used to identify this teleoperator type.
    """

    # Set these in ALL subclasses
    config_class: Type[TeleoperatorConfig]
    name: str

    def __init__(self, config: TeleoperatorConfig):
        self.id = config.id

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the teleoperator is currently connected or not. If `False`, calling :pymeth:`get_action`
        or :pymeth:`send_feedback` should raise an error.
        """
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """
        Establish communication with the teleoperator.
        """
        pass

    @abc.abstractmethod
    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the teleoperator.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    @abc.abstractmethod
    def get_action(self) -> Dict[str, Any]:
        """
        Retrieve the current action from the teleoperator.

        Returns:
            dict[str, Any]: A flat dictionary representing the teleoperator's current actions. Its
                structure should match :pymeth:`observation_features`.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the teleoperator and perform any necessary cleanup."""
        pass