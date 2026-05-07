from dataclasses import dataclass, field
from typing import Optional

from roby.hardware.robots.configs import RobotConfig


@RobotConfig.register_subclass("fr3")
@dataclass
class FR3RobotConfig(RobotConfig):
    """Configuration class for FR3 robots.

    This class provides configuration options specific to FR3 robots, including
    the robot's unique identifier and any additional parameters needed for operation.

    Attributes:
        id (Optional[str]): Unique identifier for the robot. Defaults to None.
        additional_params (dict): Additional parameters specific to the FR3 robot.
    """

    robot_ip: Optional[str] = None
    load_gripper: bool = True
    relative_dynamics_factor: float = 0.1
    initial_joint: list = None
    initial_end_pose: list = None
    buffer_size: int = 10