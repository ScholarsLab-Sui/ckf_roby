from dataclasses import dataclass, field
from typing import Optional, Tuple

from roby.hardware.teleoperators.configs import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseTeleopConfig(TeleoperatorConfig):

    """Configuration class for the SpaceMouse teleoperator.

    This class provides configuration options specific to the SpaceMouse teleoperator,
    including the maximum value for input, deadzone settings, and frequency of updates.

    Attributes:
        max_value (float): Maximum value for input. Defaults to 500.
        deadzone (Tuple[float, ...]): Deadzone settings for each axis. Defaults to (0, 0, 0, 0, 0, 0).
        frequency (Optional[float]): Frequency of updates in Hz. Defaults to None.
    """

    max_value: float = 500
    deadzone: Tuple[float, ...] = field(default_factory=lambda: (0.3, 0.3, 0.3, 0.3, 0.3, 0.3))
