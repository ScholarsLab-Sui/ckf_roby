import warnings
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BaseHardwareOutput:
    """Base class for hardware outputs.

    This class serves as a foundation for defining outputs from hardware components,
    such as cameras or sensors. It includes basic metadata like timestamp.

    Attributes:
        timestamp (Optional[float]): Timestamp of the output, if available.
    """

    timestamp: Optional[float] = None

    def _image_keys(self) -> Tuple[str,...]:
        """Returns the keys used for image data.

        Returns:
            Tuple[str, str]: Keys for RGB and depth images.
        """
        return []
