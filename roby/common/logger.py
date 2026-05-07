import abc

import rerun as rr


class Logger(abc.ABC):
    @abc.abstractmethod
    def log_image(self, image, name: str, timestamp: float = None, timestep: int = None):
        """Log an image with a given name."""
        pass

    @abc.abstractmethod
    def log_scalar(self, value, name: str, timestamp: float = None, timestep: int = None):
        """Log a scalar value with a given name."""
        pass


class RRLogger(Logger):
    def __init__(self, name: str, spawn: bool = True):
        """Initialize the Rerun logger with a specific episode ID."""
        rr.init(name, spawn=spawn)

    def log_image(self, image, name: str, timestamp: float = None, timestep: int = None):
        """Log an image to Rerun."""
        rr.log(name, rr.Image(image), timestamp=timestamp, timestep=timestep)

    def log_scalar(self, value, name: str, timestamp: float = None, timestep: int = None):
        """Log a scalar value to Rerun."""
        rr.log(name, rr.Scalar(value), timestamp=timestamp, timestep=timestep)