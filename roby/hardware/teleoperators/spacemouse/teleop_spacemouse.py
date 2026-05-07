import logging
import time
from collections import defaultdict
from queue import Queue

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from threading import Event, Lock, Thread

import numpy as np

try:
    from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
except ImportError:
    raise ImportError(
        "The 'spnav' package is required for the SpaceNav input device. "
    )

from roby.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from roby.hardware.modeling_outputs import BaseHardwareOutput
from roby.hardware.teleoperators.teleoperator import Teleoperator
from roby.hardware.teleoperators.spacemouse.configuration_spacemouse import SpacemouseTeleopConfig

logger = logging.getLogger(__name__)


@dataclass
class SpacemouseOutput(BaseHardwareOutput):
    
    offsets: Optional[Tuple[float,...]] = None
    left_button: Optional[bool] = None
    right_button: Optional[bool] = None


class SpacemouseTeleop(Teleoperator):
    """SpaceNav input device class.

    This class represents a SpaceNav input device, which is used for 3D navigation.
    It provides methods to connect to the device, read data, and manage the device state.
    """

    config_class = SpacemouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpacemouseTeleopConfig):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        if np.issubdtype(type(config.deadzone), np.number):
            deadzone = np.full(6, fill_value=config.deadzone, dtype=np.float32)
        else:
            deadzone = np.array(config.deadzone, dtype=np.float32)
        assert (deadzone >= 0).all()

        self.stop_event = Event()
        self.max_value = config.max_value
        self.deadzone = deadzone
        self.event_queue = Queue()
        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.button_state = defaultdict(lambda: False)

        self.tx_zup_spnav = np.array([
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

    def is_connected(self) -> bool:
        """Check if the SpaceNav device is currently connected."""
        try:
            spnav_poll_event()
            return True
        except DeviceNotConnectedError:
            return False

    def connect(self, warmup: bool = True) -> None:
        """Connect to the SpaceNav device."""        
        spnav_open()
        self.stop_event.clear()
        if warmup:
            time.sleep(0.1)

    def _listening_loop(self) -> None:
        """Continuously read SpaceNav events in a background thread."""
        while not self.stop_event.is_set():
            try:
                event = spnav_poll_event()
                # self.event_queue.put(event)

                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                    
                time.sleep(1 / 100)

            except Exception as e:
                logger.error(f"Error reading SpaceNav event: {e}")

    def _start_listening_thread(self, warmup=True) -> None:
        """Start the background thread to read SpaceNav events."""
        if self.stop_event.is_set():
            self.stop_event.clear()

        self.thread = Thread(target=self._listening_loop, name=f"{self.name}_listening_loop")
        self.thread.daemon = True
        self.thread.start()

        if warmup:
            time.sleep(0.1)

    def _stop_listening_thread(self) -> None:
        """Stop the background thread reading SpaceNav events."""
        self.stop_event.set()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_motion_state(self):
        me = self.motion_event
        state = np.array(me.translation + me.rotation, dtype=np.float32) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state
    
    def is_button_pressed(self, button_id):
        return self.button_state[button_id]
    
    def _drain_event_queue(self) -> None:
        while not self.event_queue.empty():
            event = self.event_queue.get_nowait()
            if isinstance(event, SpnavMotionEvent):
                self.motion_event = event
            elif isinstance(event, SpnavButtonEvent):
                self.button_state[event.bnum] = event.press
            else:
                logger.warning(f"Unknown event type: {type(event)}")

    def configure(self) -> None:
        pass

    def get_action(self) -> SpacemouseOutput:
        """Read the current state of the SpaceNav device."""
        if not self.is_connected():
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        if not hasattr(self, 'thread') or not self.thread.is_alive():
            self._start_listening_thread()

        # self._drain_event_queue()

        state = self.get_motion_state_transformed()
        return SpacemouseOutput(
            timestamp=time.time()*1e3,  # milliseconds
            offsets=state,
            left_button=self.is_button_pressed(0),
            right_button=self.is_button_pressed(1)
        )
    
    def disconnect(self) -> None:
        """Disconnect from the SpaceNav device."""
        if not self.is_connected():
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        self._stop_listening_thread()
        spnav_close()
        logger.info(f"{self.name} disconnected.")


if __name__ == "__main__":
    # Example usage
    config = SpacemouseTeleopConfig(max_value=500, deadzone=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3))
    spacemouse = SpacemouseTeleop(config)
    
    try:
        spacemouse.connect()
        while True:
            output = spacemouse.get_action()
            print(output)
            time.sleep(0.1)
    except KeyboardInterrupt:
        spacemouse.disconnect()