from typing import Optional, Union
from dataclasses import asdict
import threading
import queue
import glob
import time
import os
import io
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from roby.common.utils import WindowAverageMeter


class EpisodeRecorder:
    def __init__(
        self,
        episode_id: Optional[int] = None,
        record_devices: dict[str, dict] = None,
        save_dir: Optional[str] = None,
        fps: Union[int, float] = 30.0,
        tolerance_ms: float = 20,
    ):
        """
        Start teleoperate and record an episode.
        Args:
            episode_id (str, optional): Unique identifier for the episode. If None, a random ID will be generated.
            record_devices (dict, optional): Dictionary of devices to record, with device names as keys and their configurations as values.
            fps (Union[int, float], optional): Frames per second for the recording. Default is 30.0.
            save_dir (str, optional): Directory where the episode data will be saved. If None, defaults to the current directory.
            buffer_size (int, optional): Size of the buffer for storing frames before writing to disk. Default is 10.
            queue_size (int, optional): Maximum size of the queue for storing frames. Default is 1000.
        """

        self.episode_id = episode_id
        self.record_devices = record_devices or {}
        self.save_dir = save_dir
        self.fps = fps
        self.fps_tracker = WindowAverageMeter(window_size=10)
        self.tolerance_ms = tolerance_ms

        self._current_frames = []
        self.num_frames = 0
        self.last_frame = None
        self._writer_thread = None

        self._stop_record_event = threading.Event()
        self._record_thread = None

        self.last_timestamp = None

    def _record_loop(self):
        """ Continuously read frames from the devices and store them in the current frames list."""
        while not self._stop_record_event.is_set():
            frame = {}
            timestamps = []
            device_outputs = {}
            for device_name, device in self.record_devices.items():
                if not hasattr(device, 'async_read'):
                    continue
                device_output = device.async_read()
                device_outputs[device_name] = device_output
                timestamps.append(device_output.timestamp)
                # print("in here")
                for key, value in asdict(device_output).items():
                    # print(key)
                    # print(device_output._image_keys())
                    # if key == "end_effector_position":
                    #     print(value)
                    if value is None:
                        continue
                    if key in device_output._image_keys():
                        if key == 'depth':  # 深度图
                            buf = io.BytesIO()
                            np.save(buf, value)  # 完整保存
                            value = buf.getvalue()
                            print("save depth")
                        else:
                            ### first center crop then resize to 256 256: chenxinzhe 251212
                            # 先进行中心裁剪到 480x480
                            cv2.imwrite(f"{device_name}.jpg", value[:,:,::-1])
                            height, width = value.shape[:2]
                            crop_size = 480

                            # 计算裁剪的起始位置（中心裁剪）
                            start_y = max(0, (height - crop_size) // 2)
                            start_x = max(0, (width - crop_size) // 2)
                            end_y = min(height, start_y + crop_size)
                            end_x = min(width, start_x + crop_size)

                            # 执行中心裁剪
                            cropped_image = value[start_y:end_y, start_x:end_x]

                            # 将裁剪后的图像调整到 256x256
                            resized_image = cv2.resize(cropped_image, dsize=(256, 256))
                            
                            # 编码为 PNG
                            success, encoded = cv2.imencode(".png", resized_image)

                            ####################
                            # success, encoded = cv2.imencode(".png", value)
                            value = encoded.tobytes()
                        # _, value = cv2.imencode(".jpg", value)
                        # value = value.tobytes()
                    elif isinstance(value, np.ndarray):
                        value = value.tolist()
                        # print("key", key)
                        # print("vaue", value)
                        # print("******************************")
                    frame[f"{device_name}/{key}"] = value

            assert max(timestamps) - min(timestamps) < self.tolerance_ms, "Timestamps from different devices are too far apart."
            frame["timestamp"] = np.mean(timestamps)
            self._current_frames.append(frame)
            self.num_frames += 1
            device_outputs["timestamp"] = frame["timestamp"]
            device_outputs["timestep"] = self.num_frames
            self.last_frame = device_outputs

            if self.last_timestamp is not None:
                timestamp = time.time()
                elapsed = timestamp - self.last_timestamp
                self.fps_tracker.update(elapsed)
            self.last_timestamp = time.time()

    def stop_record_thread(self):
        """Stop the recording thread."""
        if self._record_thread and self._record_thread.is_alive():
            self._stop_record_event.set()
            self._record_thread.join()
            self._record_thread = None
        self._stop_record_event.clear()

    def new_episode(self):
        """
        Start a new episode with a new episode ID.
        Args:
            episode_id (int, optional): Unique identifier for the new episode. If None, a random ID will be generated.
        """
        if self.episode_id is None:
            self.episode_id = len(glob.glob(os.path.join(self.save_dir, "*.parquet"))) + 1
        # else:
        #     self.episode_id += 1
        
        # restart the recording
        self.stop_record_thread()
        self._current_frames = []
        self.num_frames = 0
        self.last_timestamp = None
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        print(f"Started recording episode {self.episode_id}.")

    def _write_episode(self, episode_id, frames):
        """
        Write the recorded episode to disk.
        This method is called by the writer thread to save the current frames to a Parquet file.
        """
        if not frames:
            print("No frames recorded for this episode.")
            return
        
        df = pd.DataFrame(frames)
        save_path = os.path.join(self.save_dir, f"episode_{episode_id:05d}.parquet")
        pq.write_table(pa.Table.from_pandas(df), save_path)
        print(f"Episode {episode_id} saved to {save_path}.")

    def save_episode(self):
        """
        Save the recorded episode to disk.
        The episode data is saved as a Parquet file in the specified save directory.
        """        
        self.stop_record_thread()

        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join()

        self._writer_thread = threading.Thread(
            target=self._write_episode,
            args=(self.episode_id, self._current_frames),
            daemon=True
        )
        self._writer_thread.start()

    def close(self):
        """
        Close the recorder, stopping any ongoing recording and saving the current episode.
        """
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join()
        self._current_frames = []
        self.num_frames = 0
        print("Recorder closed.")

    @property
    def reading_fps(self):
        """Get the current frames per second."""
        return 1 / self.fps_tracker.avg if self.fps_tracker.avg is not None else 0

    @property
    def total_frames(self):
        """Get the total number of frames recorded in the current episode."""
        return self.num_frames