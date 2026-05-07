from typing import Any, Tuple, Union
import numpy as np
from websocket import websocket_client_policy
import tensorflow as tf
from PIL import Image

def _to_numpy_uint8_rgb(data: Any) -> Any:
    """Convert raw image to HxWx3 uint8 RGB.
    Steps:
    - Decode bytes (prefer OpenCV). If OpenCV used: BGR->RGB.
    - Center-crop to 720x720.
    - Resize to 256x256.
    - Return numpy uint8 RGB array.
    """
    resize_to = (256, 256)

    def _center_crop_pil(pil_img: Image.Image) -> Image.Image:
        w, h = pil_img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return pil_img.crop((left, top, left + s, top + s))

    import cv2  # type: ignore
    buf = np.frombuffer(data, dtype=np.uint8)
    rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rgb is not None:
        pil = Image.fromarray(rgb, 'RGB')
        resample = getattr(Image, 'Resampling', Image).BILINEAR
        pil = _center_crop_pil(pil)
        pil = pil.resize(resize_to, resample)
        return np.asarray(pil, dtype=np.uint8)      
    
def get_franka_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["image"]
    img = _to_numpy_uint8_rgb(img)
    return img


def get_franka_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["wrist_image"]
    img = _to_numpy_uint8_rgb(img)
    return img
        
def resize_image_for_policy(img, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()

def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to 0 or 1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    if binarize:
        # Binarize to 0 or 1
        normalized_action[..., -1] = 1.0 if normalized_action[..., -1] >= 0.5 else 0.0

    return normalized_action

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_franka_image(obs)
    wrist_img = get_franka_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": obs["state"],
    }

    return observation, img  # Return both processed observation and original image for replay

def process_action(action):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    return action

if __name__ == "__main__":

    resize_size = 224
    port = 8059
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=port)
    task_description = "pick up the cube"
    # Setup
    MAX_STEPS = 300
    t = 0
    replay_images = []
    while t < MAX_STEPS:
        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)
        obs = {"observation": observation, "task_description": task_description}
        # Query model to get action
        actions = client.infer(obs)
        action = actions["actions"]
        action = process_action(action)
        t += 1
