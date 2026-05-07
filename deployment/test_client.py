import pickle
import numpy as np
from websocket import websocket_client_policy

with open("/data2/user/sunhaowen/hw_code/vla/VLA-Adapter/data.pkl", "rb") as f:
    test_observation = pickle.load(f)
observation = test_observation
new_test = {"agentview_image": observation["full_image"], "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8), "state": np.zeros((8,), dtype=np.float32)}
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8010)
task_description = "pick up the black bowl between the plate and the ramekin and place it on the plate"
obs = {"observation": new_test, "task_description": task_description}
actions = client.infer(obs)
print("Generated actions:", actions["actions"])