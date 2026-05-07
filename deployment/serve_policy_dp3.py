#!/usr/bin/env python3
"""
DP3 websocket inference server for sxy_roby real-robot client.

Protocol compatibility:
- Input (from algo/dp_client.py):
    {"observation": {"rgb": (B,To,H,W,C), "state": (B,To,D)}}
- Output:
    {"actions": (B,Ta,7)}

This server is intended as a deployment bridge when policy is trained in
`diffusion_policies` and client already uses websocket protocol.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Dict, Any

import numpy as np

_CONDA_LIB = "/hard_data1/user/chenkuifan/anaconda3/envs/mstest/lib"
if os.path.isdir(_CONDA_LIB):
    _old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _CONDA_LIB not in _old_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{_CONDA_LIB}:{_old_ld}" if _old_ld else _CONDA_LIB


_THIS = pathlib.Path(__file__).resolve()
# .../DemoGen/ManiSkill/ba20260427/sxy_roby/deployment/serve_policy_dp3.py
_ROOT = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_ROOT / "diffusion_policies"))

from websocket import base_policy as _base_policy


def dict_apply(x, fn):
    if isinstance(x, dict):
        return {k: dict_apply(v, fn) for k, v in x.items()}
    return fn(x)


def _find_latest_ckpt(ckpt_dir: pathlib.Path) -> pathlib.Path:
    cks = sorted(ckpt_dir.glob("*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no ckpt in {ckpt_dir}")

    def key(p: pathlib.Path):
        try:
            return int(p.stem)
        except Exception:
            return -1

    return sorted(cks, key=key)[-1]


def load_policy(ckpt_path: pathlib.Path, device):
    import torch
    from diffusion_policies.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import (
        TrainDiffusionUnetHybridPointcloudWorkspace,
    )

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    workspace = TrainDiffusionUnetHybridPointcloudWorkspace(payload["cfg"])
    workspace.load_checkpoint(path=ckpt_path)
    policy = workspace.ema_model if getattr(workspace, "ema_model", None) is not None else workspace.model
    policy = policy.to(device).eval()

    n_obs = int(getattr(workspace.cfg, "n_obs_steps", 2))
    n_actions = int(getattr(workspace.cfg, "n_action_steps", 5))

    agent_pos_dim = 8
    try:
        params = getattr(policy, "normalizer", None)
        if params is not None and hasattr(params, "params_dict") and "agent_pos" in params.params_dict:
            agent_pos_dim = int(params.params_dict["agent_pos"]["scale"].shape[0])
    except Exception:
        pass

    return policy, agent_pos_dim, n_obs, n_actions


class DP3PolicyBridge(_base_policy.BasePolicy):
    def __init__(
        self,
        ckpt_path: pathlib.Path,
        device: str = "cuda",
        n_points: int = 1024,
        default_gripper_open: float = 1.0,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.policy, self.agent_pos_dim, self.n_obs, self.n_actions = load_policy(ckpt_path, self.device)
        self.n_points = int(n_points)
        self.default_gripper_open = float(default_gripper_open)
        self.ckpt_path = str(ckpt_path)

    def _build_policy_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert sxy_roby client observation to DP3 policy observation.

        Preferred input from client:
            observation.point_cloud: (B,To,N,3)
            observation.agent_pos:   (B,To,D)

        Legacy fallback (old dp_client):
            observation.state:       (B,To,D)
        where point_cloud is not provided and will be placeholder zeros.
        """
        observation = obs.get("observation", obs)

        if "point_cloud" in observation and "agent_pos" in observation:
            point_cloud = np.asarray(observation["point_cloud"], dtype=np.float32)
            agent_pos_in = np.asarray(observation["agent_pos"], dtype=np.float32)

            if point_cloud.ndim == 3:
                point_cloud = point_cloud[None, ...]
            if agent_pos_in.ndim == 2:
                agent_pos_in = agent_pos_in[None, ...]

            if point_cloud.ndim != 4:
                raise ValueError(f"point_cloud must be (B,To,N,3), got {point_cloud.shape}")
            if agent_pos_in.ndim != 3:
                raise ValueError(f"agent_pos must be (B,To,D), got {agent_pos_in.shape}")
            if point_cloud.shape[0] != agent_pos_in.shape[0] or point_cloud.shape[1] != agent_pos_in.shape[1]:
                raise ValueError(
                    "point_cloud and agent_pos must share (B,To), got "
                    f"{point_cloud.shape[:2]} vs {agent_pos_in.shape[:2]}"
                )

            B, To = point_cloud.shape[:2]
            agent_pos = np.zeros((B, To, self.agent_pos_dim), dtype=np.float32)
            d_keep = min(agent_pos_in.shape[-1], self.agent_pos_dim)
            agent_pos[..., :d_keep] = agent_pos_in[..., :d_keep]
            if self.agent_pos_dim > d_keep and self.agent_pos_dim >= 8:
                agent_pos[..., 7] = self.default_gripper_open
        else:
            if "state" not in observation:
                raise KeyError("missing observation/point_cloud+agent_pos (or legacy state)")

            state = np.asarray(observation["state"], dtype=np.float32)
            if state.ndim == 2:
                state = state[None, ...]
            if state.ndim != 3:
                raise ValueError(f"state must be (B,To,D), got {state.shape}")

            B, To, D = state.shape
            agent_pos = np.zeros((B, To, self.agent_pos_dim), dtype=np.float32)
            d_keep = min(D, self.agent_pos_dim)
            agent_pos[..., :d_keep] = state[..., :d_keep]
            if self.agent_pos_dim > d_keep and self.agent_pos_dim >= 8:
                agent_pos[..., 7] = self.default_gripper_open

            point_cloud = np.zeros((B, To, self.n_points, 3), dtype=np.float32)

        return {
            "point_cloud": point_cloud,
            "agent_pos": agent_pos,
        }

    def infer(self, obs: Dict) -> Dict:
        np_obs = self._build_policy_obs(obs)
        torch_obs = dict_apply(np_obs, lambda x: self.torch.from_numpy(x).to(self.device))

        with self.torch.no_grad():
            out = self.policy.predict_action(torch_obs)
            actions = out["action"].detach().cpu().numpy().astype(np.float32)

        return {
            "actions": actions,
            "meta": {
                "ckpt": self.ckpt_path,
                "n_obs_steps": self.n_obs,
                "n_action_steps": self.n_actions,
                "agent_pos_dim": self.agent_pos_dim,
                "note": "point_cloud is placeholder zeros (client has no depth stream in request).",
            },
        }


class MockPolicyBridge(_base_policy.BasePolicy):
    """Pure software mock policy for deployment smoke tests (no model, no robot)."""

    def __init__(self, mode: str = "zero") -> None:
        self.mode = str(mode)
        self._t0 = time.time()

    def infer(self, obs: Dict) -> Dict:
        observation = obs.get("observation", obs)

        # infer batch size and horizon from any available key
        B = 1
        To = 2
        if "agent_pos" in observation:
            arr = np.asarray(observation["agent_pos"])
            if arr.ndim >= 3:
                B, To = int(arr.shape[0]), int(arr.shape[1])
        elif "state" in observation:
            arr = np.asarray(observation["state"])
            if arr.ndim >= 3:
                B, To = int(arr.shape[0]), int(arr.shape[1])
        elif "point_cloud" in observation:
            arr = np.asarray(observation["point_cloud"])
            if arr.ndim >= 4:
                B, To = int(arr.shape[0]), int(arr.shape[1])

        Ta = 1
        actions = np.zeros((B, Ta, 7), dtype=np.float32)
        if self.mode == "sine":
            t = float(time.time() - self._t0)
            actions[..., 0] = 0.05 * np.sin(t)  # dx
            actions[..., 6] = np.sign(np.sin(t))  # gripper
        elif self.mode == "open":
            actions[..., 6] = 1.0
        elif self.mode == "close":
            actions[..., 6] = -1.0

        return {
            "actions": actions,
            "meta": {
                "policy": "mock",
                "mode": self.mode,
                "note": "No checkpoint loaded. For software-only smoke tests.",
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--ckpt-dir", default="/hard_data1/user/chenkuifan/DemoGen/data/ckpts")
    parser.add_argument("--mock-policy", action="store_true", help="Run without model checkpoint for software smoke tests")
    parser.add_argument("--mock-mode", default="zero", choices=["zero", "sine", "open", "close"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-points", type=int, default=1024)
    args = parser.parse_args()

    from websocket import websocket_policy_server

    if args.mock_policy:
        policy = MockPolicyBridge(mode=args.mock_mode)
        ckpt = "<mock>"
    else:
        if args.ckpt:
            ckpt = pathlib.Path(args.ckpt)
        else:
            ckpt = _find_latest_ckpt(pathlib.Path(args.ckpt_dir))
        if not ckpt.is_file():
            raise FileNotFoundError(f"ckpt not found: {ckpt}")

        policy = DP3PolicyBridge(
            ckpt_path=ckpt,
            device=args.device,
            n_points=args.n_points,
        )

    print("[dp3-server] start")
    print(f"[dp3-server] ckpt={ckpt}")
    print(f"[dp3-server] host={args.host} port={args.port} device={args.device}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={"policy": "dp3", "ckpt": str(ckpt)},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
