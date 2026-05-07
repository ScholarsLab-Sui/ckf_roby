import logging
import sys
import draccus
import numpy as np
from collections import deque
from termcolor import colored
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from websocket import websocket_policy_server
from websocket import base_policy as _base_policy
from typing_extensions import override

# ===================== [DP3-MIN] 仅新增这两行路径 =====================
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[4]
sys.path.insert(0, str(_ROOT / "diffusion_policies"))
# ================================================================


@dataclass
class GenerateConfig:
    # 保留原文件核心字段风格，只补 DP3 必要项
    port: int = 3333

    # ===================== [DP3-MIN] DP3 必要参数 =====================
    ckpt: Union[str, Path] = ""
    ckpt_dir: Union[str, Path] = "/hard_data1/user/chenkuifan/DemoGen/data/ckpts"
    device: str = "cuda:0"
    n_points: int = 1024
    default_gripper_open: float = 1.0
    # ================================================================

    # 保留原文件里常见的配置字段（便于阅读/对比）
    seed: int = 7


def _find_latest_ckpt(ckpt_dir: Path) -> Path:
    cks = sorted(ckpt_dir.glob("*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no ckpt in {ckpt_dir}")

    def key(p: Path):
        try:
            return int(p.stem)
        except Exception:
            return -1

    return sorted(cks, key=key)[-1]


def initialize_model(cfg: GenerateConfig):
    """[DP3-MIN] 替换原 OpenVLA 初始化为 DP3 checkpoint 初始化。"""
    import torch
    from diffusion_policies.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import (
        TrainDiffusionUnetHybridPointcloudWorkspace,
    )

    ckpt_path = Path(cfg.ckpt) if str(cfg.ckpt) else _find_latest_ckpt(Path(cfg.ckpt_dir))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    device = torch.device(cfg.device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    workspace = TrainDiffusionUnetHybridPointcloudWorkspace(payload["cfg"])
    workspace.load_checkpoint(path=ckpt_path)
    policy = workspace.ema_model if getattr(workspace, "ema_model", None) is not None else workspace.model
    policy = policy.to(device).eval()

    # [DP3-MIN] 返回和原 initialize_model 类似的“模型对象”语义
    return {
        "policy": policy,
        "device": device,
        "ckpt": str(ckpt_path),
        "n_obs_steps": int(getattr(workspace.cfg, "n_obs_steps", 2)),
        "n_action_steps": int(getattr(workspace.cfg, "n_action_steps", 5)),
    }


class Policy(_base_policy.BasePolicy):
    # [DP3-MIN] 保留原类名和结构，减少对比噪音
    def __init__(self, cfg: GenerateConfig, model_bundle):
        self.cfg = cfg
        self.model_bundle = model_bundle
        self.policy = model_bundle["policy"]
        self.device = model_bundle["device"]

        self.agent_pos_dim = 8
        try:
            params = getattr(self.policy, "normalizer", None)
            if params is not None and hasattr(params, "params_dict") and "agent_pos" in params.params_dict:
                self.agent_pos_dim = int(params.params_dict["agent_pos"]["scale"].shape[0])
        except Exception:
            pass

    @override
    def infer(self, obs):
        """
        [DP3-MIN] 与原 infer() 同职责：收 observation -> 回 actions

        支持两种输入：
        1) 新格式：observation.point_cloud + observation.agent_pos
        2) 兼容旧格式：observation.state （point_cloud 用 0 占位）
        """
        import torch

        observation = obs.get("observation", obs)

        if "point_cloud" in observation and "agent_pos" in observation:
            point_cloud = np.asarray(observation["point_cloud"], dtype=np.float32)
            agent_pos_in = np.asarray(observation["agent_pos"], dtype=np.float32)

            if point_cloud.ndim == 3:
                point_cloud = point_cloud[None, ...]
            if agent_pos_in.ndim == 2:
                agent_pos_in = agent_pos_in[None, ...]

            B, To = point_cloud.shape[:2]
            agent_pos = np.zeros((B, To, self.agent_pos_dim), dtype=np.float32)
            d_keep = min(agent_pos_in.shape[-1], self.agent_pos_dim)
            agent_pos[..., :d_keep] = agent_pos_in[..., :d_keep]
        else:
            state = np.asarray(observation["state"], dtype=np.float32)
            if state.ndim == 2:
                state = state[None, ...]
            B, To, D = state.shape

            agent_pos = np.zeros((B, To, self.agent_pos_dim), dtype=np.float32)
            d_keep = min(D, self.agent_pos_dim)
            agent_pos[..., :d_keep] = state[..., :d_keep]
            point_cloud = np.zeros((B, To, int(self.cfg.n_points), 3), dtype=np.float32)

        torch_obs = {
            "point_cloud": torch.from_numpy(point_cloud).to(self.device),
            "agent_pos": torch.from_numpy(agent_pos).to(self.device),
        }

        with torch.no_grad():
            out = self.policy.predict_action(torch_obs)
            actions = out["action"].detach().cpu().numpy().astype(np.float32)

        return {
            "actions": actions,
            "meta": {
                "policy": "dp3-min",
                "ckpt": self.model_bundle["ckpt"],
                "n_obs_steps": self.model_bundle["n_obs_steps"],
                "n_action_steps": self.model_bundle["n_action_steps"],
            },
        }


@draccus.wrap()
def run(cfg: GenerateConfig):
    # [DP3-MIN] run 主流程保持原结构：initialize -> Policy -> WebsocketServer
    model_bundle = initialize_model(cfg)
    policy = Policy(cfg, model_bundle)

    print(colored("Starting websocket policy server (DP3-MIN)...", "green"))
    print(colored(f"ckpt={model_bundle['ckpt']}", "cyan"))

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=cfg.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run()
