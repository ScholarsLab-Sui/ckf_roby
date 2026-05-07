#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

from algo.utils.websocket_client_policy import WebsocketClientPolicy


def make_obs_pointcloud(to: int, n: int, d: int = 8) -> dict:
    pc = np.random.uniform(low=[-0.2, -0.2, 0.4], high=[0.2, 0.2, 1.2], size=(1, to, n, 3)).astype(np.float32)
    st = np.zeros((1, to, d), dtype=np.float32)
    st[..., 0] = 0.45
    st[..., 2] = 0.30
    st[..., 3] = 1.0
    st[..., 7] = 0.04
    return {"observation": {"point_cloud": pc, "agent_pos": st}}


def make_obs_legacy_state(to: int, d: int = 8) -> dict:
    st = np.zeros((1, to, d), dtype=np.float32)
    st[..., 0] = 0.45
    st[..., 2] = 0.30
    st[..., 3] = 1.0
    st[..., 7] = 0.04
    return {"observation": {"state": st}}


def validate_actions(name: str, out: dict) -> None:
    if "actions" not in out:
        raise RuntimeError(f"{name}: response missing 'actions': {out}")
    actions = np.asarray(out["actions"])
    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise RuntimeError(f"{name}: actions shape invalid: {actions.shape}, expect (B,Ta,7)")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError(f"{name}: actions contains non-finite values")
    print(f"[{name}] actions shape={actions.shape} first={actions[0,0].tolist()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3333)
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--n-points", type=int, default=1024)
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.2)
    args = p.parse_args()

    client = WebsocketClientPolicy(args.host, args.port)
    print("[smoke] connected, server metadata:", client.get_server_metadata())

    for i in range(args.loops):
        out = client.infer(make_obs_pointcloud(args.obs_horizon, args.n_points))
        validate_actions(f"pc-loop-{i}", out)
        time.sleep(args.sleep)

    out_legacy = client.infer(make_obs_legacy_state(args.obs_horizon))
    validate_actions("legacy-state", out_legacy)

    print("[smoke] PASS")


if __name__ == "__main__":
    main()
