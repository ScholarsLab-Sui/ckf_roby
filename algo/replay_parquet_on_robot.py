#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


def quat_to_xyzw(q_in: np.ndarray, quat_format: str) -> np.ndarray:
    q = np.asarray(q_in, dtype=np.float64).reshape(4)
    if quat_format == "xyzw":
        return q
    if quat_format == "wxyz":
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    raise ValueError(f"Unsupported quat_format={quat_format}")


def compute_delta_from_state(
    s0: np.ndarray,
    s1: np.ndarray,
    quat_format: str,
    invert_rot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    dpos = (s1[:3] - s0[:3]).astype(np.float64)

    q0 = quat_to_xyzw(s0[3:7], quat_format)
    q1 = quat_to_xyzw(s1[3:7], quat_format)

    # Keep exactly the same rotation-delta definition as training conversion:
    # delta = r1 * r0.inv()
    drot = (R.from_quat(q1) * R.from_quat(q0).inv()).as_rotvec().astype(np.float64)

    if invert_rot:
        drot = -drot

    return dpos, drot


def clip_delta(dpos: np.ndarray, drot: np.ndarray, pos_clip: float, rot_clip: float) -> tuple[np.ndarray, np.ndarray]:
    dpos = np.clip(dpos, -pos_clip, pos_clip)
    n = float(np.linalg.norm(drot))
    if n > rot_clip and n > 1e-12:
        drot = drot * (rot_clip / n)
    return dpos, drot


def compose_quat(cur_xyzw: np.ndarray, drot: np.ndarray, rot_order: str) -> np.ndarray:
    q_cur = R.from_quat(cur_xyzw)
    q_delta = R.from_rotvec(drot)
    if rot_order == "current_delta":
        q_new = q_cur * q_delta
    elif rot_order == "delta_current":
        q_new = q_delta * q_cur
    else:
        raise ValueError(f"Unsupported rot_order={rot_order}")
    return q_new.as_quat()


def load_parquet_states(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cols = ["fr3/end_effector_position", "fr3/gripper_width"]
    table = pq.read_table(str(parquet_path), columns=cols)

    ee = np.asarray(table.column("fr3/end_effector_position").to_pylist(), dtype=np.float64)
    gw = np.asarray(table.column("fr3/gripper_width").to_pylist(), dtype=np.float64)

    if ee.ndim != 2 or ee.shape[1] < 7:
        raise RuntimeError(f"Unexpected fr3/end_effector_position shape: {ee.shape}")

    state = np.zeros((ee.shape[0], 8), dtype=np.float64)
    state[:, :7] = ee[:, :7]
    state[:, 7] = gw
    return state, gw


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay raw ba20260427 parquet on FR3 for mapping diagnosis")
    parser.add_argument("--parquet", required=True, help="Path to raw parquet episode")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--hz", type=float, default=3.0)
    parser.add_argument("--pos-clip", type=float, default=0.002, help="Max |dpos| per axis in meters")
    parser.add_argument("--rot-clip", type=float, default=0.03, help="Max rotvec norm in radians")
    parser.add_argument(
        "--rot-order",
        default="current_delta",
        choices=["current_delta", "delta_current"],
        help="Quaternion composition order when applying rotation delta",
    )
    parser.add_argument("--invert-rot", action="store_true", help="Apply minus sign to rotvec delta")
    parser.add_argument("--disable-rot", action="store_true", help="Ignore rotation delta entirely")
    parser.add_argument(
        "--parquet-quat-format",
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion order used in parquet fr3/end_effector_position[3:7]",
    )
    parser.add_argument(
        "--robot-quat-format",
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion order returned by robot.read_state().end_effector_position[3:7]",
    )
    parser.add_argument("--load-gripper", action="store_true", help="Load gripper interface")
    parser.add_argument("--send-gripper", action="store_true", help="Also send open/close actions from width delta")
    parser.add_argument("--gripper-open-th", type=float, default=0.060, help="Treat width >= this as OPEN target")
    parser.add_argument("--gripper-close-th", type=float, default=0.040, help="Treat width <= this as CLOSE target")
    parser.add_argument(
        "--gripper-min-interval-steps",
        type=int,
        default=8,
        help="Minimum replay steps between two gripper commands",
    )
    parser.add_argument(
        "--gripper-delay-steps",
        type=int,
        default=0,
        help="Delay gripper events by N replay steps to better align with pose tracking",
    )
    parser.add_argument("--home", action="store_true", help="Home robot before replay")
    parser.add_argument("--dry-run", action="store_true", help="Do not send commands; only print diagnostics")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    state, _ = load_parquet_states(parquet_path)
    if state.shape[0] < 2:
        raise RuntimeError("Need at least 2 frames in parquet")

    start = max(0, int(args.start))
    end = min(state.shape[0] - 1, start + max(1, int(args.steps)))

    print("[replay] loaded", parquet_path)
    print("[replay] frames", state.shape[0], "start", start, "end", end)
    print("[replay] mode", "DRY-RUN" if args.dry_run else "LIVE")
    print("[replay] rot_order", args.rot_order, "invert_rot", args.invert_rot, "disable_rot", args.disable_rot)
    print("[replay] quat parquet", args.parquet_quat_format, "robot", args.robot_quat_format)
    print("[replay] clip pos", args.pos_clip, "rot", args.rot_clip, "hz", args.hz)
    print(
        "[replay] gripper th(open/close)",
        args.gripper_open_th,
        args.gripper_close_th,
        "min_interval_steps",
        args.gripper_min_interval_steps,
        "delay_steps",
        args.gripper_delay_steps,
    )

    if args.dry_run:
        # Fast diagnostics: print first 10 deltas
        for i in range(start, min(start + 10, end)):
            dpos, drot = compute_delta_from_state(
                state[i],
                state[i + 1],
                quat_format=args.parquet_quat_format,
                invert_rot=args.invert_rot,
            )
            dpos, drot = clip_delta(dpos, drot, args.pos_clip, args.rot_clip)
            if args.disable_rot:
                drot[:] = 0.0
            print(f"[delta {i}] dpos={dpos} drot={drot} |norm(drot)|={np.linalg.norm(drot):.6f}")
        print("[replay] dry-run done")
        return

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotAction, FR3RobotConfig, FR3ActionMode

    cfg = FR3RobotConfig(
        id="fr3",
        robot_ip=args.robot_ip,
        load_gripper=bool(args.load_gripper),
        relative_dynamics_factor=0.05,
        buffer_size=10,
    )

    robot = FR3Robot(cfg)
    robot.connect()
    robot.read_state()
    robot._start_read_thread()

    if args.home:
        robot.home()
        if args.load_gripper and robot.gripper is not None:
            robot.gripper.open(0.1)

    period = 1.0 / max(0.5, float(args.hz))

    # Gripper replay state: use hysteresis + cooldown to avoid noisy quick toggles.
    last_gripper_cmd_step = -10**9
    cur_width0 = float(state[start, 7])
    if cur_width0 >= args.gripper_open_th:
        gripper_target = "open"
    elif cur_width0 <= args.gripper_close_th:
        gripper_target = "close"
    else:
        gripper_target = "open"

    try:
        for i in range(start, end):
            dpos, drot = compute_delta_from_state(
                state[i],
                state[i + 1],
                quat_format=args.parquet_quat_format,
                invert_rot=args.invert_rot,
            )
            dpos, drot = clip_delta(dpos, drot, args.pos_clip, args.rot_clip)
            if args.disable_rot:
                drot[:] = 0.0

            rs = robot.read_state()
            cur = np.asarray(rs.end_effector_position, dtype=np.float64)  # [x,y,z,w,x,y,z]
            cur_xyzw = quat_to_xyzw(cur[3:7], args.robot_quat_format)

            new_pos = cur[:3] + dpos
            new_xyzw = compose_quat(cur_xyzw, drot, args.rot_order)

            grip_cmd = 0.0
            if args.send_gripper:
                idx = min(i + 1 + max(0, int(args.gripper_delay_steps)), state.shape[0] - 1)
                w_next = float(state[idx, 7])
                desired_target = gripper_target
                if w_next >= args.gripper_open_th:
                    desired_target = "open"
                elif w_next <= args.gripper_close_th:
                    desired_target = "close"

                if (
                    desired_target != gripper_target
                    and (i - last_gripper_cmd_step) >= int(args.gripper_min_interval_steps)
                ):
                    gripper_target = desired_target
                    last_gripper_cmd_step = i
                    grip_cmd = -1.0 if gripper_target == "open" else 1.0

            cmd = np.array([
                new_pos[0],
                new_pos[1],
                new_pos[2],
                new_xyzw[0],
                new_xyzw[1],
                new_xyzw[2],
                new_xyzw[3],
                grip_cmd,
            ])

            if args.verbose:
                print(
                    f"[step {i}] dpos={dpos} drot={drot} grip={grip_cmd} "
                    f"w_next={float(state[i + 1, 7]):.5f} target={gripper_target}"
                )

            robot.send_action(
                FR3RobotAction(cartesian_positions=cmd.tolist(), action_mode=FR3ActionMode.ABSOLUTE),
                asynchronous=False,
            )
            time.sleep(period)

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass

    print("[replay] live replay done")


if __name__ == "__main__":
    main()
