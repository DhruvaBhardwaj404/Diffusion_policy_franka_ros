#!/usr/bin/env python3
"""
sim_node.py  (sim client test version)
---------------------------------------
Publishes action chunks to /diffusion_policy/action_chunk to drive
controller_node running in --mode sim (PyBullet via Polymetis).

Does NOT work for hardware — positions are computed relative to the
Panda home pose in PyBullet and are not validated against real workspace limits.

Setup:
    # Terminal 1: start PyBullet sim server
    launch_robot.py robot_client=bullet_sim gui=true

    # Terminal 2: controller node in sim mode
    python controller_node.py --mode sim --interp_hz 50

    # Terminal 3: this script
    source /opt/ros/noetic/setup.bash
    python sim_node.py

Options:
    --action_hz       chunk publish rate Hz  (default: 10, match controller)
    --n_action_steps  steps per chunk        (default: 8)
    --amplitude       metres of sine motion  (default: 0.03 = 3cm, safe for sim)
    --verbose         print each chunk
"""

import argparse
import time

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


# ── Panda home EEF pose in PyBullet (what robot.go_home() lands at) ──────────
# Polymetis bullet_sim starts at qr = [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]
# FK of that configuration gives approximately:
PANDA_HOME_POS = np.array([0.307, 0.0, 0.487], dtype=np.float64)  # metres

# ── Normalisation limits — must match controller_node + data conversion ───────
EEF_POS_LOWER_LIMITS = np.array([0.3,  0.02, 0.07], dtype=np.float64)
EEF_POS_UPPER_LIMITS = np.array([0.65, 0.25, 0.55], dtype=np.float64)


def normalize_eef_pos(pos: np.ndarray) -> np.ndarray:
    pos_range = EEF_POS_UPPER_LIMITS - EEF_POS_LOWER_LIMITS
    return 2.0 * (pos - EEF_POS_LOWER_LIMITS) / pos_range - 1.0


def build_action_msg(actions: np.ndarray) -> Float64MultiArray:
    """actions: (N, 7)  [norm_pos(3) | euler_xyz(3) | gripper(1)]"""
    n_steps, action_dim = actions.shape
    msg = Float64MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="n_steps",
                            size=n_steps,
                            stride=n_steps * action_dim),
        MultiArrayDimension(label="action_dim",
                            size=action_dim,
                            stride=action_dim),
    ]
    msg.data = actions.flatten().tolist()
    return msg


def make_chunk(t_start: float, n_steps: int,
               action_hz: float, amplitude: float) -> np.ndarray:
    """
    Slow sine/cosine motion centred on Panda home position.
    Orientation fixed (wrist pointing down).
    Gripper opens and closes every 5 seconds.
    """
    dt    = 1.0 / action_hz
    times = t_start + np.arange(n_steps) * dt
    freq  = 0.05   # Hz — very slow, easy to watch in PyBullet GUI

    x = PANDA_HOME_POS[0] + amplitude * np.sin(2 * np.pi * freq * times)
    y = PANDA_HOME_POS[1] + amplitude * np.cos(2 * np.pi * freq * times)
    z = np.full(n_steps, PANDA_HOME_POS[2])

    real_pos = np.stack([x, y, z], axis=1)   # (N, 3) metres
    norm_pos = normalize_eef_pos(real_pos)    # (N, 3) in [-1,1]

    # neutral orientation — wrist pointing straight down
    euler = np.zeros((n_steps, 3), dtype=np.float64)
    euler[:, 0] = np.pi

    # toggle gripper every 5 s
    gripper = np.where((times % 10.0) < 5.0, 1.0, 0.0).reshape(-1, 1)

    return np.concatenate([norm_pos, euler, gripper], axis=1)  # (N, 7)


def print_chunk(chunk: np.ndarray, idx: int):
    print(f"\n{'─'*65}")
    print(f"  Chunk #{idx}   shape={chunk.shape}")
    print(f"{'─'*65}")
    print(f"  {'step':>4}  "
          f"{'eef_x':>8} {'eef_y':>8} {'eef_z':>8}  "
          f"{'roll':>8} {'pitch':>8} {'yaw':>8}  {'grip':>5}")
    print(f"  {'':->4}  {'':->8} {'':->8} {'':->8}  "
          f"{'':->8} {'':->8} {'':->8}  {'':->5}")
    for i, row in enumerate(chunk):
        print(f"  {i:>4}  "
              f"{row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f}  "
              f"{row[3]:>8.4f} {row[4]:>8.4f} {row[5]:>8.4f}  "
              f"{'open' if row[6] > 0.5 else 'close':>5}")
    print(f"{'─'*65}")


def main():
    parser = argparse.ArgumentParser(
        description="Sim node — drives controller_node --mode sim only")
    parser.add_argument("--action_hz",      type=float, default=10.0,
                        help="Publish rate Hz — must match controller (default 10)")
    parser.add_argument("--n_action_steps", type=int,   default=8,
                        help="Steps per chunk (default 8)")
    parser.add_argument("--amplitude",      type=float, default=0.03,
                        help="Sine amplitude in metres (default 0.03 = 3cm)")
    parser.add_argument("--verbose",        action="store_true",
                        help="Print each chunk")
    args, unknown = parser.parse_known_args()

    rospy.init_node("sim_node", anonymous=False)

    pub = rospy.Publisher(
        "/diffusion_policy/action_chunk",
        Float64MultiArray,
        queue_size=1,
    )

    rate      = rospy.Rate(args.action_hz)
    t_start   = time.time()
    pub_count = 0

    rospy.loginfo(
        f"[SimNode] Starting\n"
        f"  action_hz      : {args.action_hz}\n"
        f"  n_action_steps : {args.n_action_steps}\n"
        f"  amplitude      : {args.amplitude} m\n"
        f"  home pos       : {PANDA_HOME_POS}\n"
        f"Publishing to    : /diffusion_policy/action_chunk\n"
        f"NOTE: for sim client only — do not use with hardware mode\n"
        f"Press Ctrl+C to stop."
    )

    while not rospy.is_shutdown():
        t_now = time.time() - t_start
        chunk = make_chunk(
            t_start=t_now,
            n_steps=args.n_action_steps,
            action_hz=args.action_hz,
            amplitude=args.amplitude,
        )

        if args.verbose:
            print_chunk(chunk, pub_count + 1)

        pub.publish(build_action_msg(chunk))
        pub_count += 1

        rospy.loginfo(
            f"[SimNode] chunk {pub_count:4d} | "
            f"x={chunk[0,0]:.3f} y={chunk[0,1]:.3f} z={chunk[0,2]:.3f} "
            f"(norm) | grip={'open' if chunk[0,6] > 0.5 else 'close'}"
        )

        rate.sleep()


if __name__ == "__main__":
    main()