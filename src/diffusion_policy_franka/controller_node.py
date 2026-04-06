#!/usr/bin/env python3
"""
controller_node.py  (Polymetis version — hardware + sim modes)
---------------------------------------------------------------
Two modes via --mode flag:

  hardware  : connects to Polymetis server on RT machine over network
              RT machine must be running:
                launch_robot.py robot_client=franka_hardware ...
                launch_gripper.py gripper=franka_hand

  sim       : connects to Polymetis PyBullet sim server on localhost
              Start the sim server first (on this machine):
                launch_robot.py robot_client=bullet_sim gui=true
              Gripper is stubbed — polysim does not support gripper simulation.

The RobotInterface API is identical in both modes — only the IP and
gripper behaviour differ. Everything else (interpolator, queue, ROS
subscriber) is exactly the same.

Usage:
    # simulation (no RT machine needed)
    launch_robot.py robot_client=bullet_sim gui=true   # separate terminal
    python controller_node.py --mode sim

    # hardware
    python controller_node.py --mode hardware --robot_ip <rt_machine_ip>
"""

import argparse
import enum
import multiprocessing as mp
import time
import threading

import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray

from polymetis import RobotInterface
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


# ── Constants ─────────────────────────────────────────────────────────────────

EEF_POS_LOWER_LIMITS = np.array([0.3,  0.02, 0.07], dtype=np.float64)
EEF_POS_UPPER_LIMITS = np.array([0.65, 0.25, 0.55], dtype=np.float64)

GRIPPER_OPEN_WIDTH  = 0.08
GRIPPER_CLOSE_WIDTH = 0.04
GRIPPER_SPEED       = 0.05
GRIPPER_FORCE       = 10.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def unnormalize_eef_pos(norm_pos: np.ndarray) -> np.ndarray:
    pos_range = EEF_POS_UPPER_LIMITS - EEF_POS_LOWER_LIMITS
    return (norm_pos + 1.0) / 2.0 * pos_range + EEF_POS_LOWER_LIMITS


def actions_to_poses(actions: np.ndarray) -> np.ndarray:
    """(N,7) action → (N,7) pose  [xyz | xyzw quat]"""
    real_pos  = unnormalize_eef_pos(actions[:, :3])
    quats     = Rotation.from_euler('xyz', actions[:, 3:6]).as_quat()  # xyzw
    return np.concatenate([real_pos, quats], axis=1)


# ── Command enum ──────────────────────────────────────────────────────────────

class Command(enum.Enum):
    STOP              = 0
    SCHEDULE_WAYPOINT = 1


# ── Gripper abstraction ───────────────────────────────────────────────────────

class HardwareGripper:
    """Real Franka gripper via Polymetis GripperInterface."""

    def __init__(self, robot_ip: str):
        from polymetis import GripperInterface
        print(f"[Gripper] Connecting to {robot_ip} ...")
        self.gripper   = GripperInterface(ip_address=robot_ip)
        self.is_closed = False
        print("[Gripper] Ready ✓")

    def command(self, state: int):
        if state == 0 and not self.is_closed:
            self.gripper.grasp(
                speed=GRIPPER_SPEED,
                force=GRIPPER_FORCE,
                grasp_width=GRIPPER_CLOSE_WIDTH,
            )
            self.is_closed = True
            rospy.loginfo("[Gripper] → CLOSING")
        elif state == 1 and self.is_closed:
            self.gripper.goto(
                width=GRIPPER_OPEN_WIDTH,
                speed=GRIPPER_SPEED,
                force=GRIPPER_FORCE,
            )
            self.is_closed = False
            rospy.loginfo("[Gripper] → OPENING")


class SimGripper:
    """
    Stub gripper for sim mode.
    polysim does not support gripper simulation, so we just log commands.
    """

    def __init__(self):
        self.is_closed = False
        print("[Gripper] Sim mode — gripper commands will be logged only.")

    def command(self, state: int):
        if state == 0 and not self.is_closed:
            self.is_closed = True
            rospy.loginfo("[Gripper][SIM] → CLOSING (simulated)")
        elif state == 1 and self.is_closed:
            self.is_closed = False
            rospy.loginfo("[Gripper][SIM] → OPENING (simulated)")


# ── Franka controller process ─────────────────────────────────────────────────

class FrankaController(mp.Process):
    """
    Owns the Polymetis RobotInterface and runs PoseTrajectoryInterpolator
    at interp_hz in its own process (escapes Python GIL).

    Identical for both hardware and sim — only robot_ip differs:
        hardware : IP of RT machine
        sim      : "localhost"
    """

    def __init__(self,
                 command_queue: mp.Queue,
                 robot_ip: str,
                 interp_hz: float   = 200.0,
                 max_pos_speed: float = 0.25,
                 max_rot_speed: float = 0.6,
                 verbose: bool      = False):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.robot_ip      = robot_ip
        self.interp_hz     = interp_hz
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.verbose       = verbose
        self.ready_event   = mp.Event()

    def run(self):
        import torch

        print(f"[FrankaController] Connecting to Polymetis @ {self.robot_ip} ...")
        robot = RobotInterface(ip_address=self.robot_ip)
        robot.go_home()

        # seed interpolator from current EEF pose
        state  = robot.get_ee_pose()
        pos0   = state[0].numpy()
        # Polymetis returns wxyz → convert to xyzw for interpolator
        q_wxyz = state[1].numpy()
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        pose0  = np.concatenate([pos0, q_xyzw])   # (7,)
        t0     = time.monotonic()

        interp = PoseTrajectoryInterpolator(times=[t0], poses=[pose0])

        dt = 1.0 / self.interp_hz
        self.ready_event.set()
        print(f"[FrankaController] Running at {self.interp_hz} Hz")

        while True:
            t_now = time.monotonic()

            # ── drain command queue ───────────────────────────────────────────
            while not self.command_queue.empty():
                try:
                    cmd = self.command_queue.get_nowait()
                except Exception:
                    break

                if cmd['cmd'] == Command.STOP.value:
                    print("[FrankaController] STOP — exiting.")
                    return

                elif cmd['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = np.array(cmd['pose'])   # (7,) xyzw
                    target_time = float(cmd['time'])       # wall-clock

                    # wall-clock → monotonic  (same trick as RTDE version)
                    target_time_mono = (
                        time.monotonic() - time.time() + target_time
                    )

                    interp = interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time_mono,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                        curr_time=t_now,
                        last_waypoint_time=t_now,
                    )

            # ── interpolate → send ────────────────────────────────────────────
            pose_now  = interp(t_now)          # (7,) xyzw
            pos       = pose_now[:3]
            q_xyzw    = pose_now[3:]
            # Polymetis expects wxyz
            q_wxyz    = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

            robot.move_to_ee_pose(
                position=torch.tensor(pos,    dtype=torch.float32),
                orientation=torch.tensor(q_wxyz, dtype=torch.float32),
                time_to_go=dt * 2,
            )

            if self.verbose:
                print(f"[FrankaController] pos={np.round(pos, 4)}")

            # ── precise sleep ─────────────────────────────────────────────────
            elapsed = time.monotonic() - t_now
            time.sleep(max(0.0, dt - elapsed))


# ── ROS controller node ───────────────────────────────────────────────────────

class ControllerNode:

    def __init__(self, mode: str, robot_ip: str,
                 action_hz: float, interp_hz: float,
                 max_pos_speed: float, max_rot_speed: float,
                 verbose: bool):

        self.action_hz = action_hz

        # ── resolve robot IP based on mode ────────────────────────────────────
        if mode == "sim":
            effective_ip = "localhost"
            print("[ControllerNode] Mode: SIM  (connecting to localhost)")
        else:
            if not robot_ip:
                raise ValueError("--robot_ip is required for hardware mode")
            effective_ip = robot_ip
            print(f"[ControllerNode] Mode: HARDWARE  (robot_ip={robot_ip})")

        # ── start controller process ──────────────────────────────────────────
        self.command_queue = mp.Queue(maxsize=256)

        self.franka = FrankaController(
            command_queue=self.command_queue,
            robot_ip=effective_ip,
            interp_hz=interp_hz,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            verbose=verbose,
        )
        self.franka.start()
        print("[ControllerNode] Waiting for FrankaController ...")
        self.franka.ready_event.wait()
        print("[ControllerNode] FrankaController ready ✓")

        # ── gripper ───────────────────────────────────────────────────────────
        if mode == "sim":
            self.gripper = SimGripper()
        else:
            self.gripper = HardwareGripper(robot_ip=robot_ip)

        # ── ROS ───────────────────────────────────────────────────────────────
        rospy.init_node("controller_node", anonymous=False)

        rospy.Subscriber(
            "/diffusion_policy/action_chunk",
            Float64MultiArray,
            self._action_callback,
            queue_size=1,
        )

        rospy.loginfo("[ControllerNode] Ready — waiting for action chunks.")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _action_callback(self, msg: Float64MultiArray):
        dims       = msg.layout.dim
        n_steps    = dims[0].size
        action_dim = dims[1].size

        if action_dim != 7:
            rospy.logerr(
                f"[ControllerNode] Expected action_dim=7, got {action_dim}")
            return

        actions = np.array(msg.data, dtype=np.float64).reshape(n_steps, action_dim)

        threading.Thread(
            target=self._schedule_chunk, args=(actions,), daemon=True
        ).start()

    def _schedule_chunk(self, actions: np.ndarray):
        n_steps    = len(actions)
        poses      = actions_to_poses(actions)   # (N, 7)
        gripper    = actions[:, 6]               # (N,)

        t_start    = time.time()
        dt         = 1.0 / self.action_hz
        timestamps = t_start + np.arange(n_steps) * dt

        # enqueue waypoints — non-blocking
        for i in range(n_steps):
            self.command_queue.put({
                'cmd':  Command.SCHEDULE_WAYPOINT.value,
                'pose': poses[i].tolist(),
                'time': float(timestamps[i]),
            })

        # gripper sync at action_hz
        rate = rospy.Rate(self.action_hz)
        for i in range(n_steps):
            if rospy.is_shutdown():
                break
            self.gripper.command(int(round(gripper[i])))
            rate.sleep()

    def spin(self):
        rospy.spin()

    def shutdown(self):
        rospy.loginfo("[ControllerNode] Shutting down ...")
        self.command_queue.put({'cmd': Command.STOP.value})
        self.franka.join(timeout=3.0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diffusion policy controller node (Polymetis)")
    parser.add_argument(
        "--mode", choices=["hardware", "sim"], required=True,
        help="hardware: real Franka via RT machine | sim: local PyBullet")
    parser.add_argument(
        "--robot_ip", default=None,
        help="IP of RT machine (required for hardware mode)")
    parser.add_argument("--action_hz",     type=float, default=10.0)
    parser.add_argument("--interp_hz",     type=float, default=200.0,
                        help="Interpolation Hz (use 50 for sim, 200 for hardware)")
    parser.add_argument("--max_pos_speed", type=float, default=0.25,
                        help="Max EEF translation speed m/s")
    parser.add_argument("--max_rot_speed", type=float, default=0.6,
                        help="Max EEF rotation speed rad/s")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    node = ControllerNode(
        mode=args.mode,
        robot_ip=args.robot_ip,
        action_hz=args.action_hz,
        interp_hz=args.interp_hz,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
        verbose=args.verbose,
    )

    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()