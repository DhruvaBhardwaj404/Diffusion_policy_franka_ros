#!/usr/bin/env python3
"""
controller_node.py  (Polymetis version — hardware + sim modes)
"""

import enum
import multiprocessing as mp
import time
import threading
import sys
sys.path.insert(0, '/media/prabhav/SATA_SSD/dhruv/Diffusion-Transformer')
import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray
from polymetis import RobotInterface
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import torch
from polymetis import GripperInterface

# ── Constants ─────────────────────────────────────────────────────────────────


EEF_POS_LOWER_LIMITS   = np.array([ 0.15, -0.12,  0.13])
EEF_POS_UPPER_LIMITS   = np.array([ 0.65,  0.30,  0.60])

EEF_EULER_LOWER_LIMITS = np.array([-3.1416, -0.35, -2.40])
EEF_EULER_UPPER_LIMITS = np.array([ 3.1416,  0.40,  0.25])


BBOX_LOWER = np.array([-0.3,  -0.3, 0.1], dtype=np.float64)  # x_min, y_min, z_min
BBOX_UPPER = np.array([0.8,   0.5, 0.7], dtype=np.float64)  # x_max, y_max, z_max


BBOX_VIOLATION_MODE = "clamp"

GRIPPER_OPEN_WIDTH  = 0.08
GRIPPER_CLOSE_WIDTH = 0.04
GRIPPER_SPEED       = 0.05
GRIPPER_FORCE       = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def unnormalize_eef_pos(norm_pos: np.ndarray) -> np.ndarray:
    pos_range = EEF_POS_UPPER_LIMITS - EEF_POS_LOWER_LIMITS
    return (norm_pos + 1.0) / 2.0 * pos_range + EEF_POS_LOWER_LIMITS

def unnormalize_eef_euler(norm_euler: np.ndarray) -> np.ndarray:
    euler_range = EEF_EULER_UPPER_LIMITS - EEF_EULER_LOWER_LIMITS
    return (norm_euler + 1.0) / 2.0 * euler_range + EEF_EULER_LOWER_LIMITS


def actions_to_poses(actions: np.ndarray) -> np.ndarray:
    """(N, 7) action → (N, 7) pose  [xyz | xyzw quat]"""
    real_pos = unnormalize_eef_pos(actions[:, :3])
    real_euler = unnormalize_eef_euler(actions[:,3:6])
    quats    = Rotation.from_euler('xyz', real_euler).as_quat()  # xyzw
    return np.concatenate([real_pos, quats], axis=1)


# ── BBOX: Bounding box utilities ───────────────────────────────────────────────

def is_within_bbox(pos: np.ndarray) -> bool:
    """Return True if pos (3,) is inside [BBOX_LOWER, BBOX_UPPER]."""
    return bool(np.all(pos >= BBOX_LOWER) and np.all(pos <= BBOX_UPPER))


def clamp_to_bbox(pos: np.ndarray) -> np.ndarray:
    """Clamp pos (3,) to bbox boundary. Returns a new array."""
    return np.clip(pos, BBOX_LOWER, BBOX_UPPER)


def check_pose_bbox(pose: np.ndarray, label: str = "") -> tuple[bool, np.ndarray]:
    """
    Check and optionally correct a 7-element pose [xyz | xyzw].

    Returns
    -------
    (accepted, corrected_pose)
        accepted        : False only in "skip" mode when out-of-bounds
        corrected_pose  : pose with position clamped (clamp mode) or original
    """
    pos = pose[:3]
    if is_within_bbox(pos):
        return True, pose

    if BBOX_VIOLATION_MODE == "clamp":
        clamped_pos = clamp_to_bbox(pos)
        rospy.logwarn(
            f"[BBox{' ' + label if label else ''}] Position {np.round(pos, 4)} "
            f"out of bounds → clamped to {np.round(clamped_pos, 4)}"
        )
        corrected = pose.copy()
        corrected[:3] = clamped_pos
        return True, corrected
    else:  # "skip"
        rospy.logwarn(
            f"[BBox{' ' + label if label else ''}] Position {np.round(pos, 4)} "
            f"out of bounds → waypoint SKIPPED"
        )
        return False, pose
# ─────────────────────────────────────────────────────────────────────────────


# ── Command enum ──────────────────────────────────────────────────────────────

class Command(enum.Enum):
    STOP              = 0
    SCHEDULE_WAYPOINT = 1


# ── Gripper abstraction ───────────────────────────────────────────────────────

class HardwareGripper:
    def __init__(self, robot_ip: str):
        print(f"[Gripper] Connecting to {robot_ip} ...")
        self.gripper   = GripperInterface(ip_address=robot_ip)
        self.is_closed = True
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
    def __init__(self):
        self.is_closed = False
        print("[Gripper] Sim mode — gripper commands logged only.")

    def command(self, state: int):
        if state == 0 and not self.is_closed:
            self.is_closed = True
            rospy.loginfo("[Gripper][SIM] → CLOSING (simulated)")
        elif state == 1 and self.is_closed:
            self.is_closed = False
            rospy.loginfo("[Gripper][SIM] → OPENING (simulated)")


# ── Franka controller process ─────────────────────────────────────────────────
class FrankaController(mp.Process):
    def __init__(self, command_queue: mp.Queue, robot_ip: str, mode: str,  # Added mode
                 interp_hz: float = 200.0, max_pos_speed: float = 0.25,
                 max_rot_speed: float = 0.6, verbose: bool = False):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.robot_ip = robot_ip
        self.mode = mode  # Store mode
        self.interp_hz = interp_hz
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.verbose = verbose
        self.ready_event = mp.Event()

    def run(self):
        try:
            print(f"[FrankaController] Connecting to Polymetis @ {self.robot_ip} ...")
            robot = RobotInterface(ip_address=self.robot_ip)

            # Initialize Gripper inside the process
            if self.mode == "sim":
                gripper_provider = SimGripper()
            else:
                gripper_provider = HardwareGripper(robot_ip=self.robot_ip)

            state = robot.get_ee_pose()
            pos0 = state[0].numpy()
            q_xyzw = state[1].numpy()
            rotvec0 = Rotation.from_quat(q_xyzw).as_rotvec()
            pose0 = np.concatenate([pos0, rotvec0])

            t0 = time.monotonic()
            mono_wall_offset = time.monotonic() - time.time()

            interp = PoseTrajectoryInterpolator(times=[t0], poses=[pose0])
            gripper_tasks = []  # Track (monotonic_time, state)

            dt = 1.0 / self.interp_hz
            robot.start_cartesian_impedance()
            self.ready_event.set()

            while True:
                t_now = time.monotonic()

                while not self.command_queue.empty():
                    try:
                        cmd = self.command_queue.get_nowait()
                    except Exception:
                        break

                    if cmd['cmd'] == Command.STOP.value:
                        robot.terminate_current_policy()
                        return

                    elif cmd['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = np.array(cmd['pose'])
                        target_time_mono = mono_wall_offset + float(cmd['time'])

                        gripper_tasks.append((target_time_mono, cmd['gripper']))

                        t_pos = target_pose[:3]
                        t_rotvec = Rotation.from_quat(target_pose[3:]).as_rotvec()
                        interp = interp.schedule_waypoint(
                            pose=np.concatenate([t_pos, t_rotvec]),
                            time=target_time_mono,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=t_now,
                            last_waypoint_time=interp.times[-1],
                        )

                while gripper_tasks and t_now >= gripper_tasks[0][0]:
                    _, g_state = gripper_tasks.pop(0)
                    threading.Thread(
                        target=gripper_provider.command,
                        args=(g_state,),
                        daemon=True
                    ).start()

                pose_now = interp(t_now)
                q_xyzw_now = Rotation.from_rotvec(pose_now[3:]).as_quat()
                robot.update_desired_ee_pose(
                    position=torch.tensor(pose_now[:3], dtype=torch.float32),
                    orientation=torch.tensor(q_xyzw_now, dtype=torch.float32),
                )

                elapsed = time.monotonic() - t_now
                time.sleep(max(0.0, dt - elapsed))

        except Exception as e:
            print(f"[FrankaController] CRASHED: {e}")
            self.ready_event.set()

# ── ROS controller node ───────────────────────────────────────────────────────

class ControllerNode:

    def __init__(self):
        rospy.init_node("controller_node", anonymous=False)

        mode          = rospy.get_param("~mode",           "sim")
        robot_ip      = rospy.get_param("~robot_ip",       "")
        action_hz     = rospy.get_param("~action_hz",      10.0)
        interp_hz     = rospy.get_param("~interp_hz",      50.0 if mode == "sim" else 200.0)
        max_pos_speed = rospy.get_param("~max_pos_speed",  0.25)
        max_rot_speed = rospy.get_param("~max_rot_speed",  0.6)
        verbose       = rospy.get_param("~verbose",        False)

        self.action_hz = action_hz

        # ── BBOX: log bbox config on startup ─────────────────────────────────
        rospy.loginfo(
            f"[ControllerNode] Collision bbox\n"
            f"  lower : {BBOX_LOWER.tolist()} m\n"
            f"  upper : {BBOX_UPPER.tolist()} m\n"
            f"  mode  : {BBOX_VIOLATION_MODE}"
        )
        # ─────────────────────────────────────────────────────────────────────

        rospy.loginfo(
            f"[ControllerNode] Config\n"
            f"  mode          : {mode}\n"
            f"  robot_ip      : {robot_ip if robot_ip else 'localhost (sim)'}\n"
            f"  action_hz     : {action_hz}\n"
            f"  interp_hz     : {interp_hz}\n"
            f"  max_pos_speed : {max_pos_speed} m/s\n"
            f"  max_rot_speed : {max_rot_speed} rad/s\n"
        )

        if mode == "sim":
            effective_ip = "localhost"
            rospy.loginfo("[ControllerNode] Mode: SIM → localhost")
        else:
            if not robot_ip:
                rospy.logfatal("[ControllerNode] hardware mode requires ~robot_ip param")
                raise ValueError("~robot_ip is required for hardware mode")
            effective_ip = robot_ip
            rospy.loginfo(f"[ControllerNode] Mode: HARDWARE → {robot_ip}")

        self.command_queue = mp.Queue(maxsize=256)

        self.franka = FrankaController(
            command_queue=self.command_queue,
            robot_ip=effective_ip,
            mode=mode,
            interp_hz=interp_hz,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            verbose=verbose,
        )

        self.franka.start()
        rospy.loginfo("[ControllerNode] Waiting for FrankaController ...")
        self.franka.ready_event.wait()
        rospy.loginfo("[ControllerNode] FrankaController ready ✓")

        rospy.Subscriber(
            "/diffusion_policy/action_chunk",
            Float64MultiArray,
            self._action_callback,
            queue_size=1,
        )

        rospy.loginfo("[ControllerNode] Ready — waiting for action chunks.")

    def _action_callback(self, msg: Float64MultiArray):
        rospy.loginfo(f"[ControllerNode] franka process alive: {self.franka.is_alive()}")
        try:
            dims = msg.layout.dim
            n_steps    = dims[0].size
            action_dim = dims[1].size
            rospy.loginfo(f"[ControllerNode] n_steps={n_steps} action_dim={action_dim}")

            if action_dim != 7:
                rospy.logerr(f"[ControllerNode] Expected action_dim=7, got {action_dim}")
                return

            actions = np.array(msg.data, dtype=np.float64).reshape(n_steps, action_dim)
            threading.Thread(
                target=self._schedule_chunk, args=(actions,), daemon=True
            ).start()
        except Exception as e:
            rospy.logerr(f"[ControllerNode] _action_callback error: {e}")

    def _schedule_chunk(self, actions: np.ndarray):
        rospy.loginfo(f"[ControllerNode] _schedule_chunk called with {len(actions)} steps")
        rospy.loginfo(f"[ControllerNode] command_queue size: {self.command_queue.qsize()}")

        n_steps = len(actions)
        poses   = actions_to_poses(actions)   # (N, 7)  absolute xyz + xyzw
        gripper = actions[:, 6]

        t_start    = time.time()
        dt         = 1.0 / self.action_hz
        timestamps = t_start + np.arange(n_steps) * dt

        for i in range(n_steps):
            # ── BBOX: primary guard — filter before entering the queue ────────
            accepted, safe_pose = check_pose_bbox(poses[i], label=f"step{i}")
            if not accepted:
                continue  # skip mode: don't queue this waypoint
            # ─────────────────────────────────────────────────────────────────

            # self.command_queue.put({
            #     'cmd':  Command.SCHEDULE_WAYPOINT.value,
            #     'pose': safe_pose.tolist(),
            #     'time': float(timestamps[i]),
            # })

            self.command_queue.put({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'pose': safe_pose.tolist(),
                'time': float(timestamps[i]),
                'gripper': 1 if gripper[i] > 0 else 0
            })

        # rate = rospy.Rate(self.action_hz)
        # for i in range(n_steps):
        #     if rospy.is_shutdown():
        #         break
        #     self.gripper.command(1 if gripper[i] > 0 else 0)
        #     rate.sleep()

    def spin(self):
        rospy.spin()

    def shutdown(self):
        rospy.loginfo("[ControllerNode] Shutting down ...")
        self.command_queue.put({'cmd': Command.STOP.value})
        self.franka.join(timeout=3.0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    node = ControllerNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
