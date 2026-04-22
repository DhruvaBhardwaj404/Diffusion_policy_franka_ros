#!/usr/bin/env python3
"""
sim_node.py  (NPZ demo replay version)
----------------------------------------
Flow:
  1. Load NPZ demo file.
  2. Connect to Polymetis -> move robot to demo start (home) joints.
  3. Briefly start then terminate cartesian impedance to cycle Polymetis
     back to a clean gravity-comp idle so controller_node's
     start_cartesian_impedance() call succeeds.
  4. Disconnect from Polymetis.
  5. Wait for user to press Enter.
  6. Create ROS publishers and start replaying camera images onto sensor topics.

observation_node.py subscribes to the camera topics published here, and polls
Polymetis directly for joint/gripper state (frozen at home after step 2).

NPZ keys expected:
    images1          (T, H, W, 3)  uint8   -- cam1 (wrist)
    images2          (T, H, W, 3)  uint8   -- cam2 (external)
    joints           (T, 7)        float64
    gripper_pos      (T, 2)        float64

Startup order:
    1. roscore
    2. launch_robot.py robot_client=bullet_sim gui=true
    3. python sim_node.py --npz demo.npz          <- moves to home, cycles impedance
    4. python controller_node.py                  <- start_cartesian_impedance() works
    5. python observation_node.py
    6. python eval_real.py
    7. press Enter in sim_node terminal           <- image replay begins

Options:
    --npz           path to NPZ demo file (required)
    --replay_hz     playback rate Hz           (default: 10)
    --loop          loop the demo continuously
    --robot_ip      Polymetis server IP         (default: localhost)
    --seed_time     seconds to move to home     (default: 5.0)
    --skip_seed     skip the home move entirely
    --cam1_topic    (default: /eih/color/image_raw)
    --cam2_topic    (default: /ext/color/image_raw)
    --verbose       print per-step info
"""

import argparse

import numpy as np
import torch
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from polymetis import RobotInterface


# ── NPZ loader ────────────────────────────────────────────────────────────────

def load_demo(npz_path: str) -> dict:
    rospy.loginfo(f"[SimNode] Loading NPZ: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["images1", "images2", "joints", "gripper_pos"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"NPZ file is missing required keys: {missing}\n"
            f"Found keys: {list(data.keys())}"
        )

    images1     = data["images1"]      # (T, H, W, 3) uint8
    images2     = data["images2"]      # (T, H, W, 3) uint8
    joints      = data["joints"]       # (T, 7)        float64
    gripper_pos = data["gripper_pos"]  # (T, 2)        float64
    T           = images1.shape[0]

    rospy.loginfo(
        f"[SimNode] Demo loaded\n"
        f"  steps       : {T}\n"
        f"  image shape : {images1.shape[1:]}\n"
        f"  joint shape : {joints.shape[1:]}\n"
        f"  gripper     : {gripper_pos.shape[1:]}"
    )

    return {
        "images1":     images1,
        "images2":     images2,
        "joints":      joints,
        "gripper_pos": gripper_pos,
        "n_steps":     T,
    }


# ── Polymetis home move ───────────────────────────────────────────────────────

def move_to_home(joints_t0: np.ndarray,
                 robot_ip: str = "localhost",
                 time_to_go: float = 5.0):
    """
    Connect to Polymetis, move the robot to joints_t0, then cycle through
    cartesian impedance once and terminate it.

    Why the impedance cycle:
        move_to_joint_positions() completes and leaves Polymetis in a hard
        idle state where a subsequent start_cartesian_impedance() call (from
        controller_node) fails with "unable to update desired joint position".
        Briefly starting and then terminating cartesian impedance here resets
        the server to a clean gravity-comp idle that accepts the next
        start_cartesian_impedance() call without any changes to controller_node.

    Args:
        joints_t0  : (7,) float64 -- target joint angles [rad]
        robot_ip   : Polymetis server IP ("localhost" for PyBullet sim)
        time_to_go : seconds for the move (5 for sim, 8-10 for hardware)
    """
    rospy.loginfo(f"[SimNode] Connecting to Polymetis @ {robot_ip} ...")

    try:
        robot = RobotInterface(ip_address=robot_ip, enforce_version=False)
    except Exception as e:
        rospy.logerr(f"[SimNode] Could not connect to Polymetis: {e}")
        raise

    # stop any policy already running from a previous session
    try:
        robot.terminate_current_policy()
        rospy.loginfo("[SimNode] Terminated existing policy.")
        rospy.sleep(0.3)
    except Exception:
        pass  # nothing was running -- fine

    q0 = torch.tensor(joints_t0, dtype=torch.float32)
    rospy.loginfo(
        f"[SimNode] Moving to home:\n"
        f"  joints     = {np.round(joints_t0, 4)}\n"
        f"  time_to_go = {time_to_go}s"
    )

    robot.move_to_joint_positions(q0, time_to_go=time_to_go)
    rospy.loginfo("[SimNode] Home reached ✓")

    # ── impedance cycle ────────────────────────────────────────────────────
    # move_to_joint_positions leaves Polymetis in hard idle. Briefly starting
    # and terminating cartesian impedance resets it to a clean state so
    # controller_node's start_cartesian_impedance() works without modification.
    rospy.sleep(0.5)
    try:
        robot.start_cartesian_impedance()
        rospy.loginfo("[SimNode] Impedance cycle: started ...")
        rospy.sleep(0.5)
        robot.terminate_current_policy()
        rospy.loginfo("[SimNode] Impedance cycle: terminated ✓")
        rospy.loginfo("[SimNode] Polymetis is in clean idle — controller_node can connect.")
    except Exception as e:
        rospy.logwarn(
            f"[SimNode] Impedance cycle failed: {e}\n"
            f"  controller_node may still fail — try restarting the sim server."
        )

    rospy.sleep(0.3)
    del robot
    rospy.loginfo("[SimNode] Polymetis connection released ✓")


# ── Publisher helpers ─────────────────────────────────────────────────────────

def make_image_msg(bridge: CvBridge,
                   rgb_img: np.ndarray,
                   stamp: rospy.Time,
                   frame_id: str = "camera") -> Image:
    """
    NPZ stores images as RGB (from the data collector).
    observation_node.py calls imgmsg_to_cv2(msg, "bgr8"), so we must send BGR.
    Flip RGB -> BGR before encoding.
    """
    bgr_img = rgb_img[..., ::-1].copy()
    msg = bridge.cv2_to_imgmsg(bgr_img, encoding="bgr8")
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    return msg


def make_joint_state_msg(positions: np.ndarray,
                          joint_names: list,
                          stamp: rospy.Time) -> JointState:
    msg                 = JointState()
    msg.header.stamp    = stamp
    msg.header.frame_id = ""
    msg.name            = joint_names
    msg.position        = positions.tolist()
    msg.velocity        = [0.0] * len(positions)
    msg.effort          = [0.0] * len(positions)
    return msg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NPZ demo replay: move to home -> wait for Enter -> replay images")

    parser.add_argument("--npz",           type=str,   required=True,
                        help="Path to NPZ demo file")
    parser.add_argument("--replay_hz",     type=float, default=10.0,
                        help="Image publish rate Hz -- match observation_node publish_hz (default: 10)")
    parser.add_argument("--loop",          action="store_true",
                        help="Loop the image replay continuously")
    parser.add_argument("--robot_ip",      type=str,   default="localhost",
                        help="Polymetis server IP (default: localhost for sim)")
    parser.add_argument("--seed_time",     type=float, default=5.0,
                        help="Seconds to move to home pose (default: 5.0, use 8-10 for hardware)")
    parser.add_argument("--skip_seed",     action="store_true",
                        help="Skip the home move (robot already in position)")
    parser.add_argument("--cam1_topic",    type=str,   default="/eih/color/image_raw",
                        help="Wrist camera topic (default: /eih/color/image_raw)")
    parser.add_argument("--cam2_topic",    type=str,   default="/ext/color/image_raw",
                        help="External camera topic (default: /ext/color/image_raw)")
    parser.add_argument("--joint_topic",   type=str,   default="/franka_state_controller/joint_states",
                        help="Joint states topic (informational only)")
    parser.add_argument("--gripper_topic", type=str,   default="/franka_gripper/joint_states",
                        help="Gripper states topic (informational only)")
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-step info")

    args, unknown = parser.parse_known_args()

    # ── init ROS ──────────────────────────────────────────────────────────────
    rospy.init_node("sim_node", anonymous=False)

    # ── step 1: load demo ─────────────────────────────────────────────────────
    demo = load_demo(args.npz)

    # ── step 2: move to home and reset Polymetis to clean idle ────────────────
    if not args.skip_seed:
        move_to_home(
            joints_t0  = demo["joints"][0],
            robot_ip   = args.robot_ip,
            time_to_go = args.seed_time,
        )
    else:
        rospy.loginfo("[SimNode] --skip_seed set -- skipping home move.")

    # ── step 3: wait for user confirmation ────────────────────────────────────
    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("[SimNode] Robot is at home. Polymetis is in clean idle.")
    rospy.loginfo("[SimNode] Start controller_node, observation_node, eval_real.py now.")
    rospy.loginfo("[SimNode] Press Enter when ready to begin image replay ...")
    rospy.loginfo("=" * 60 + "\n")

    try:
        input()
    except EOFError:
        # non-interactive launch (e.g. roslaunch) -- proceed immediately
        rospy.loginfo("[SimNode] Non-interactive -- starting replay immediately.")

    # ── step 4: create publishers and start replaying ─────────────────────────
    bridge = CvBridge()

    pub_cam1    = rospy.Publisher(args.cam1_topic,    Image,      queue_size=1)
    pub_cam2    = rospy.Publisher(args.cam2_topic,    Image,      queue_size=1)
    pub_joints  = rospy.Publisher(args.joint_topic,   JointState, queue_size=1)
    pub_gripper = rospy.Publisher(args.gripper_topic, JointState, queue_size=1)

    # brief pause so observation_node subscribers can handshake with publishers
    rospy.sleep(0.5)

    JOINT_NAMES   = [f"panda_joint{i}" for i in range(1, 8)]
    GRIPPER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

    rate     = rospy.Rate(args.replay_hz)
    n_steps  = demo["n_steps"]
    step_idx = 0
    loop_num = 0

    rospy.loginfo(
        f"[SimNode] Replay started\n"
        f"  npz        : {args.npz}\n"
        f"  n_steps    : {n_steps}\n"
        f"  replay_hz  : {args.replay_hz}\n"
        f"  loop       : {args.loop}\n"
        f"  cam1 topic : {args.cam1_topic}\n"
        f"  cam2 topic : {args.cam2_topic}\n"
        f"Press Ctrl+C to stop."
    )

    while not rospy.is_shutdown():

        if step_idx >= n_steps:
            if args.loop:
                step_idx = 0
                loop_num += 1
                rospy.loginfo(f"[SimNode] Loop {loop_num} -- restarting replay ...")
                rospy.sleep(0.5)
                continue
            else:
                rospy.loginfo("[SimNode] Replay finished.")
                break

        stamp = rospy.Time.now()

        # camera images -- NPZ is RGB, flip to BGR to match real camera on wire
        pub_cam1.publish(make_image_msg(
            bridge, demo["images1"][step_idx], stamp, frame_id="camera_wrist"))
        pub_cam2.publish(make_image_msg(
            bridge, demo["images2"][step_idx], stamp, frame_id="camera_ext"))

        # joint / gripper states -- informational only, observation_node uses Polymetis
        pub_joints.publish(make_joint_state_msg(
            demo["joints"][step_idx], JOINT_NAMES, stamp))
        pub_gripper.publish(make_joint_state_msg(
            demo["gripper_pos"][step_idx], GRIPPER_NAMES, stamp))

        if args.verbose:
            q  = demo["joints"][step_idx]
            gp = demo["gripper_pos"][step_idx]
            rospy.loginfo(
                f"[SimNode] step {step_idx:4d}/{n_steps} | "
                f"q[0:3]={np.round(q[:3], 4)} | "
                f"gripper={np.round(gp, 4)}"
            )

        step_idx += 1
        rate.sleep()


if __name__ == "__main__":
    main()