#!/usr/bin/env python3
"""
sim_node.py  (NPZ demo replay version)
----------------------------------------
Replays a recorded demonstration from an NPZ file by publishing raw sensor
data on the same topics that observation_node.py subscribes to.

Also seeds the Polymetis server (real or PyBullet sim) to the demo's first
joint configuration so that observation_node.py (which polls Polymetis
directly) starts from the correct robot state.

observation_node.py then preprocesses the data identically to training,
and publishes to /diffusion_policy/observation for eval_real.py.

NPZ keys expected:
    images1          (T, H, W, 3)  uint8   — cam1 (wrist)
    images2          (T, H, W, 3)  uint8   — cam2 (external)
    joints           (T, 7)        float64
    gripper_pos      (T, 2)        float64

Setup:
    # Terminal 1: ROS master
    roscore

    # Terminal 2: PyBullet sim server
    launch_robot.py robot_client=bullet_sim gui=true

    # Terminal 3: controller node in sim mode
    python controller_node.py --mode sim

    # Terminal 4: observation node
    python observation_node.py

    # Terminal 5: eval node
    python eval_real.py

    # Terminal 6: this script
    source /opt/ros/noetic/setup.bash
    python sim_node.py --npz /path/to/demo.npz [options]

Options:
    --npz           path to NPZ demo file (required)
    --replay_hz     playback rate Hz    (default: 10, match observation_node)
    --loop          loop the demo continuously
    --cam1_topic    topic for wrist camera    (default: /eih/color/image_raw)
    --cam2_topic    topic for external camera (default: /ext/color/image_raw)
    --joint_topic   topic for joint states    (default: /franka_state_controller/joint_states)
    --gripper_topic topic for gripper states  (default: /franka_gripper/joint_states)
    --robot_ip      Polymetis server IP       (default: localhost for sim)
    --seed_time     seconds to move to start pose (default: 5.0)
    --skip_seed     skip seeding Polymetis start pose
    --verbose       print per-step info
"""

import argparse
import sys

import numpy as np
import torch
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from polymetis import RobotInterface


# ── NPZ loader ────────────────────────────────────────────────────────────────

def load_demo(npz_path: str) -> dict:
    """
    Load a single demonstration from an NPZ file.
    Returns dict with numpy arrays for each sensor stream.
    """
    rospy.loginfo(f"[SimNode] Loading NPZ: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["images1", "images2", "joints", "gripper_pos"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"NPZ file is missing required keys: {missing}\n"
            f"Found keys: {list(data.keys())}"
        )

    images1     = data["images1"]     # (T, H, W, 3)  uint8
    images2     = data["images2"]     # (T, H, W, 3)  uint8
    joints      = data["joints"]      # (T, 7)         float64
    gripper_pos = data["gripper_pos"] # (T, 2)         float64

    T = images1.shape[0]

    rospy.loginfo(
        f"[SimNode] Demo loaded ✓\n"
        f"  steps          : {T}\n"
        f"  image shape    : {images1.shape[1:]}\n"
        f"  joint shape    : {joints.shape[1:]}\n"
        f"  gripper shape  : {gripper_pos.shape[1:]}\n"
    )

    return {
        "images1":     images1,
        "images2":     images2,
        "joints":      joints,
        "gripper_pos": gripper_pos,
        "n_steps":     T,
    }


# ── Polymetis start pose seeding ──────────────────────────────────────────────

def seed_polymetis_start_pose(joints_t0: np.ndarray,
                               robot_ip: str = "localhost",
                               time_to_go: float = 5.0):
    """
    Move the Polymetis server (real or PyBullet) to the demo's first joint
    configuration so observation_node.py starts from the correct state.

    observation_node.py polls Polymetis directly via RobotInterface, so
    camera images are the only thing still driven by ROS topic replay.
    Joint state seen by observation_node will be frozen at joints[0] for
    the duration of the replay unless you also drive it step-by-step.

    Args:
        joints_t0  : (7,) float64 — first-frame joint angles from NPZ [rad]
        robot_ip   : Polymetis server IP ("localhost" for PyBullet sim)
        time_to_go : seconds to reach the start pose
                     (use 5.0 for sim, 8-10 for hardware)
    """
    rospy.loginfo(
        f"[SimNode] Connecting to Polymetis @ {robot_ip} to seed start pose ..."
    )

    try:
        robot = RobotInterface(ip_address=robot_ip, enforce_version=False)
    except Exception as e:
        rospy.logerr(f"[SimNode] Could not connect to Polymetis: {e}")
        raise

    q0 = torch.tensor(joints_t0, dtype=torch.float32)
    rospy.loginfo(
        f"[SimNode] Moving to demo start joints:\n"
        f"  {np.round(joints_t0, 4)}\n"
        f"  time_to_go = {time_to_go}s"
    )

    robot.move_to_joint_positions(q0, time_to_go=time_to_go)
    rospy.loginfo("[SimNode] Start pose reached ✓")


# ── Publisher helpers ─────────────────────────────────────────────────────────

def make_image_msg(bridge: CvBridge, rgb_img: np.ndarray,
                   stamp: rospy.Time, frame_id: str = "camera") -> Image:
    """
    Publish as bgr8 — observation_node.py calls imgmsg_to_cv2(msg, "bgr8")
    which forces BGR output regardless of encoding.

    NPZ stores raw RGB from the data collector; observation_node expects BGR
    on the wire (it mirrors the real camera which publishes bgr8). So we
    flip RGB → BGR here before encoding.
    """
    bgr_img = rgb_img[..., ::-1].copy()   # RGB → BGR to match real camera
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
        description="Replay NPZ demo on raw sensor topics for observation_node")

    parser.add_argument("--npz",           type=str, required=True,
                        help="Path to NPZ demo file")
    parser.add_argument("--replay_hz",     type=float, default=10.0,
                        help="Playback rate Hz — match observation_node publish_hz (default: 10)")
    parser.add_argument("--loop",          action="store_true",
                        help="Loop demo continuously")
    parser.add_argument("--cam1_topic",    type=str, default="/eih/color/image_raw",
                        help="Wrist camera topic (default: /eih/color/image_raw)")
    parser.add_argument("--cam2_topic",    type=str, default="/ext/color/image_raw",
                        help="External camera topic (default: /ext/color/image_raw)")
    parser.add_argument("--joint_topic",   type=str,
                        default="/franka_state_controller/joint_states",
                        help="Joint states topic")
    parser.add_argument("--gripper_topic", type=str,
                        default="/franka_gripper/joint_states",
                        help="Gripper states topic")
    parser.add_argument("--robot_ip",      type=str, default="localhost",
                        help="Polymetis server IP (default: localhost for sim)")
    parser.add_argument("--seed_time",     type=float, default=5.0,
                        help="Seconds to move to start pose (default: 5.0, use 8-10 for hardware)")
    parser.add_argument("--skip_seed",     action="store_true",
                        help="Skip seeding Polymetis start pose (use if robot is already positioned)")
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-step info")

    args, unknown = parser.parse_known_args()

    # ── init ROS ──────────────────────────────────────────────────────────────
    rospy.init_node("sim_node", anonymous=False)

    # ── load demo ─────────────────────────────────────────────────────────────
    demo = load_demo(args.npz)

    # ── seed Polymetis to demo start pose ─────────────────────────────────────
    # observation_node.py polls Polymetis directly, so we must put the robot
    # in the correct starting configuration before the observation buffer fills.
    # Camera images are still driven by this node's topic replay below.
    if not args.skip_seed:
        seed_polymetis_start_pose(
            joints_t0  = demo["joints"][0],
            robot_ip   = args.robot_ip,
            time_to_go = args.seed_time,
        )
    else:
        rospy.loginfo("[SimNode] Skipping Polymetis start pose seed (--skip_seed set).")

    # ── publishers — raw sensor topics, same as observation_node subscribes to ─
    bridge = CvBridge()

    pub_cam1    = rospy.Publisher(args.cam1_topic,    Image,      queue_size=1)
    pub_cam2    = rospy.Publisher(args.cam2_topic,    Image,      queue_size=1)
    pub_joints  = rospy.Publisher(args.joint_topic,   JointState, queue_size=1)
    pub_gripper = rospy.Publisher(args.gripper_topic, JointState, queue_size=1)

    # Joint names expected by franka_state_controller
    JOINT_NAMES   = [f"panda_joint{i}" for i in range(1, 8)]

    # Gripper finger names expected by franka_gripper
    GRIPPER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

    rate     = rospy.Rate(args.replay_hz)
    n_steps  = demo["n_steps"]
    step_idx = 0
    loop_num = 0

    rospy.loginfo(
        f"[SimNode] Starting replay\n"
        f"  npz            : {args.npz}\n"
        f"  n_steps        : {n_steps}\n"
        f"  replay_hz      : {args.replay_hz}\n"
        f"  loop           : {args.loop}\n"
        f"  robot_ip       : {args.robot_ip}\n"
        f"  skip_seed      : {args.skip_seed}\n"
        f"  cam1 topic     : {args.cam1_topic}\n"
        f"  cam2 topic     : {args.cam2_topic}\n"
        f"  joint topic    : {args.joint_topic}\n"
        f"  gripper topic  : {args.gripper_topic}\n"
        f"Press Ctrl+C to stop."
    )

    # Brief pause so subscribers (observation_node) can connect
    rospy.sleep(1.0)

    while not rospy.is_shutdown():
        if step_idx >= n_steps:
            if args.loop:
                step_idx = 0
                loop_num += 1
                rospy.loginfo(f"[SimNode] Loop {loop_num} restarting demo ...")
                # Re-seed start pose at the beginning of each loop
                if not args.skip_seed:
                    seed_polymetis_start_pose(
                        joints_t0  = demo["joints"][0],
                        robot_ip   = args.robot_ip,
                        time_to_go = args.seed_time,
                    )
                rospy.sleep(0.5)
                continue
            else:
                rospy.loginfo("[SimNode] Demo finished. Shutting down.")
                break

        stamp = rospy.Time.now()

        # ── publish raw camera images ──────────────────────────────────────────
        # NPZ stores RGB; make_image_msg flips to BGR to match real camera output
        # which observation_node.py expects (it calls imgmsg_to_cv2(msg, "bgr8"))
        img1_msg = make_image_msg(
            bridge, demo["images1"][step_idx], stamp, frame_id="camera_wrist")
        img2_msg = make_image_msg(
            bridge, demo["images2"][step_idx], stamp, frame_id="camera_ext")

        pub_cam1.publish(img1_msg)
        pub_cam2.publish(img2_msg)

        # ── publish joint states (informational — observation_node uses Polymetis) ─
        # These are published for any other nodes that may be listening, but
        # observation_node.py ignores them and polls Polymetis directly.
        joint_msg = make_joint_state_msg(
            demo["joints"][step_idx], JOINT_NAMES, stamp)
        pub_joints.publish(joint_msg)

        # ── publish gripper states (informational — same caveat as joints) ────
        gripper_msg = make_joint_state_msg(
            demo["gripper_pos"][step_idx], GRIPPER_NAMES, stamp)
        pub_gripper.publish(gripper_msg)

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