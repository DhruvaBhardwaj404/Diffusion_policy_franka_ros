#!/usr/bin/env python3
"""
sim_node.py  (NPZ demo replay version)
----------------------------------------
Replays a recorded demonstration from an NPZ file by publishing raw sensor
data on the same topics that observation_node.py subscribes to.

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
    --verbose       print per-step info
"""

import argparse
import sys

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState


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


# ── Publisher helpers ─────────────────────────────────────────────────────────

def make_image_msg(bridge: CvBridge, rgb_img: np.ndarray,
                   stamp: rospy.Time, frame_id: str = "camera") -> Image:
    """
    Publish as rgb8 — observation_node.py calls imgmsg_to_cv2() without
    a desired encoding, so it will receive what we send.
    NPZ stores raw RGB from the data collector, so we send RGB directly.
    """
    msg = bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
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
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-step info")

    args, unknown = parser.parse_known_args()

    # ── init ROS ──────────────────────────────────────────────────────────────
    rospy.init_node("sim_node", anonymous=False)

    # ── load demo ─────────────────────────────────────────────────────────────
    demo = load_demo(args.npz)

    # ── publishers — raw sensor topics, same as observation_node subscribes to ─
    bridge = CvBridge()

    pub_cam1    = rospy.Publisher(args.cam1_topic,    Image,      queue_size=1)
    pub_cam2    = rospy.Publisher(args.cam2_topic,    Image,      queue_size=1)
    pub_joints  = rospy.Publisher(args.joint_topic,   JointState, queue_size=1)
    pub_gripper = rospy.Publisher(args.gripper_topic, JointState, queue_size=1)

    # Joint names expected by franka_state_controller
    JOINT_NAMES  = [f"panda_joint{i}" for i in range(1, 8)]

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
                rospy.sleep(0.5)
                continue
            else:
                rospy.loginfo("[SimNode] Demo finished. Shutting down.")
                break

        stamp = rospy.Time.now()

        # ── publish raw camera images (uint8 RGB — identical to data collector) ─
        img1_msg = make_image_msg(
            bridge, demo["images1"][step_idx], stamp, frame_id="camera_wrist")
        img2_msg = make_image_msg(
            bridge, demo["images2"][step_idx], stamp, frame_id="camera_ext")

        pub_cam1.publish(img1_msg)
        pub_cam2.publish(img2_msg)

        # ── publish joint states ───────────────────────────────────────────────
        joint_msg = make_joint_state_msg(
            demo["joints"][step_idx], JOINT_NAMES, stamp)
        pub_joints.publish(joint_msg)

        # ── publish gripper states ─────────────────────────────────────────────
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