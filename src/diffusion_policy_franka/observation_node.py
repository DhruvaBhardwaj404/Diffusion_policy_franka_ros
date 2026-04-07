#!/usr/bin/env python3
"""
observation_node.py
-------------------
Mirrors the data collection pipeline exactly so that inference observations
are preprocessed identically to training data.

Data collector → HDF5 conversion → This node (must all match)

Topic mapping (from data collector):
    cam1  /eih/color/image_raw   → images1 → camera_image        (wrist)
    cam2  /ext/color/image_raw   → images2 → camera_wrist_image  (external)
    /franka_state_controller/joint_states  → joints (7,)
    /franka_gripper/joint_states           → gripper_pos (2,)

Preprocessing applied (matching data conversion script):
    Images : rgb8 → resize 84x84 INTER_AREA → rgb2RGB → /255 → CHW float32
    EEF    : FK via roboticstoolbox Panda → normalize_eef_pos()
    Gripper: normalize_gripper() per finger

Published topics:
    /diffusion_policy/observation  (std_msgs/Float64MultiArray)
"""

import threading
from collections import deque

import cv2
import numpy as np
import rospy
import roboticstoolbox as rtb
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


# ── Normalisation constants — must match data conversion script exactly ────────

DESIRED_H = 84
DESIRED_W = 84

EEF_POS_LOWER_LIMITS  = np.array([0.3,  0.02, 0.07],        dtype=np.float32)
EEF_POS_UPPER_LIMITS  = np.array([0.65, 0.25, 0.55],        dtype=np.float32)

ROBOT_FINGER_OPEN     = 0.041   # metres
ROBOT_FINGER_CLOSED   = 0.018   # metres


# ── Preprocessing (identical to data conversion script) ───────────────────────

def normalize_eef_pos(pos: np.ndarray) -> np.ndarray:
    pos_range = EEF_POS_UPPER_LIMITS - EEF_POS_LOWER_LIMITS
    return 2.0 * (pos - EEF_POS_LOWER_LIMITS) / pos_range - 1.0


def normalize_gripper(gripper_pos: np.ndarray) -> np.ndarray:
    gripper_range = ROBOT_FINGER_OPEN - ROBOT_FINGER_CLOSED
    if gripper_range == 0:
        return gripper_pos.copy()
    return 2.0 * (gripper_pos - ROBOT_FINGER_CLOSED) / gripper_range - 1.0


def preprocess_image(rgb_img: np.ndarray) -> np.ndarray:
    """
    Matches resize_images() in data conversion script exactly:
      (H, W, 3) uint8 rgb  →  (3, 84, 84) float32 [0,1] RGB CHW

    The data collector stores raw rgb from cv_bridge.
    The conversion script does rgb→RGB before normalising.
    So we must do the same here.
    """
    rgb = cv2.resize(rgb_img, (DESIRED_W, DESIRED_H),
                         interpolation=cv2.INTER_AREA)
    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return chw   # (3, 84, 84)


def joints_to_eef(robot, q: np.ndarray):
    """
    Forward kinematics via roboticstoolbox — identical to joints_to_eef()
    in the data conversion script.

    Returns:
        norm_eef_pos  (3,)  normalised xyz
        eef_quat      (4,)  xyzw quaternion (scipy convention)
    """
    fk  = robot.fkine(q)
    mat = fk.A                                    # 4x4 homogeneous matrix
    pos = mat[:3, 3].astype(np.float32)
    quat = Rotation.from_matrix(
        mat[:3, :3]).as_quat().astype(np.float32) # xyzw

    norm_pos = normalize_eef_pos(pos)
    return norm_pos, quat


# ── ROS Node ──────────────────────────────────────────────────────────────────

class ObservationNode:
    def __init__(self):
        rospy.init_node("observation_node", anonymous=False)

        # ── params ────────────────────────────────────────────────────────────
        self.n_obs_steps = rospy.get_param("~n_obs_steps",  2)
        self.publish_hz  = rospy.get_param("~publish_hz",   10)

        # Mirror the data collector's topic params exactly
        self.cam1_topic = rospy.get_param("~cam1_topic", "/eih/color/image_raw")
        self.cam2_topic = rospy.get_param("~cam2_topic", "/ext/color/image_raw")

        # ── FK model — same as data conversion script ─────────────────────────
        rospy.loginfo("[ObsNode] Loading Franka FK model ...")
        self.robot = rtb.models.Panda()
        rospy.loginfo("[ObsNode] FK model ready ✓")

        # ── latest snapshots (mirrors data collector pattern) ─────────────────
        self.latest_image_1  = None   # (H, W, 3) uint8 rgb
        self.latest_image_2  = None   # (H, W, 3) uint8 rgb
        self.latest_joints   = None   # (7,) float64
        self.latest_gripper  = None   # (2,) float64  per-finger positions
        self.snap_lock       = threading.Lock()

        # ── rolling obs buffers ───────────────────────────────────────────────
        self.buf_cam1     = deque(maxlen=self.n_obs_steps)  # (3,84,84) float32
        self.buf_cam2     = deque(maxlen=self.n_obs_steps)  # (3,84,84) float32
        self.buf_eef_pos  = deque(maxlen=self.n_obs_steps)  # (3,)      float32
        self.buf_eef_quat = deque(maxlen=self.n_obs_steps)  # (4,)      float32
        self.buf_gripper  = deque(maxlen=self.n_obs_steps)  # (2,)      float32
        self.buf_lock     = threading.Lock()

        self.bridge = CvBridge()

        # ── publishers ────────────────────────────────────────────────────────
        self.obs_pub = rospy.Publisher(
            "/diffusion_policy/observation",
            Float64MultiArray,
            queue_size=1
        )

        # ── subscribers — same topics as data collector ───────────────────────
        rospy.Subscriber(self.cam1_topic, Image,
                         self.cam1_callback, queue_size=1, buff_size=2**24)

        rospy.Subscriber(self.cam2_topic, Image,
                         self.cam2_callback, queue_size=1, buff_size=2**24)

        # joints from franka_state_controller (same as collector)
        rospy.Subscriber("/franka_state_controller/joint_states", JointState,
                         self.joint_callback, queue_size=1)

        # gripper from franka_gripper (same as collector)
        rospy.Subscriber("/franka_gripper/joint_states", JointState,
                         self.gripper_callback, queue_size=1)

        # ── snapshot → buffer timer at publish_hz ─────────────────────────────
        # Mirrors the data collector's timer_callback pattern:
        # snapshot all latest values together at 10 Hz
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_hz),
            self.timer_callback
        )

        rospy.loginfo(
            f"[ObsNode] Ready\n"
            f"  n_obs_steps : {self.n_obs_steps}\n"
            f"  publish_hz  : {self.publish_hz}\n"
            f"  cam1 (wrist)   : {self.cam1_topic}\n"
            f"  cam2 (external): {self.cam2_topic}"
        )

    # ── sensor callbacks — store latest snapshot only ─────────────────────────
    # Identical pattern to the data collector

    def cam1_callback(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg)
            if rgb is not None and rgb.size > 0:
                with self.snap_lock:
                    self.latest_image_1 = np.array(rgb, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(f"[ObsNode] cam1 error: {e}")

    def cam2_callback(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg)
            if rgb is not None and rgb.size > 0:
                with self.snap_lock:
                    self.latest_image_2 = np.array(rgb, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(f"[ObsNode] cam2 error: {e}")

    def joint_callback(self, msg: JointState):
        if len(msg.position) >= 7:
            with self.snap_lock:
                self.latest_joints = np.array(msg.position[:7], dtype=np.float64)

    def gripper_callback(self, msg: JointState):
        """Gripper from /franka_gripper/joint_states — same as data collector."""
        if len(msg.position) >= 2:
            with self.snap_lock:
                self.latest_gripper = np.array(msg.position[:2], dtype=np.float64)

    # ── core loop — mirrors timer_callback in data collector ──────────────────

    def timer_callback(self, event=None):
        """
        Runs at 10 Hz. Snapshots all sensors, preprocesses, appends to
        rolling buffers, then publishes if buffers are full.
        Mirrors the data collector's timer_callback exactly.
        """
        with self.snap_lock:
            img1    = self.latest_image_1
            img2    = self.latest_image_2
            joints  = self.latest_joints
            gripper = self.latest_gripper

        # wait for all sensors — same check as data collector
        if any(v is None for v in [img1, img2, joints, gripper]):
            rospy.logwarn_throttle(
                2.0,
                f"[ObsNode] Waiting for sensors: "
                f"cam1={'ok' if img1 is not None else 'MISSING'} "
                f"cam2={'ok' if img2 is not None else 'MISSING'} "
                f"joints={'ok' if joints is not None else 'MISSING'} "
                f"gripper={'ok' if gripper is not None else 'MISSING'}"
            )
            return

        # ── preprocess snapshot ───────────────────────────────────────────────
        proc_img1 = preprocess_image(img1)                          # (3,84,84)
        proc_img2 = preprocess_image(img2)                          # (3,84,84)
        norm_pos, quat = joints_to_eef(self.robot, joints)         # (3,), (4,)
        norm_gripper   = normalize_gripper(
            gripper.astype(np.float32))                             # (2,)

        # ── append to rolling buffers ─────────────────────────────────────────
        with self.buf_lock:
            self.buf_cam1.append(proc_img1)
            self.buf_cam2.append(proc_img2)
            self.buf_eef_pos.append(norm_pos)
            self.buf_eef_quat.append(quat)
            self.buf_gripper.append(norm_gripper)
            buf_len = len(self.buf_cam1)

        if buf_len < self.n_obs_steps:
            rospy.logwarn_throttle(
                2.0,
                f"[ObsNode] Buffer filling: {buf_len}/{self.n_obs_steps}"
            )
            return

        self._publish_obs()

    # ── publisher ─────────────────────────────────────────────────────────────

    def _publish_obs(self):
        """
        Packs stacked buffers into a flat Float64MultiArray.

        Flat layout (policy_node unpacks in the same order):
            robot0_eef_pos      n_obs_steps * 3
            robot0_eef_quat     n_obs_steps * 4
            robot0_eef_gpos     n_obs_steps * 2
            camera_image        n_obs_steps * 3 * 84 * 84   (cam1 = wrist)
            camera_wrist_image  n_obs_steps * 3 * 84 * 84   (cam2 = external)
        """
        with self.buf_lock:
            cam1     = np.stack(list(self.buf_cam1),     axis=0)  # (T,3,84,84)
            cam2     = np.stack(list(self.buf_cam2),     axis=0)  # (T,3,84,84)
            eef_pos  = np.stack(list(self.buf_eef_pos),  axis=0)  # (T,3)
            eef_quat = np.stack(list(self.buf_eef_quat), axis=0)  # (T,4)
            gripper  = np.stack(list(self.buf_gripper),  axis=0)  # (T,2)

        data = np.concatenate([
            eef_pos.flatten(),   # T*3
            eef_quat.flatten(),  # T*4
            gripper.flatten(),   # T*2
            cam1.flatten(),      # T*3*84*84  (camera_image = wrist = cam1)
            cam2.flatten(),      # T*3*84*84  (camera_wrist_image = external = cam2)
        ]).astype(np.float32)

        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="obs_flat",
                                size=len(data), stride=len(data))
        ]
        msg.data = data.tolist()
        self.obs_pub.publish(msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = ObservationNode()
    node.spin()