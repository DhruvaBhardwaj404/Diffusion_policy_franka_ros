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
    joints + gripper polled directly from Polymetis

Preprocessing applied (matching data conversion script):
    Images : bgr8 → BGR2RGB → resize 84x84 INTER_AREA → /255 → CHW float32
    EEF    : FK via Polymetis RobotModelPinocchio → normalize_eef_pos() + normalise_eef_euler()
    Gripper: normalize_gripper() per finger

Published topics:
    /diffusion_policy/observation  (std_msgs/Float64MultiArray)
"""

import threading
from collections import deque

import cv2
import numpy as np
import torch
import rospy
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from polymetis import RobotInterface, GripperInterface
from torchcontrol.models import RobotModelPinocchio


# ── Normalisation constants — must match data conversion script exactly ────────

DESIRED_H = 240
DESIRED_W  = 320

EEF_POS_LOWER_LIMITS   = np.array([ 0.15, -0.12,  0.13], dtype=np.float32)
EEF_POS_UPPER_LIMITS   = np.array([ 0.65,  0.30,  0.60], dtype=np.float32)

EEF_EULER_LOWER_LIMITS = np.array([-3.1416, -0.35, -2.40], dtype=np.float32)
EEF_EULER_UPPER_LIMITS = np.array([ 3.1416,  0.40,  0.25], dtype=np.float32)

ROBOT_FINGER_OPEN   = 0.045   # metres
ROBOT_FINGER_CLOSED = 0.015   # metres

# URDF for offline FK — must be the same URDF used during data conversion
URDF_PATH    = rospy.get_param("~urdf_path",  "src/panda.urdf")
EE_LINK_NAME = "panda_EndEffector"


# ── Normalisation (identical to data conversion script) ───────────────────────

def normalize_eef_pos(pos: np.ndarray) -> np.ndarray:
    pos_range = EEF_POS_UPPER_LIMITS - EEF_POS_LOWER_LIMITS
    return 2.0 * (pos - EEF_POS_LOWER_LIMITS) / pos_range - 1.0


def normalise_eef_euler(euler: np.ndarray) -> np.ndarray:
    euler_range = EEF_EULER_UPPER_LIMITS - EEF_EULER_LOWER_LIMITS
    return 2.0 * (euler - EEF_EULER_LOWER_LIMITS) / euler_range - 1.0


def normalize_gripper(gripper_pos: np.ndarray) -> np.ndarray:
    gripper_range = ROBOT_FINGER_OPEN - ROBOT_FINGER_CLOSED
    if gripper_range == 0:
        return gripper_pos.copy()
    return 2.0 * (gripper_pos - ROBOT_FINGER_CLOSED) / gripper_range - 1.0


# ── Image preprocessing (identical to resize_images() in converter) ───────────

def preprocess_image(bgr_img: np.ndarray) -> np.ndarray:
    """
    (H, W, 3) uint8 BGR  →  (3, DESIRED_H, DESIRED_W) float32 [0,1] RGB CHW

    Matches resize_images() in data conversion script exactly:
      - BGR → RGB  (camera publishes bgr8, same as data collector)
      - resize to DESIRED_H x DESIRED_W with INTER_AREA
      - /255 normalise
      - HWC → CHW transpose
    """
    resized = cv2.resize(bgr_img, (DESIRED_W, DESIRED_H),
                         interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw     = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return chw   # (3, DESIRED_H, DESIRED_W)


# ── FK via Polymetis (identical to joints_to_eef() in converter) ──────────────

def joints_to_eef(robot_model, q: np.ndarray):
    """
    Forward kinematics via Polymetis RobotModelPinocchio.
    Matches joints_to_eef() in the data conversion script exactly.

    Args:
        robot_model : RobotModelPinocchio instance
        q           : (7,) float64 joint angles [rad]

    Returns:
        norm_eef_pos   (3,)  normalised xyz
        norm_eef_euler (3,)  normalised XYZ euler [rad]
        eef_quat       (4,)  raw xyzw quaternion
    """
    joint_tensor = torch.tensor(q, dtype=torch.float32)
    pos_t, quat_t = robot_model.forward_kinematics(joint_tensor)

    pos  = pos_t.detach().numpy().astype(np.float32)
    quat = quat_t.detach().numpy().astype(np.float32)          # xyzw
    euler = R.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)

    norm_pos   = normalize_eef_pos(pos)
    norm_euler = normalise_eef_euler(euler)
    print(pos,euler)
    return norm_pos, norm_euler, quat


# ── ROS Node ──────────────────────────────────────────────────────────────────

class ObservationNode:
    def __init__(self):
        rospy.init_node("observation_node", anonymous=False)

        # ── params ────────────────────────────────────────────────────────────
        self.n_obs_steps = rospy.get_param("~n_obs_steps",  2)
        self.publish_hz  = rospy.get_param("~publish_hz",   10)
        self.robot_ip    = rospy.get_param("~robot_ip",     "10.42.0.1")
        self.poll_hz     = rospy.get_param("~poll_hz",      100)
        self.urdf_path   = rospy.get_param("~urdf_path",    URDF_PATH)
        self.ee_link     = rospy.get_param("~ee_link",      EE_LINK_NAME)

        self.cam1_topic  = rospy.get_param("~cam1_topic", "/ext/color/image_raw")  #temporary fix because policy trained on switched labels
        self.cam2_topic  = rospy.get_param("~cam2_topic", "/eih/color/image_raw")

        # ── FK model (Polymetis Pinocchio — same as converter) ────────────────
        rospy.loginfo(f"[ObsNode] Loading Pinocchio model: {self.urdf_path} (ee: {self.ee_link})")
        self.robot_model = RobotModelPinocchio(self.urdf_path, self.ee_link)
        rospy.loginfo("[ObsNode] FK model ready ✓")

        # ── connect to Polymetis ──────────────────────────────────────────────
        rospy.loginfo(f"[ObsNode] Connecting to Polymetis @ {self.robot_ip} ...")
        self.poly_robot   = RobotInterface(ip_address=self.robot_ip)
        self.poly_gripper = GripperInterface(ip_address=self.robot_ip)
        rospy.loginfo("[ObsNode] Polymetis connected ✓")

        # ── latest snapshots ──────────────────────────────────────────────────
        self.latest_image_1  = None   # (H, W, 3) uint8 BGR
        self.latest_image_2  = None   # (H, W, 3) uint8 BGR
        self.latest_joints   = None   # (7,) float64
        self.latest_gripper  = None   # (2,) float64 per-finger positions
        self.snap_lock       = threading.Lock()

        # ── rolling obs buffers ───────────────────────────────────────────────
        self.buf_cam1      = deque(maxlen=self.n_obs_steps)
        self.buf_cam2      = deque(maxlen=self.n_obs_steps)
        self.buf_eef_pos   = deque(maxlen=self.n_obs_steps)
        self.buf_eef_euler = deque(maxlen=self.n_obs_steps)
        self.buf_eef_quat  = deque(maxlen=self.n_obs_steps)
        self.buf_gripper   = deque(maxlen=self.n_obs_steps)
        self.buf_lock      = threading.Lock()

        self.bridge = CvBridge()

        # ── publisher ─────────────────────────────────────────────────────────
        self.obs_pub = rospy.Publisher(
            "/diffusion_policy/observation",
            Float64MultiArray,
            queue_size=1
        )

        # ── camera subscribers (bgr8 — matches data collector exactly) ────────
        rospy.Subscriber(self.cam1_topic, Image,
                         self.cam1_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.cam2_topic, Image,
                         self.cam2_callback, queue_size=1, buff_size=2**24)

        # ── Polymetis polling thread ──────────────────────────────────────────
        self._poly_thread = threading.Thread(
            target=self._polymetis_poll_loop, daemon=True
        )
        self._poly_thread.start()
        rospy.loginfo("[ObsNode] Polymetis poll thread started ✓")

        # ── snapshot → buffer timer ───────────────────────────────────────────
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_hz),
            self.timer_callback
        )

        rospy.loginfo(
            f"[ObsNode] Ready\n"
            f"  n_obs_steps : {self.n_obs_steps}\n"
            f"  publish_hz  : {self.publish_hz}\n"
            f"  poll_hz     : {self.poll_hz}\n"
            f"  robot_ip    : {self.robot_ip}\n"
            f"  urdf        : {self.urdf_path}\n"
            f"  ee_link     : {self.ee_link}\n"
            f"  cam1 (wrist)   : {self.cam1_topic}\n"
            f"  cam2 (external): {self.cam2_topic}"
        )

    # ── Polymetis polling loop ─────────────────────────────────────────────────

    # def _polymetis_poll_loop(self):
    #     """
    #     Polls Polymetis at poll_hz for joint positions and gripper width.
    #     gripper.get_state().width → total width → split /2 per finger
    #     """
    #     rate = 1.0 / self.poll_hz
    #     while not rospy.is_shutdown():
    #         try:
    #             joints = self.poly_robot.get_joint_positions().numpy()  # (7,)

    #             gripper_state = self.poly_gripper.get_state()
    #             per_finger    = float(gripper_state.width) / 2.0
    #             gripper       = np.array([per_finger, per_finger], dtype=np.float64)

    #             with self.snap_lock:
    #                 self.latest_joints  = joints
    #                 self.latest_gripper = gripper

    #         except Exception as e:
    #             rospy.logerr_throttle(2.0, f"[ObsNode] Polymetis poll error: {e}")

    #         rospy.sleep(rate)

    def _polymetis_poll_loop(self):
        """
        Polls Polymetis at poll_hz for joint positions and gripper width.
        If the gripper server is missing (common in sim), mocks the gripper open state.
        """
        rate = 1.0 / self.poll_hz
        while not rospy.is_shutdown():
            
            # 1. Try to get arm joints
            try:
                joints = self.poly_robot.get_joint_positions().numpy()  # (7,)
            except Exception as e:
                rospy.logerr_throttle(2.0, f"[ObsNode] ARM poll error: {e}")
                rospy.sleep(rate)
                continue  # Skip updating if we don't have arm joints

            # 2. Try to get gripper state
            try:
                gripper_state = self.poly_gripper.get_state()
                per_finger    = float(gripper_state.width) / 2.0
                gripper       = np.array([per_finger, per_finger], dtype=np.float64)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[ObsNode] GRIPPER poll error. Mocking open gripper. (Ignore if in sim)")
                # Mock gripper as fully open for simulation
                mock_finger = ROBOT_FINGER_OPEN / 2.0
                gripper     = np.array([mock_finger, mock_finger], dtype=np.float64)

            # 3. Update buffers
            with self.snap_lock:
                self.latest_joints  = joints
                self.latest_gripper = gripper

            rospy.sleep(rate)

    # ── camera callbacks — force bgr8 to match data collector ─────────────────

    def cam1_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")   # force BGR like data collector
            if bgr is not None and bgr.size > 0:
                with self.snap_lock:
                    self.latest_image_1 = np.array(bgr, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(f"[ObsNode] cam1 error: {e}")

    def cam2_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")   # force BGR like data collector
            if bgr is not None and bgr.size > 0:
                with self.snap_lock:
                    self.latest_image_2 = np.array(bgr, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(f"[ObsNode] cam2 error: {e}")

    # ── core loop ─────────────────────────────────────────────────────────────

    def timer_callback(self, event=None):
        with self.snap_lock:
            img1    = self.latest_image_1
            img2    = self.latest_image_2
            joints  = self.latest_joints
            gripper = self.latest_gripper

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

        # ── preprocess — matches converter exactly ─────────────────────────
        proc_img1                    = preprocess_image(img1)   # BGR input

        # debug_img = (proc_img1.transpose(1, 2, 0) * 255).astype(np.uint8)
        # cv2.imshow("Debug Cam1 (Model Input)", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        proc_img2                    = preprocess_image(img2)   # BGR input

        # debug_img = (proc_img2.transpose(1, 2, 0) * 255).astype(np.uint8)
        # cv2.imshow("Debug Cam1 (Model Input)", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        norm_pos, norm_euler, quat   = joints_to_eef(self.robot_model, joints)
        norm_gripper                 = normalize_gripper(gripper.astype(np.float32))

        # ── append to rolling buffers ──────────────────────────────────────
        with self.buf_lock:
            self.buf_cam1.append(proc_img1)
            self.buf_cam2.append(proc_img2)
            self.buf_eef_pos.append(norm_pos)
            self.buf_eef_euler.append(norm_euler)
            self.buf_eef_quat.append(quat)
            self.buf_gripper.append(norm_gripper)
            buf_len = len(self.buf_cam1)

        if buf_len < self.n_obs_steps:
            rospy.logwarn_throttle(
                2.0, f"[ObsNode] Buffer filling: {buf_len}/{self.n_obs_steps}"
            )
            return

        self._publish_obs()

    # ── publisher ─────────────────────────────────────────────────────────────

    def _publish_obs(self):
        """
        Flat layout (must match eval_real.flat_msg_to_obs_dict exactly):
            robot0_eef_pos      T * 3
            robot0_eef_euler    T * 3   (normalised, replaces quat)
            robot0_eef_quat     T * 4   (raw, for reference)
            robot0_eef_gpos     T * 2
            camera_image        T * 3 * DESIRED_H * DESIRED_W
            camera_wrist_image  T * 3 * DESIRED_H * DESIRED_W
        """
        with self.buf_lock:
            cam1      = np.stack(list(self.buf_cam1),      axis=0)  # (T,3,H,W)
            cam2      = np.stack(list(self.buf_cam2),      axis=0)  # (T,3,H,W)
            eef_pos   = np.stack(list(self.buf_eef_pos),   axis=0)  # (T,3)
            eef_euler = np.stack(list(self.buf_eef_euler), axis=0)  # (T,3)
            eef_quat  = np.stack(list(self.buf_eef_quat),  axis=0)  # (T,4)
            gripper   = np.stack(list(self.buf_gripper),   axis=0)  # (T,2)

        data = np.concatenate([
            eef_pos.flatten(),
            eef_euler.flatten(),
            eef_quat.flatten(),
            gripper.flatten(),
            cam1.flatten(),
            cam2.flatten(),
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
