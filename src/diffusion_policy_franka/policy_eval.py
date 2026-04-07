#!/usr/bin/env python3
"""
eval_real.py
------------
Runs the diffusion policy on a real or simulated Franka robot via ROS.

Reads all configuration from ROS params (set by launch file) so it
works with both diffusion_policy_sim.launch and diffusion_policy_real.launch.

ROS params (all set by launch file):
    ~checkpoint_path      path to .ckpt file
    ~device               torch device  (default: cuda:0)
    ~num_inference_steps  DDIM steps    (default: 10)
    ~n_obs_steps          obs window    (default: 2)
    ~output_dir           log output    (default: /tmp/eval_real_output)

Requires:
    - observation_node.py running  → /diffusion_policy/observation
    - controller_node.py running   → /diffusion_policy/action_chunk
    - ROS master running
"""

import sys
import os
import pathlib
import threading
import time
from copy import deepcopy

import dill
import hydra
import numpy as np
import torch
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from diffusion_policy.workspace.base_workspace import BaseWorkspace


# ── Obs unpacking ─────────────────────────────────────────────────────────────

def flat_msg_to_obs_dict(msg, n_obs_steps: int, device: str) -> dict:
    """
    Flat layout (must match observation_node._publish_obs exactly):
        robot0_eef_pos      n_obs_steps * 3
        robot0_eef_quat     n_obs_steps * 4
        robot0_eef_gpos     n_obs_steps * 2
        camera_image        n_obs_steps * 3 * 84 * 84
        camera_wrist_image  n_obs_steps * 3 * 84 * 84
    """
    data   = np.array(msg.data, dtype=np.float32)
    T      = n_obs_steps
    offset = 0

    def pull(size):
        nonlocal offset
        chunk = data[offset: offset + size]
        offset += size
        return chunk

    eef_pos   = pull(T * 3).reshape(T, 3)
    eef_quat  = pull(T * 4).reshape(T, 4)
    gripper   = pull(T * 2).reshape(T, 2)
    cam       = pull(T * 3 * 84 * 84).reshape(T, 3, 84, 84)
    cam_wrist = pull(T * 3 * 84 * 84).reshape(T, 3, 84, 84)

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32,
                            device=device).unsqueeze(0)  # (1, T, ...)

    return {
        "robot0_eef_pos":     to_tensor(eef_pos),
        "robot0_eef_quat":    to_tensor(eef_quat),
        "robot0_eef_gpos":    to_tensor(gripper),
        "camera_image":       to_tensor(cam),
        "camera_wrist_image": to_tensor(cam_wrist),
    }


# ── Policy runner ─────────────────────────────────────────────────────────────

class RealRobotRunner:

    def __init__(self, policy, device: str, n_obs_steps: int,
                 num_inference_steps: int, output_dir: str):
        self.policy              = policy
        self.device              = device
        self.n_obs_steps         = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.output_dir          = output_dir

        self.latest_obs       = None
        self.obs_lock         = threading.Lock()
        self.inference_thread = None
        self.inference_lock   = threading.Lock()

        self.inference_times  = []
        self.chunks_published = 0

        self.action_pub = rospy.Publisher(
            "/diffusion_policy/action_chunk",
            Float64MultiArray,
            queue_size=1
        )

        rospy.Subscriber(
            "/diffusion_policy/observation",
            Float64MultiArray,
            self._obs_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo(
            f"[EvalReal] Ready\n"
            f"  device              : {device}\n"
            f"  n_obs_steps         : {n_obs_steps}\n"
            f"  num_inference_steps : {num_inference_steps}\n"
        )

    def _obs_callback(self, msg: Float64MultiArray):
        with self.obs_lock:
            self.latest_obs = msg
        self._trigger_inference()

    def _trigger_inference(self):
        with self.inference_lock:
            if self.inference_thread is not None and self.inference_thread.is_alive():
                return

            with self.obs_lock:
                if self.latest_obs is None:
                    return
                obs_snapshot = deepcopy(self.latest_obs)

            self.inference_thread = threading.Thread(
                target=self._run_inference,
                args=(obs_snapshot,),
                daemon=True
            )
            self.inference_thread.start()

    def _run_inference(self, obs_msg):
        t0 = time.time()
        try:
            obs_dict = flat_msg_to_obs_dict(
                obs_msg, self.n_obs_steps, self.device)

            with torch.no_grad():
                result = self.policy.predict_action(obs_dict)

            actions = result['action'][0].cpu().numpy()  # (n_action_steps, 7)
            self._publish_action_chunk(actions)

            elapsed = (time.time() - t0) * 1000
            self.inference_times.append(elapsed)
            self.chunks_published += 1

            rospy.loginfo(
                f"[EvalReal] chunk {self.chunks_published:4d} | "
                f"inference {elapsed:6.1f} ms | "
                f"actions {actions.shape}"
            )

        except Exception as e:
            rospy.logerr(f"[EvalReal] Inference error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def _publish_action_chunk(self, actions: np.ndarray):
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="n_steps",
                                size=actions.shape[0],
                                stride=actions.shape[0] * actions.shape[1]),
            MultiArrayDimension(label="action_dim",
                                size=actions.shape[1],
                                stride=actions.shape[1]),
        ]
        msg.data = actions.flatten().tolist()
        self.action_pub.publish(msg)

    def run(self):
        rospy.loginfo("[EvalReal] Running — press Ctrl+C to stop.")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass

        inf_times = np.array(self.inference_times) if self.inference_times else np.array([0.0])
        runner_log = {
            "chunks_published":       self.chunks_published,
            "inference_time_mean_ms": float(inf_times.mean()),
            "inference_time_max_ms":  float(inf_times.max()),
            "inference_time_min_ms":  float(inf_times.min()),
            "inference_time_std_ms":  float(inf_times.std()),
        }

        rospy.loginfo(
            f"\n[EvalReal] Done.\n"
            f"  Chunks published : {self.chunks_published}\n"
            f"  Inference mean   : {inf_times.mean():.1f} ms\n"
            f"  Inference max    : {inf_times.max():.1f} ms\n"
        )
        return runner_log


# ── Main — reads everything from ROS params ───────────────────────────────────

def main():
    # init node first so we can read params
    rospy.init_node("eval_real", anonymous=False)

    # ── read params from launch file ──────────────────────────────────────────
    checkpoint_path     = rospy.get_param("~checkpoint_path")
    device              = rospy.get_param("~device",              "cuda:0")
    num_inference_steps = rospy.get_param("~num_inference_steps", 10)
    n_obs_steps         = rospy.get_param("~n_obs_steps",         2)
    output_dir          = rospy.get_param("~output_dir",          "/tmp/eval_real_output")

    rospy.loginfo(
        f"[EvalReal] Config\n"
        f"  checkpoint      : {checkpoint_path}\n"
        f"  device          : {device}\n"
        f"  n_obs_steps     : {n_obs_steps}\n"
        f"  inference_steps : {num_inference_steps}\n"
        f"  output_dir      : {output_dir}\n"
    )

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── load checkpoint ───────────────────────────────────────────────────────
    rospy.loginfo("[EvalReal] Loading checkpoint ...")
    payload   = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg       = payload['cfg']
    cls       = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()

    # override inference steps (DDPM → DDIM for speed)
    policy.num_inference_steps = num_inference_steps

    # use n_obs_steps from param — must match training config
    # (param takes priority; fall back to cfg if not set differently)
    rospy.loginfo("[EvalReal] Checkpoint loaded ✓")

    # ── run ───────────────────────────────────────────────────────────────────
    runner = RealRobotRunner(
        policy=policy,
        device=device,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        output_dir=output_dir,
    )
    runner_log = runner.run()

    # ── save log ──────────────────────────────────────────────────────────────
    import json
    out_path = os.path.join(output_dir, 'eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(runner_log, f, indent=2, sort_keys=True)
    rospy.loginfo(f"[EvalReal] Log saved to {out_path}")


if __name__ == '__main__':
    main()