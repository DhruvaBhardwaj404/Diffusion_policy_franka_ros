#!/usr/bin/env python3
"""
eval_real.py
------------
Drop-in replacement for eval.py that runs the policy on a real Franka robot
via ROS instead of a simulation environment.

Everything above "run eval" is identical to eval.py.
The env_runner block is replaced by a ROS pub/sub loop.

Usage:
    python eval_real.py \
        --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
        --output_dir data/real_eval_output

Requires:
    - observation_node.py running  (publishes /diffusion_policy/observation)
    - controller_node.py running   (subscribes /diffusion_policy/action_chunk)
    - ROS master running
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import threading
import time
from copy import deepcopy

import click
import dill
import hydra
import numpy as np
import torch

# ROS imports
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import sys
sys.path.insert(0, '/home/dhruv/Diffusion-Transformer')

from diffusion_policy.workspace.base_workspace import BaseWorkspace


# ── Obs unpacking — identical to policy_node.py ───────────────────────────────

def flat_msg_to_obs_dict(msg, n_obs_steps: int, device: str) -> dict:
    """
    Flat layout (must match observation_node.publish_obs exactly):
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


# ── Real robot eval loop ──────────────────────────────────────────────────────

class RealRobotRunner:
    """
    Replaces env_runner.run(policy) for real robot execution.

    Subscribes to observation_node, runs policy inference,
    publishes action chunks to controller_node.
    Mirrors the async inference pattern from policy_node.py.
    """

    def __init__(self, policy, device: str, n_obs_steps: int,
                 num_inference_steps: int, output_dir: str):
        self.policy              = policy
        self.device              = device
        self.n_obs_steps         = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.output_dir          = output_dir

        # ── state ─────────────────────────────────────────────────────────────
        self.latest_obs       = None
        self.obs_lock         = threading.Lock()
        self.inference_thread = None
        self.inference_lock   = threading.Lock()

        # metrics
        self.inference_times  = []
        self.chunks_published = 0

        # ── ROS ───────────────────────────────────────────────────────────────
        rospy.init_node("eval_real", anonymous=False)

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

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _obs_callback(self, msg: Float64MultiArray):
        """Store latest obs and trigger async inference — same as policy_node."""
        with self.obs_lock:
            self.latest_obs = msg
        self._trigger_inference()

    def _trigger_inference(self):
        with self.inference_lock:
            if self.inference_thread is not None and self.inference_thread.is_alive():
                return  # previous inference still running

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

            # (1, n_action_steps, 7) → (n_action_steps, 7)
            actions = result['action'][0].cpu().numpy()

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
            import traceback; rospy.logerr(traceback.format_exc())

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

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self):
        """
        Spin until Ctrl+C, then save a summary log — mirrors runner_log
        structure from eval.py so the rest of eval_real.py stays the same.
        """
        rospy.loginfo("[EvalReal] Running — press Ctrl+C to stop.")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass

        # ── summary log (replaces runner_log from env_runner.run()) ───────────
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


# ── CLI — identical to eval.py above the run block ───────────────────────────

@click.command()
@click.option('-c', '--checkpoint',  required=True)
@click.option('-o', '--output_dir',  required=True)
@click.option('-d', '--device',      default='cuda:0')
def main(checkpoint, output_dir, device, num_inference_steps):

    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── load checkpoint — identical to eval.py ────────────────────────────────
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg     = payload['cfg']
    cls     = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy — identical to eval.py
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()

    # read n_obs_steps from the saved config so it always matches training
    n_obs_steps = cfg.policy.n_obs_steps

    # ── run eval — this replaces env_runner.run(policy) ──────────────────────
    runner = RealRobotRunner(
        policy=policy,
        device=device,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        output_dir=output_dir,
    )
    runner_log = runner.run()

    # ── save log — identical to eval.py ──────────────────────────────────────
    import json
    out_path = os.path.join(output_dir, 'eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(runner_log, f, indent=2, sort_keys=True)

    print(f"Log saved to {out_path}")


if __name__ == '__main__':
    main()
