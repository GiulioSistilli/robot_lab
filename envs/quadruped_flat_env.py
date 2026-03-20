# =============================================================================
# envs/quadruped_flat_env.py
#
# Quadruped walking on flat ground.
# Observation: qpos(15) + qvel(14) + foot contacts(4) = 33 values
# Action     : 8 joint targets in [-1, 1]
# Reward     : forward velocity + uprightness + foot contact + survival
# Termination: torso height < 0.15m
# =============================================================================

import gymnasium as gym
import numpy as np
from envs.base_env import BaseRobotEnv


class QuadrupedFlatEnv(BaseRobotEnv):
    """Quadruped on flat terrain with foot contact sensors."""

    def __init__(self, xml_path="robots/quadruped/model.xml", render_mode=None):
        super().__init__(xml_path=xml_path, render_mode=render_mode)

        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,), dtype="float32"
        )
        self._observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"),
            shape=(self.model.nq + self.model.nv + 4,), dtype="float32"
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _get_obs(self):
        # Normalise foot contact forces to [0, 1]
        foot_contacts = np.clip(self.data.sensordata[:4] / 100.0, 0.0, 1.0)
        return np.concatenate([
            self.data.qpos.flat,   # 15 values: torso pose + 8 joint angles
            self.data.qvel.flat,   # 14 values: torso velocity + 8 joint vels
            foot_contacts          #  4 values: one per foot
        ]).astype("float32")

    def _compute_reward(self):
        forward_vel  = float(self.data.qvel[0])
        lateral_vel  = float(abs(self.data.qvel[1]))
        upright      = float(self.data.qpos[3] ** 2)   # quaternion w^2
        ctrl_penalty = float(0.001 * np.sum(self.data.ctrl ** 2))
        feet_contact = float(np.sum(self.data.sensordata[:4] > 0.5))
        return (
            + 2.0 * forward_vel
            - 0.5 * lateral_vel
            + 1.0 * upright
            - ctrl_penalty
            + 0.5
            + 0.2 * feet_contact
        )

    def _is_terminated(self):
        return bool(self.data.qpos[2] < 0.15)
