# =============================================================================
# envs/quadruped_terrain_env.py
#
# Quadruped on procedurally randomised terrain with curriculum learning.
# Terrain is regenerated every episode: flat / bumpy / ramp.
# Difficulty increases gradually based on episode count.
# =============================================================================

import gymnasium as gym
import numpy as np
import mujoco
from envs.base_env import BaseRobotEnv


class QuadrupedTerrainEnv(BaseRobotEnv):
    """Quadruped on randomised heightfield terrain with curriculum."""

    def __init__(self, xml_path="robots/quadruped/terrain.xml", render_mode=None):
        super().__init__(xml_path=xml_path, render_mode=render_mode)

        self._episode_count  = 0
        self._hfield_nrow    = self.model.hfield_nrow[0]   # 64
        self._hfield_ncol    = self.model.hfield_ncol[0]   # 64

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

    def reset(self, seed=None, options=None):
        # Generate new terrain BEFORE resetting physics
        gym.Env.reset(self, seed=seed)
        self._randomise_terrain()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.6  # spawn above terrain
        n_joints = self.model.nq - 7
        if n_joints > 0:
            self.data.qpos[7:] += self.np_random.uniform(-0.1, 0.1, size=n_joints)
        mujoco.mj_forward(self.model, self.data)
        self._episode_count += 1
        return self._get_obs(), {}

    def _randomise_terrain(self):
        """Fill heightfield with flat, bumpy, or ramp terrain."""
        weights = self._get_terrain_weights()
        terrain_type = self.np_random.choice(["flat", "bumpy", "ramp"], p=weights)
        nrow, ncol   = self._hfield_nrow, self._hfield_ncol

        if terrain_type == "flat":
            heights = np.full((nrow, ncol), 0.5)

        elif terrain_type == "bumpy":
            x  = np.linspace(0, 2 * np.pi, ncol)
            y  = np.linspace(0, 2 * np.pi, nrow)
            xx, yy = np.meshgrid(x, y)
            heights = np.zeros((nrow, ncol))
            for freq in [1, 2, 3, 4]:
                px = self.np_random.uniform(0, 2 * np.pi)
                py = self.np_random.uniform(0, 2 * np.pi)
                heights += (0.15 / freq) * np.sin(freq * xx + px) * np.cos(freq * yy + py)
            heights = (heights - heights.min()) / (heights.max() - heights.min())
            heights = 0.3 + heights * 0.4

        elif terrain_type == "ramp":
            slope   = np.linspace(0.5, 0.8, ncol)
            heights = np.tile(slope, (nrow, 1))
            heights = np.clip(heights + self.np_random.uniform(-0.02, 0.02, (nrow, ncol)), 0, 1)

        # Flat spawn zone in the centre
        cx, cy = nrow // 2, ncol // 2
        heights[cx-4:cx+4, cy-4:cy+4] = 0.5

        self.model.hfield_data[:] = heights.flatten()

    def _get_terrain_weights(self):
        """Curriculum: more flat early, more bumpy/ramp later."""
        ep = self._episode_count
        if   ep < 500:  return [0.90, 0.08, 0.02]
        elif ep < 1500: return [0.60, 0.35, 0.05]
        elif ep < 3000: return [0.30, 0.55, 0.15]
        else:           return [0.20, 0.60, 0.20]

    def _get_obs(self):
        foot_contacts = np.clip(self.data.sensordata[:4] / 100.0, 0.0, 1.0)
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            foot_contacts
        ]).astype("float32")

    def _compute_reward(self):
        forward_vel  = float(self.data.qvel[0])
        lateral_vel  = float(abs(self.data.qvel[1]))
        upright      = float(self.data.qpos[3] ** 2)
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
