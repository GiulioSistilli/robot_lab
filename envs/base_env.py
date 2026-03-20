# =============================================================================
# envs/base_env.py
#
# Abstract base class all robot environments inherit from.
# Provides shared boilerplate so each child env only defines what is unique:
#   - observation_space / action_space (properties)
#   - _get_obs()         -> np.ndarray
#   - _compute_reward()  -> float
#   - _is_terminated()   -> bool
#
# To add a new robot:
#   1. Create envs/my_robot_task_env.py
#   2. class MyRobotTaskEnv(BaseRobotEnv): ...
#   3. Override the four abstract items above
#   4. Write a training/configs/my_robot_task.yaml
# =============================================================================

import gymnasium as gym
import numpy as np
import mujoco
from abc import ABC, abstractmethod


class BaseRobotEnv(gym.Env, ABC):
    """Abstract base for all MuJoCo robot environments."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str,
        physics_steps_per_action: int = 4,
        render_mode: str = None
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.physics_steps_per_action = physics_steps_per_action
        self.render_mode = render_mode
        self._viewer     = None
        self._ctrl_scale = 0.5  # maps [-1,1] actions to joint angle targets (rad)

    # --------------------------------------------------------------------------
    # Subclasses must implement these
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Box:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        pass

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def _compute_reward(self) -> float:
        pass

    @abstractmethod
    def _is_terminated(self) -> bool:
        pass

    # --------------------------------------------------------------------------
    # Shared implementation
    # --------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        n_joints = self.model.nq - 7
        if n_joints > 0:
            self.data.qpos[7:] += self.np_random.uniform(-0.1, 0.1, size=n_joints)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0) * self._ctrl_scale
        for _ in range(self.physics_steps_per_action):
            mujoco.mj_step(self.model, self.data)
        obs        = self._get_obs()
        reward     = self._compute_reward()
        terminated = self._is_terminated()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, {}

    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
