# envs/__init__.py
# Makes envs/ a Python package so train.py can import from it dynamically.
from envs.base_env import BaseRobotEnv
from envs.quadruped_flat_env import QuadrupedFlatEnv
from envs.quadruped_terrain_env import QuadrupedTerrainEnv
