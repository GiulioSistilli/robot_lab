# =============================================================================
# scripts/evaluate.py
#
# Watch a trained policy run in the MuJoCo viewer.
# Works for any robot — just pass the model and config.
#
# Usage:
#   python scripts/evaluate.py \
#     --model experiments/quadruped_terrain_2025-03-20_14-32/final_model.zip \
#     --config training/configs/quadruped_terrain.yaml
# =============================================================================

import argparse
import yaml
import sys
import os
import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train import load_env_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to .zip model file")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    EnvClass   = load_env_class(config["env_class"])
    env_kwargs = config.get("env_kwargs", {})
    env = EnvClass(**env_kwargs)

    AlgoClass = {"PPO": PPO, "SAC": SAC}[config["algorithm"]]
    model     = AlgoClass.load(args.model)

    print(f"Model  : {args.model}")
    print(f"Robot  : {config['robot']} / {config['task']}")
    print("Close the viewer window to stop.\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            obs, _     = env.reset()
            terminated = False
            total_rew  = 0.0
            steps      = 0

            while not terminated and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, _, _ = env.step(action)
                total_rew += reward
                steps     += 1
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.physics_steps_per_action)

            episode += 1
            print(f"Episode {episode:3d} | steps={steps:6d} | reward={total_rew:8.1f}")

    env.close()


if __name__ == "__main__":
    main()
