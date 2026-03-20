# =============================================================================
# scripts/watch_training.py
#
# Live viewer — run alongside training/train.py to watch the robot improve.
# Automatically finds and reloads the latest checkpoint every N episodes.
#
# Usage:
#   python scripts/watch_training.py --config training/configs/quadruped_terrain.yaml
# =============================================================================

import argparse
import yaml
import sys
import os
import glob
import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train import load_env_class

REFRESH_EVERY = 20  # reload latest checkpoint every N episodes


def get_latest_checkpoint(config):
    """Find the most recently saved checkpoint for this robot+task."""
    pattern = os.path.join(
        "experiments",
        f"{config['robot']}_{config['task']}_*",
        "checkpoints",
        "*.zip"
    )
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    EnvClass   = load_env_class(config["env_class"])
    env_kwargs = config.get("env_kwargs", {})
    env        = EnvClass(**env_kwargs)
    AlgoClass  = {"PPO": PPO, "SAC": SAC}[config["algorithm"]]

    viewer  = mujoco.viewer.launch_passive(env.model, env.data)
    model   = None
    episode = 0

    print(f"Watching: {config['robot']} / {config['task']}")
    print(f"Reloading checkpoint every {REFRESH_EVERY} episodes.\n")

    while viewer.is_running():
        if episode % REFRESH_EVERY == 0:
            ckpt = get_latest_checkpoint(config)
            if ckpt:
                print(f"\n[loading: {os.path.basename(ckpt)}]")
                model = AlgoClass.load(ckpt)
            else:
                print("[no checkpoint yet — random policy]")

        obs, _     = env.reset()
        terminated = False
        total_rew  = 0.0

        while not terminated and viewer.is_running():
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(action)
            total_rew += reward
            viewer.sync()
            time.sleep(0.008)

        episode += 1
        print(f"ep {episode:4d}  reward={total_rew:8.1f}", end="\r")

    env.close()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
