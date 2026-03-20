# =============================================================================
# training/train.py
#
# Single unified training entry point for ALL robots and tasks.
# Controlled entirely by YAML config — never edit this file directly.
#
# Usage:
#   python training/train.py --config training/configs/quadruped_terrain.yaml
#   python training/train.py --config training/configs/quadruped_flat.yaml
#   python training/train.py --config training/configs/quadruped_terrain.yaml \
#       --resume experiments/quadruped_terrain_2025-03-20_14-32/checkpoints/model_500000_steps.zip
# =============================================================================

import argparse
import yaml
import os
import re
import importlib
import shutil
from datetime import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_env_class(env_name):
    """Dynamically import env class from envs/ by CamelCase name."""
    module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", env_name).lower()
    module = importlib.import_module(f"envs.{module_name}")
    return getattr(module, env_name)


def make_experiment_dir(config):
    """Create timestamped output directory: experiments/{robot}_{task}_{timestamp}/"""
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = f"{config['robot']}_{config['task']}_{ts}"
    path = os.path.join("experiments", name)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(path, "tb_logs"),     exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="YAML config path")
    parser.add_argument("--resume",  default=None,  help="Checkpoint .zip to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"\nRobot     : {config['robot']}")
    print(f"Task      : {config['task']}")
    print(f"Algorithm : {config['algorithm']}")
    print(f"Envs      : {config['n_envs']} parallel")
    print(f"Timesteps : {config['total_timesteps']:,}\n")

    exp_dir = make_experiment_dir(config)
    shutil.copy(args.config, os.path.join(exp_dir, "config.yaml"))
    print(f"Output    : {exp_dir}\n")

    EnvClass   = load_env_class(config["env_class"])
    env_kwargs = config.get("env_kwargs", {})
    env = make_vec_env(EnvClass, n_envs=config["n_envs"], env_kwargs=env_kwargs)

    AlgoClass   = {"PPO": PPO, "SAC": SAC}[config["algorithm"]]
    algo_kwargs = config.get("algo_kwargs", {})

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = AlgoClass.load(args.resume, env=env)
        reset_num_timesteps = False
    else:
        model = AlgoClass(
            "MlpPolicy", env,
            verbose=1,
            tensorboard_log=os.path.join(exp_dir, "tb_logs"),
            **algo_kwargs
        )
        reset_num_timesteps = True

    checkpoint = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 50_000),
        save_path=os.path.join(exp_dir, "checkpoints"),
        name_prefix=f"{config['robot']}_{config['task']}"
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=checkpoint,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=f"{config['robot']}_{config['task']}"
    )

    final = os.path.join(exp_dir, "final_model")
    model.save(final)
    print(f"\nDone. Model: {final}.zip")
    print(f"Evaluate : python scripts/evaluate.py --model {final}.zip --config {args.config}")


if __name__ == "__main__":
    main()
