# =============================================================================
# scripts/export_policy.py
#
# Exports a trained policy's network weights to a lightweight format
# suitable for deployment on Raspberry Pi.
#
# Exports:
#   - policy_weights.npz  : raw numpy weights (no SB3/PyTorch needed at runtime)
#   - policy_info.yaml    : observation/action space metadata
#
# Usage:
#   python scripts/export_policy.py \
#     --model experiments/.../final_model.zip \
#     --config training/configs/quadruped_terrain.yaml \
#     --output deploy/
#
# On the Raspberry Pi, use deploy/pi_inference.py to run the exported policy.
# =============================================================================

import argparse
import yaml
import os
import numpy as np
from stable_baselines3 import PPO, SAC


def export_policy(model_path, config_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    AlgoClass = {"PPO": PPO, "SAC": SAC}[config["algorithm"]]
    model     = AlgoClass.load(model_path)

    # Extract network weights as numpy arrays
    weights = {}
    for name, param in model.policy.named_parameters():
        weights[name.replace(".", "/")] = param.detach().cpu().numpy()

    weights_path = os.path.join(output_dir, "policy_weights.npz")
    np.savez(weights_path, **weights)
    print(f"Weights saved : {weights_path}")

    # Save metadata needed by the Pi inference script
    info = {
        "robot"      : config["robot"],
        "task"       : config["task"],
        "algorithm"  : config["algorithm"],
        "obs_size"   : int(model.observation_space.shape[0]),
        "action_size": int(model.action_space.shape[0]),
        "ctrl_scale" : 0.5,
        "model_path" : model_path,
    }
    info_path = os.path.join(output_dir, "policy_info.yaml")
    with open(info_path, "w") as f:
        yaml.dump(info, f)
    print(f"Info saved    : {info_path}")

    # Also copy the full .zip — SB3 is available on Pi if needed
    import shutil
    zip_path = os.path.join(output_dir, "final_model.zip")
    shutil.copy(model_path, zip_path)
    print(f"Model copied  : {zip_path}")
    print(f"\nDeploy folder ready: {output_dir}")
    print("Copy this folder to your Raspberry Pi and run: python pi_inference.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="deploy")
    args = parser.parse_args()
    export_policy(args.model, args.config, args.output)


if __name__ == "__main__":
    main()
