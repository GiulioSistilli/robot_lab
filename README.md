# robot_lab

A MuJoCo reinforcement learning project for training quadruped (and future) robots using PPO.
Built with MuJoCo 3.x, Stable-Baselines3, Gymnasium, and ROS2.

## Project structure

```
robot_lab/
├── robots/          # robot XML models and meshes (one folder per robot)
├── envs/            # Gymnasium environments (one file per robot+task)
├── training/        # train.py entry point + YAML configs
├── experiments/     # auto-created output folders (gitignored)
├── ros2/            # ROS2 bridge nodes and launch files
└── scripts/         # evaluate, watch training, tensorboard launcher
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/robot_lab.git
cd robot_lab
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

## Train a robot

```bash
# Quadruped on flat ground
python training/train.py --config training/configs/quadruped_flat.yaml

# Quadruped on randomised terrain (curriculum learning)
python training/train.py --config training/configs/quadruped_terrain.yaml

# Resume from a checkpoint
python training/train.py --config training/configs/quadruped_terrain.yaml \
  --resume experiments/quadruped_terrain_2025-03-20_14-32/checkpoints/quadruped_terrain_500000_steps.zip
```

## Watch training live (run alongside train.py)

```bash
# Terminal 1 - training
python training/train.py --config training/configs/quadruped_terrain.yaml

# Terminal 2 - live viewer
python scripts/watch_training.py --config training/configs/quadruped_terrain.yaml

# Terminal 3 - TensorBoard
python scripts/start_tensorboard.py
# then open http://localhost:6006
```

## Evaluate a trained model

```bash
python scripts/evaluate.py --model experiments/quadruped_terrain_2025-03-20_14-32/final_model.zip \
                            --config training/configs/quadruped_terrain.yaml
```

## ROS2 integration

```bash
# Terminal 1 - simulation bridge
python ros2/mujoco_ros2_node.py

# Terminal 2 - PPO policy node
python ros2/ppo_policy_node.py --model experiments/.../final_model.zip

# Terminal 3 - inspect topics
ros2 topic echo /joint_states
ros2 topic hz /joint_commands
```

## Adding a new robot

1. Add your XML model to `robots/your_robot/model.xml`
2. Create `envs/your_robot_task_env.py` inheriting from `BaseRobotEnv`
3. Write `training/configs/your_robot_task.yaml`
4. Run `python training/train.py --config training/configs/your_robot_task.yaml`

No other files need to be modified.

## Training results (this session)

| Run | ep_len_mean | ep_rew_mean | Notes |
|-----|-------------|-------------|-------|
| PPO_1 | 455 | 219 | v1 flat, no sensors |
| PPO_2 | 1,147 | 1,044 | v1 flat, no sensors |
| PPO_v2 | 3,038 | 3,471 | foot sensors, still climbing |
| PPO_terrain | 119,571 | 53,899 | terrain curriculum — explosive |

## Requirements

- Python 3.12
- MuJoCo 3.6.0
- stable-baselines3 2.7.1
- gymnasium 1.2.3
- PyTorch 2.10.0
- ROS2 Humble (optional, for ros2/ nodes)
