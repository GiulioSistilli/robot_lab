# arm robot

Placeholder for future robotic arm model.

To add an arm:
1. Place model.xml here
2. Place mesh files in meshes/
3. Create envs/arm_reach_env.py inheriting BaseRobotEnv
4. Create training/configs/arm_reach.yaml
5. Run: python training/train.py --config training/configs/arm_reach.yaml
