# quadruped robot

A 4-legged robot with:
- 1 torso (freejoint: 6 DOF)
- 4 legs, each with hip + knee hinge joint = 8 joints total
- 8 position actuators (kp=50)
- 4 touch sensors at foot sites

## Files
- model.xml   : flat ground version
- terrain.xml : heightfield terrain version
- meshes/     : place STL/OBJ files from FreeCAD here
- urdf/       : URDF export for ROS2 / MoveIt2

## Observation space (33 values)
- qpos[0:3]  : torso x, y, z
- qpos[3:7]  : torso quaternion (w, x, y, z)
- qpos[7:15] : 8 joint angles (rad)
- qvel[0:6]  : torso linear + angular velocity
- qvel[6:14] : 8 joint velocities (rad/s)
- sensors    : 4 foot contact forces normalised to [0, 1]

## Action space (8 values in [-1, 1])
Scaled by ctrl_scale=0.5 to joint angle targets in radians.
Order: fl_hip, fl_knee, fr_hip, fr_knee, bl_hip, bl_knee, br_hip, br_knee
