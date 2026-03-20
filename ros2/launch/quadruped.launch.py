# =============================================================================
# ros2/launch/quadruped.launch.py
#
# ROS2 launch file — starts the sim bridge and policy node together.
#
# Usage (with ROS2 sourced):
#   ros2 launch ros2/launch/quadruped.launch.py model:=experiments/.../final_model.zip
# =============================================================================

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        "model",
        description="Path to trained policy .zip file"
    )
    config_arg = DeclareLaunchArgument(
        "config",
        default_value="training/configs/quadruped_terrain.yaml",
        description="Path to YAML config"
    )

    sim_node = Node(
        package="robot_lab",
        executable="mujoco_ros2_node.py",
        name="mujoco_sim",
        output="screen"
    )

    policy_node = Node(
        package="robot_lab",
        executable="ppo_policy_node.py",
        name="ppo_policy",
        output="screen",
        parameters=[{
            "model":  LaunchConfiguration("model"),
            "config": LaunchConfiguration("config"),
        }]
    )

    return LaunchDescription([model_arg, config_arg, sim_node, policy_node])
