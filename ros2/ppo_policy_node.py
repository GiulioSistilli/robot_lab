# =============================================================================
# ros2/ppo_policy_node.py
#
# Runs a trained PPO policy as a ROS2 node.
# Subscribes to robot state topics, publishes joint commands.
#
# Usage:
#   python ros2/ppo_policy_node.py \
#     --model experiments/quadruped_terrain_.../final_model.zip \
#     --config training/configs/quadruped_terrain.yaml
# =============================================================================

import argparse
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from stable_baselines3 import PPO, SAC
import numpy as np


class PPOPolicyNode(Node):
    def __init__(self, model_path, algorithm):
        super().__init__("ppo_policy")
        AlgoClass  = {"PPO": PPO, "SAC": SAC}[algorithm]
        self.model = AlgoClass.load(model_path)
        self.get_logger().info(f"Policy loaded from: {model_path}")

        self._qpos      = np.zeros(15)
        self._qvel      = np.zeros(14)
        self._foot      = np.zeros(4)
        self._obs_ready = False

        self.cmd_pub = self.create_publisher(Float32MultiArray, "/joint_commands", 10)
        self.create_subscription(JointState,       "/joint_states",  self._on_joint_states,  10)
        self.create_subscription(Odometry,          "/odom",          self._on_odom,           10)
        self.create_subscription(Float32MultiArray, "/foot_contacts", self._on_foot_contacts,  10)
        self.create_timer(0.008, self._policy_step)

    def _on_joint_states(self, msg):
        self._qpos[7:] = np.array(msg.position)
        self._qvel[6:] = np.array(msg.velocity)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist
        self._qpos[0:3] = [p.x, p.y, p.z]
        self._qpos[3:7] = [o.w, o.x, o.y, o.z]
        self._qvel[0:3] = [v.linear.x,  v.linear.y,  v.linear.z]
        self._qvel[3:6] = [v.angular.x, v.angular.y, v.angular.z]
        self._obs_ready = True

    def _on_foot_contacts(self, msg):
        self._foot = np.clip(np.array(msg.data) / 100.0, 0.0, 1.0)

    def _policy_step(self):
        if not self._obs_ready:
            return
        obs = np.concatenate([self._qpos, self._qvel, self._foot]).astype("float32")
        action, _ = self.model.predict(obs, deterministic=True)
        msg = Float32MultiArray()
        msg.data = (action * 0.5).tolist()
        self.cmd_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    rclpy.init()
    node = PPOPolicyNode(args.model, config["algorithm"])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
