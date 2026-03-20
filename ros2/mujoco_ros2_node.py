# =============================================================================
# ros2/mujoco_ros2_node.py
#
# ROS2 node that runs the MuJoCo simulation and exposes it as standard topics.
#
# Publishes:
#   /joint_states   sensor_msgs/JointState
#   /odom           nav_msgs/Odometry
#   /foot_contacts  std_msgs/Float32MultiArray
#
# Subscribes:
#   /joint_commands std_msgs/Float32MultiArray
#
# Usage (requires ROS2 Humble + rclpy):
#   python ros2/mujoco_ros2_node.py
# =============================================================================

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TransformStamped
import tf2_ros
import mujoco
import numpy as np

JOINT_NAMES = [
    "fl_hip", "fl_knee", "fr_hip", "fr_knee",
    "bl_hip", "bl_knee", "br_hip", "br_knee",
]


class MuJoCoROS2Node(Node):
    def __init__(self):
        super().__init__("mujoco_quadruped")
        self.model = mujoco.MjModel.from_xml_path("robots/quadruped/model.xml")
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        self.joint_pub   = self.create_publisher(JointState,          "/joint_states",  10)
        self.odom_pub    = self.create_publisher(Odometry,             "/odom",          10)
        self.contact_pub = self.create_publisher(Float32MultiArray,    "/foot_contacts", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.cmd_sub = self.create_subscription(
            Float32MultiArray, "/joint_commands", self._on_joint_command, 10
        )

        self.create_timer(self.model.opt.timestep, self._sim_step)
        self.get_logger().info("MuJoCo ROS2 node started.")

    def _sim_step(self):
        mujoco.mj_step(self.model, self.data)
        now = self.get_clock().now().to_msg()
        self._publish_joint_states(now)
        self._publish_odometry(now)
        self._publish_foot_contacts(now)
        self._publish_tf(now)

    def _on_joint_command(self, msg):
        if len(msg.data) == self.model.nu:
            self.data.ctrl[:] = np.array(msg.data, dtype=np.float64)

    def _publish_joint_states(self, stamp):
        msg = JointState()
        msg.header.stamp    = stamp
        msg.header.frame_id = "base_link"
        msg.name     = JOINT_NAMES
        msg.position = self.data.qpos[7:].tolist()
        msg.velocity = self.data.qvel[6:].tolist()
        msg.effort   = self.data.actuator_force[:].tolist()
        self.joint_pub.publish(msg)

    def _publish_odometry(self, stamp):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id  = "base_link"
        msg.pose.pose.position.x = float(self.data.qpos[0])
        msg.pose.pose.position.y = float(self.data.qpos[1])
        msg.pose.pose.position.z = float(self.data.qpos[2])
        w, x, y, z = self.data.qpos[3:7]
        msg.pose.pose.orientation.x = float(x)
        msg.pose.pose.orientation.y = float(y)
        msg.pose.pose.orientation.z = float(z)
        msg.pose.pose.orientation.w = float(w)
        msg.twist.twist.linear.x  = float(self.data.qvel[0])
        msg.twist.twist.linear.y  = float(self.data.qvel[1])
        msg.twist.twist.linear.z  = float(self.data.qvel[2])
        msg.twist.twist.angular.x = float(self.data.qvel[3])
        msg.twist.twist.angular.y = float(self.data.qvel[4])
        msg.twist.twist.angular.z = float(self.data.qvel[5])
        self.odom_pub.publish(msg)

    def _publish_foot_contacts(self, stamp):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in self.data.sensordata[:4]]
        self.contact_pub.publish(msg)

    def _publish_tf(self, stamp):
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = "odom"
        t.child_frame_id  = "base_link"
        t.transform.translation.x = float(self.data.qpos[0])
        t.transform.translation.y = float(self.data.qpos[1])
        t.transform.translation.z = float(self.data.qpos[2])
        w, x, y, z = self.data.qpos[3:7]
        t.transform.rotation.x = float(x)
        t.transform.rotation.y = float(y)
        t.transform.rotation.z = float(z)
        t.transform.rotation.w = float(w)
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = MuJoCoROS2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
