#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import tf_transformations as tf


class IMUGravityAligner(Node):
    def __init__(self):
        super().__init__('imu_gravity_aligner')
        
        # Declare and get parameters
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('parent_frame', 'world')
        self.declare_parameter('child_frame', 'world_aligned')
        self.declare_parameter('smoothing_factor', 0.1)  # EMA smoothing factor
        
        self.imu_topic = self.get_parameter('imu_topic').value
        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        self.alpha = self.get_parameter('smoothing_factor').value  # EMA factor
        
        self.count = 0
        self.smoothed_quat = np.array([0, 0, 0, 1])  # Initial quaternion (identity rotation)
        
        self.subscription = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10)
        self.tf_broadcaster = TransformBroadcaster(self)        

    def imu_callback(self, msg: Imu):
        self.count += 1
        if self.count % 100 == 0:
            self.get_logger().info("msgs received: %i" % self.count)        

        # Extract acceleration (gravity vector)
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        
        # Normalize acceleration vector (to avoid scaling issues)
        if np.linalg.norm(acc) < 1e-6:
            self.get_logger().warn("Received nearly zero acceleration, skipping transformation.")
            return
        
        gravity_vector = acc / np.linalg.norm(acc)

        # Compute correction to align Z-axis with gravity
        world_z = np.array([0, 0, 1])
        correction_axis = np.cross(gravity_vector, world_z)
        correction_angle = np.arccos(np.clip(np.dot(gravity_vector, world_z), -1.0, 1.0))  # Prevent NaN

        if np.linalg.norm(correction_axis) > 1e-6:
            correction_axis /= np.linalg.norm(correction_axis)
            correction_quat = tf.quaternion_about_axis(correction_angle, correction_axis)
        else:
            correction_quat = np.array([0, 0, 0, 1])  # No rotation needed

        # Apply exponential moving average for smoothing
        self.smoothed_quat = self.alpha * np.array(correction_quat) + (1 - self.alpha) * self.smoothed_quat
        self.smoothed_quat /= np.linalg.norm(self.smoothed_quat)  # Normalize quaternion

        # Publish the smoothed transformation
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.rotation.x = self.smoothed_quat[0]
        t.transform.rotation.y = self.smoothed_quat[1]
        t.transform.rotation.z = self.smoothed_quat[2]
        t.transform.rotation.w = self.smoothed_quat[3]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = IMUGravityAligner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
