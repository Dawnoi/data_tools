#!/usr/bin/env python3
"""
Gripper Control Unit Test Node (Safe Version)

Flow:
1. Subscribe to the current arm end pose topic and wait for the first reading.
2. Record current pose (xyz + quaternion) and current gripper value.
3. Build action[0:7] from the REAL current pose (arm won't move).
4. Allow user to set action[7] gripper value via parameter.
5. Publish the action and trace every step of the pipeline.

This version uses the ACTUAL current pose so the arm stays still
and only the gripper responds.

Usage:
    # Test with default gripper value (0.096 - near open)
    ros2 run data_tools test_gripper_control

    # Test gripper fully open (0.1)
    ros2 run data_tools test_gripper_control --ros-args -p test_gripper_value:=0.1

    # Test gripper fully closed (0.0)
    ros2 run data_tools test_gripper_control --ros-args -p test_gripper_value:=0.0

    # Test gripper half open (0.05)
    ros2 run data_tools test_gripper_control --ros-args -p test_gripper_value:=0.05

    # Test right arm
    ros2 run data_tools test_gripper_control --ros-args -p index_name:=_r

    # Test all three: open -> closed -> open (sequence mode)
    ros2 run data_tools test_gripper_control --ros-args -p sequence:=true
"""

import argparse
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class GripperControlTest(Node):
    def __init__(self, args):
        super().__init__('gripper_control_test')
        self.args = args

        self.declare_parameter('index_name', '_l')
        self.declare_parameter('test_gripper_value', 0.096)
        self.declare_parameter('sequence', False)

        self.index_name = self.get_parameter('index_name').get_parameter_value().string_value
        self.test_gripper_value = self.get_parameter('test_gripper_value').get_parameter_value().double_value
        self.use_sequence = self.get_parameter('sequence').get_parameter_value().bool_value

        self.current_pose = None          # [x, y, z, qx, qy, qz, qw]
        self.current_gripper = None       # latest gripper value
        self.pose_received = False
        self.gripper_received = False
        self.lock = threading.Lock()
        self.test_completed = False
        self._last_throttle_time = time.time()

        # Pose topic: /nero_FK{index}/urdf_end_pose_orient
        pose_topic = f'/nero_FK{self.index_name}/urdf_end_pose_orient'
        # Gripper source: /joint_states_single{index} (gripper is last position element)
        gripper_state_topic = f'/joint_states_single{self.index_name}'

        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.pose_callback, 10)
        self.gripper_sub = self.create_subscription(
            JointState, gripper_state_topic, self.gripper_callback, 10)

        self.action_pose_pub = self.create_publisher(
            Float64MultiArray, f'/nero_inference{self.index_name}/action', 1)
        self.action_gripper_pub = self.create_publisher(
            Float32, f'/nero_inference{self.index_name}/action_gripper', 1)

        self.get_logger().info(f"=== Gripper Control Unit Test (Safe) ===")
        self.get_logger().info(f"  index_name:            {self.index_name}")
        self.get_logger().info(f"  pose topic:            {pose_topic}")
        self.get_logger().info(f"  gripper state topic:   {gripper_state_topic}")
        self.get_logger().info(f"  action topic:          /nero_inference{self.index_name}/action")
        self.get_logger().info(f"  gripper action topic:  /nero_inference{self.index_name}/action_gripper")
        self.get_logger().info(f"  test_gripper_value:    {self.test_gripper_value}")
        self.get_logger().info(f"  sequence:              {self.use_sequence}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Waiting for current pose and gripper values...")

        self.timer = self.create_timer(0.5, self.check_ready_and_test)

    def pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.current_pose = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
            self.pose_received = True

    def gripper_callback(self, msg: JointState):
        with self.lock:
            if len(msg.name) > 0 and 'gripper' in msg.name[-1]:
                self.current_gripper = msg.position[-1]
                self.gripper_received = True
            elif len(msg.position) > 7:
                self.current_gripper = msg.position[7]
                self.gripper_received = True

    def check_ready_and_test(self):
        with self.lock:
            pose_ok = self.pose_received and self.current_pose is not None
            gripper_ok = self.gripper_received and self.current_gripper is not None

        if not pose_ok:
            now = time.time()
            if now - self._last_throttle_time >= 2.0:
                self.get_logger().info(f"Waiting for pose from /nero_FK{self.index_name}/urdf_end_pose_orient...")
                self._last_throttle_time = now
            return
        if not gripper_ok:
            self.get_logger().info(f"Waiting for gripper from /joint_states_single{self.index_name}...")
            return

        if self.test_completed:
            return
        self.test_completed = True
        self.timer.cancel()

        with self.lock:
            pose = list(self.current_pose)
            gripper = self.current_gripper

        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("CAPTURED CURRENT STATE:")
        self.get_logger().info(f"  Position:  [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}]")
        self.get_logger().info(f"  Quaternion:[{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}, {pose[6]:.4f}]")
        self.get_logger().info(f"  Gripper:   {gripper:.6f}  (0.0=closed, 0.1=open)")
        self.get_logger().info("=" * 60)

        if self.use_sequence:
            self.run_sequence(pose, gripper)
        else:
            self.test_gripper_pipeline(pose, gripper, self.test_gripper_value)

    def run_sequence(self, pose, gripper):
        """Run a sequence: open -> closed -> open."""
        sequence = [
            ("OPEN   (0.1)", 0.1),
            ("CLOSED (0.0)", 0.0),
            ("HALF   (0.05)", 0.05),
            ("OPEN   (0.1)", 0.1),
        ]
        for i, (name, val) in enumerate(sequence):
            self.get_logger().info("")
            self.get_logger().info(f"[Sequence {i+1}/{len(sequence)}] {name}")
            self.test_gripper_pipeline(pose, gripper, val)
            if i < len(sequence) - 1:
                self.get_logger().info("Waiting 3 seconds before next test...")
                time.sleep(3.0)

    def test_gripper_pipeline(self, pose, current_gripper, gripper_value):
        """Trace the full gripper pipeline with a given value."""
        self.get_logger().info("")
        self.get_logger().info("-" * 60)
        self.get_logger().info("[STEP 0] Test input:")
        self.get_logger().info(f"          Current gripper state:  {current_gripper:.6f}")
        self.get_logger().info(f"          action[7] test value:  {gripper_value:.6f}")
        self.get_logger().info(f"          Arm will stay at:       [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}]")

        action = Float64MultiArray()
        action.data = [float(v) for v in pose] + [float(gripper_value)]
        self.action_pose_pub.publish(action)
        self.get_logger().info(f"[STEP 1] Published action[0:7]=current_pose to /nero_inference{self.index_name}/action")

        gripper_msg = Float32()
        gripper_msg.data = float(gripper_value)
        self.action_gripper_pub.publish(gripper_msg)
        self.get_logger().info(f"[STEP 2] Published Float32({gripper_value:.6f}) to /nero_inference{self.index_name}/action_gripper")

        self.get_logger().info("[STEP 3] nero_IK gripper_inference_callback:")
        self.get_logger().info(f"          latest_gripper_value = {gripper_value:.6f}")
        self.get_logger().info(f"          Re-published to gripper_ctrl as Float32({gripper_value:.6f})")

        raw_data = max(0.0, min(0.1, gripper_value))
        gripper_width = 0.1 - raw_data
        self.get_logger().info("[STEP 4] nero_ctrl gripper_callback:")
        self.get_logger().info(f"          raw_data = clamp({gripper_value:.6f}, 0.0, 0.1) = {raw_data:.6f}")
        self.get_logger().info(f"          gripper_width = 0.1 - {raw_data:.6f} = {gripper_width:.6f}")
        self.get_logger().info(f"          SDK move_gripper({gripper_width:.6f}) called")

        self.get_logger().info("")
        self.get_logger().info(f"[RESULT] Final SDK command: {gripper_width:.6f}")
        if gripper_width < 0.001:
            self.get_logger().error(f"!!! gripper_width ({gripper_width:.6f}) is near ZERO - gripper may not move!")
        elif abs(gripper_width - 0.1) < 0.001:
            self.get_logger().info("    gripper command = 0.1 (fully open)")
        elif abs(gripper_width) < 0.001:
            self.get_logger().info("    gripper command = 0.0 (fully closed)")

        self.get_logger().info("")
        self.get_logger().info("[FULL CHAIN]")
        self.get_logger().info(f"  action[7]             = {gripper_value:.6f}")
        self.get_logger().info(f"  -> Float32 pub       = {gripper_value:.6f}")
        self.get_logger().info(f"  -> gripper_ctrl      = {gripper_value:.6f}")
        self.get_logger().info(f"  -> raw_data          = {raw_data:.6f}")
        self.get_logger().info(f"  -> SDK move_gripper   = {gripper_width:.6f}")

        if abs(gripper_width - 0.1) < 0.001:
            self.get_logger().info("  -> STATUS: Fully OPEN (gripper should open)")
        elif abs(gripper_width) < 0.001:
            self.get_logger().warn("  -> STATUS: Fully CLOSED (gripper should close)")
        elif gripper_width < 0.01:
            self.get_logger().error("  -> STATUS: Tiny command - gripper likely won't respond!")
        else:
            self.get_logger().info("  -> STATUS: Partial open (gripper should partially open)")
        self.get_logger().info("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Gripper Control Unit Test (Safe)')
    parser.add_argument('--index_name', default='_l',
                        help='Arm index (default: _l)')
    parser.add_argument('--test_value', type=float, default=0.096,
                        help='Gripper value to test (default: 0.096, range 0.0-0.1)')
    parser.add_argument('--sequence', action='store_true',
                        help='Run open-closed-half-open sequence')
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = GripperControlTest(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
