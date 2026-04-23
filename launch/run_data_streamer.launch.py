from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    share_dir = get_package_share_directory('data_tools')

    declared_arguments = [
        DeclareLaunchArgument('params_file',
                             default_value=share_dir + '/config/streamer_gripper_params.yaml',
                             description='Path to params YAML file'),
    ]

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription(declared_arguments + [
        Node(
            package='data_tools',
            executable='data_tools_dataStreamer',
            name='data_streamer',
            parameters=[params_file],
            output='screen',
            remappings=[
                # gripper action → nero_ctrl_single_node 的 gripper_ctrl（带 flip 回调）
                ('/nero_inference_l/action_gripper', '/gripper_ctrl_l'),
                ('/nero_inference_r/action_gripper', '/gripper_ctrl_r'),
            ],
        )
    ])
