from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'index_name',
            default_value='',
            description='Arm index name: "" for single, "_l" for left, "_r" for right'
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='/home/yxgn/inference_capture',
            description='Directory to save captured inference action JSON files'
        ),
        DeclareLaunchArgument(
            'verbose',
            default_value='true',
            description='Enable verbose logging (true/false)'
        ),
    ]

    index_name = LaunchConfiguration('index_name')
    output_dir = LaunchConfiguration('output_dir')
    verbose = LaunchConfiguration('verbose')

    return LaunchDescription(declared_arguments + [
        Node(
            package='data_tools',
            executable='inference_action_capture',
            name='inference_action_capture',
            parameters=[{
                'index_name': index_name,
                'output_dir': output_dir,
                'verbose': verbose,
            }],
            output='screen',
        )
    ])
