"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import cv2
import os
import argparse
import math
import yaml
import glob
from random import choice


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 8
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    args,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    states = []
    actions = []
    if len(states) == 0:
        for i in range(len(args.armJointStateNames)):
            if "puppet" in args.armJointStateNames[i]:
                for j in range(args.armJointStateDims[i]):
                    states += [f'arm.jointStatePosition.{args.armJointStateNames[i]}.joint{j}']
            if "master" in args.armJointStateNames[i]:
                for j in range(args.armJointStateDims[i]):
                    actions += [f'arm.jointStatePosition.{args.armJointStateNames[i]}.joint{j}']

    if len(states) == 0:
        for i in range(len(args.armEndPoseNames)):
            if "puppet" in args.armEndPoseNames[i]:
                for j in range(args.armEndPoseDims[i]):
                    states += [f'arm.endPose.{args.armEndPoseNames[i]}.joint{j}']
            if "master" in args.armEndPoseNames[i]:
                for j in range(args.armEndPoseDims[i]):
                    actions += [f'arm.endPose.{args.armEndPoseNames[i]}.joint{j}']

    if len(states) == 0:
        for i in range(len(args.localizationPoseNames)):
            states += [f'localization.pose.{args.localizationPoseNames[i]}.x']
            states += [f'localization.pose.{args.localizationPoseNames[i]}.y']
            states += [f'localization.pose.{args.localizationPoseNames[i]}.z']
            states += [f'localization.pose.{args.localizationPoseNames[i]}.roll']
            states += [f'localization.pose.{args.localizationPoseNames[i]}.pitch']
            states += [f'localization.pose.{args.localizationPoseNames[i]}.yaw']
            states += [f'localization.pose.{args.localizationPoseNames[i]}.gripper']
            actions += [f'localization.pose.{args.localizationPoseNames[i]}.x']
            actions += [f'localization.pose.{args.localizationPoseNames[i]}.y']
            actions += [f'localization.pose.{args.localizationPoseNames[i]}.z']
            actions += [f'localization.pose.{args.localizationPoseNames[i]}.roll']
            actions += [f'localization.pose.{args.localizationPoseNames[i]}.pitch']
            actions += [f'localization.pose.{args.localizationPoseNames[i]}.yaw']
            actions += [f'localization.pose.{args.localizationPoseNames[i]}.gripper']
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": (len(states),),
            "names": [
                states,
            ],
        },
        "action": {
            "dtype": "float64",
            "shape": (len(actions),),
            "names": [
                actions,
            ],
        }
    }
    for camera in args.cameraColorNames:
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (3, args.imageHeight, args.imageWidth),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    # for camera in args.cameraDepthNames:
    #     features[f"observation.depths.{camera}"] = {
    #         # "dtype": mode,
    #         # "shape": (3, args.imageHeight, args.imageWidth),
    #         # "names": [
    #         #     "channels",
    #         #     "height",
    #         #     "width",
    #         # ],
    #         "dtype": "uint16",
    #         "shape": (args.imageHeight, args.imageWidth)
    #     }
    if args.useCameraPointCloud:
        for camera in args.cameraPointCloudNames:
            features[f"observation.pointClouds.{camera}"] = {
                "dtype": "float64",
                "shape": ((args.pointNum * 6),)
            }

    return LeRobotDataset.create(
        repo_id=args.datasetName,
        root=args.targetDir,
        fps=args.fps,
        robot_type=args.robotType,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_episode_data(
    args,
    episode_path: Path,
):
    with h5py.File(episode_path, "r") as episode:
        try:
            states = None
            if states is None and len(args.armJointStateNames) > 0:
                states = torch.from_numpy(
                    np.concatenate(
                        [episode[f"arm/jointStatePosition/{name}"][()] for name in args.armJointStateNames if "puppet" in name], axis=1
                    )
                )
                actions = torch.from_numpy(
                    np.concatenate(
                        [episode[f"arm/jointStatePosition/{name}"][()] for name in args.armJointStateNames if "master" in name], axis=1
                    )
                )
            if states is None and len(args.armEndPoseNames) > 0:
                states = torch.from_numpy(
                    np.concatenate(
                        [episode[f"arm/endPose/{name}"][()] for name in args.armEndPoseNames if "puppet" in name], axis=1
                    )
                )
                actions = torch.from_numpy(
                    np.concatenate(
                        [episode[f"arm/endPose/{name}"][()] for name in args.armEndPoseNames if "master" in name], axis=1
                    )
                )
            if states is None and len(args.localizationPoseNames) > 0:
                states = torch.from_numpy(
                    np.concatenate(
                        [np.concatenate((episode[f"localization/pose/{name}"][()], episode[f"gripper/encoderDistance/{name}"][()].reshape(-1, 1)), axis=1) for name in args.localizationPoseNames], axis=1
                    )
                )
                actions = torch.from_numpy(
                    np.concatenate(
                        [np.concatenate((episode[f"localization/pose/{name}"][()], episode[f"gripper/encoderDistance/{name}"][()].reshape(-1, 1)), axis=1) for name in args.localizationPoseNames], axis=1
                    )
                )
            colors = {}
            for camera in args.cameraColorNames:
                colors[camera] = []
                for i in range(episode[f'camera/color/{camera}'].shape[0]):
                    if episode[f'/camera/color/{camera}'].ndim == 1:
                        colors[camera].append(os.path.join(os.path.dirname(str(episode_path)), episode[f'camera/color/{camera}'][i].decode('utf-8')))
                        # colors[camera].append(cv2.cvtColor(cv2.imread(
                        #    os.path.join(os.path.dirname(str(episode_path)), episode[f'camera/color/{camera}'][i].decode('utf-8')),
                        #    cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
                    else:
                        colors[camera].append(episode[f'camera/color/{camera}'][i])
                colors[camera] = colors[camera]
            depths = {}
            # for camera in args.cameraDepthNames:
            #     depths[camera] = []
            #     for i in range(episode[f'camera/depth/{camera}'].shape[0]):
            #         depths[camera].append(cv2.imread(
            #             os.path.join(os.path.dirname(str(episode_path)), episode[f'camera/depth/{camera}'][i].decode('utf-8')),
            #             cv2.IMREAD_UNCHANGED))
            pointclouds = {}
            if args.useCameraPointCloud:
                for camera in args.cameraPointCloudNames:
                    pointclouds[camera] = []
                    for i in range(episode[f'camera/pointCloud/{camera}'].shape[0]):
                        pointclouds[camera].append(np.load(
                            os.path.join(os.path.dirname(str(episode_path)), episode[f'camera/color/{camera}'][i].decode('utf-8'))))
            return colors, depths, pointclouds, states, actions
        except:
            return None, None, None, None, None

def populate_dataset(
    args,
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
) -> LeRobotDataset:
    error_file = []
    episodes = range(len(hdf5_files))
    for ep_idx in tqdm.tqdm(episodes):
        episode_path = hdf5_files[ep_idx]
        task = 'null'
        try:
            with h5py.File(episode_path, 'r') as root:
                task = choice(root[f'instructions/full_instructions/text'][()]).decode('utf-8')
        except:
            pass
        colors, depths, pointclouds, states, actions = load_episode_data(args, episode_path)
        if colors is not None:
            num_frames = states.shape[0]

            for i in range(num_frames):
                frame = {
                    # 'task': task,
                    "observation.state": states[i],
                    "action": actions[i],
                }
                for camera, color in colors.items():
                    frame[f"observation.images.{camera}"] = color[i]
                # for camera, depth in depths.items():
                #     frame[f"observation.depths.{camera}"] = depth[i]
                if args.useCameraPointCloud:
                    for camera, pointcloud in pointclouds.items():
                        frame[f"observation.pointClouds.{camera}"] = pointcloud[i]
                dataset.add_frame(frame, task)
            dataset.save_episode()
        else:
            error_file.append(episode_path)
    # print("error:", error_file)
    return dataset


def process(
    args,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    hdf5_files = []
    if Path(args.targetDir).exists():
        shutil.rmtree(Path(args.targetDir))

    for datasetDir in args.datasetDir:
        dataset_dir = Path(datasetDir)
        if not dataset_dir.exists():
            raise ValueError(f"{dataset_dir} does not exist")
        for f in os.listdir(dataset_dir):
            if f.endswith(".hdf5"):
                hdf5_files.append(os.path.join(dataset_dir, f))
            if os.path.isdir(os.path.join(dataset_dir, f)):
                hdf5_files.extend(glob.glob(os.path.join(dataset_dir, f, "*.hdf5")))
    hdf5_files = sorted(hdf5_files)
    dataset = create_empty_dataset(
        args,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        args,
        dataset,
        hdf5_files
    )
    
    # Generate stats.json for backward compatibility
    from lerobot.datasets.utils import write_stats
    if dataset.meta.stats:
        write_stats(dataset.meta.stats, dataset.root)
        print(f"Stats written to {dataset.root / 'meta/stats.json'}")
    else:
        print("Warning: No stats available to write")

    # persist the sorted list into targetDir/meta/episode_list.txt
    meta_dir = Path(args.targetDir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episode_list_path = meta_dir / "episode_list.txt"
    with open(episode_list_path, "w", encoding="utf-8") as f:
        for file_path in hdf5_files:
            f.write(f"{Path(file_path).stem}\n")
    print(f"Saved HDF5 list to {episode_list_path}")

    if push_to_hub:
        dataset.push_to_hub()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetDir', action='store', type=str, help='datasetDir', nargs='+', required=True)
    parser.add_argument('--datasetName', action='store', type=str, help='datasetName',
                        default="data", required=False)
    parser.add_argument('--type', action='store', type=str, help='type',
                        default="aloha", required=False)
    parser.add_argument('--targetDir', action='store', type=str, help='targetDir',
                        default="/home/agilex/data", required=False)
    parser.add_argument('--robotType', action='store', type=str, help='robotType',
                        default="cobot_magic", required=False)
    parser.add_argument('--fps', action='store', type=int, help='fps',
                        default=30, required=False)
    parser.add_argument('--imageWidth', action='store', type=int, help='imageWidth',
                        default=1280, required=False)
    parser.add_argument('--imageHeight', action='store', type=int, help='imageHeight',
                        default=720, required=False)
    parser.add_argument('--cameraColorNames', action='store', type=str, help='cameraColorNames',
                        default=[], required=False)
    parser.add_argument('--cameraDepthNames', action='store', type=str, help='cameraDepthNames',
                        default=[], required=False)
    parser.add_argument('--cameraPointCloudNames', action='store', type=str, help='cameraPointCloudNames',
                        default=[], required=False)
    parser.add_argument('--useCameraPointCloud', action='store', type=bool, help='useCameraPointCloud',
                        default=False, required=False)
    parser.add_argument('--pointNum', action='store', type=int, help='point_num',
                        default=5000, required=False)
    parser.add_argument('--armJointStateNames', action='store', type=str, help='armJointStateNames',
                        default=[], required=False)
    parser.add_argument('--armJointStateDims', action='store', type=int, help='armJointStateDims',
                        default=[], required=False)
    parser.add_argument('--armEndPoseNames', action='store', type=str, help='armEndPoseNames',
                        default=[], required=False)
    parser.add_argument('--armEndPoseDims', action='store', type=int, help='armEndPoseDims',
                        default=[], required=False)
    parser.add_argument('--localizationPoseNames', action='store', type=str, help='localizationPoseNames',
                        default=[], required=False)
    parser.add_argument('--gripperEncoderNames', action='store', type=str, help='gripperEncoderNames',
                        default=[], required=False)
    parser.add_argument('--imu9AxisNames', action='store', type=str, help='imu9AxisNames',
                        default=[], required=False)
    parser.add_argument('--lidarPointCloudNames', action='store', type=str, help='lidarPointCloudNames',
                        default=[], required=False)
    parser.add_argument('--robotBaseVelNames', action='store', type=str, help='robotBaseVelNames',
                        default=[], required=False)
    parser.add_argument('--liftMotorNames', action='store', type=str, help='liftMotorNames',
                        default=[], required=False)
    args = parser.parse_args()

    with open(f'../config/{args.type}_data_params.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)
        args.cameraColorNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('camera', {}).get('color', {}).get('names', [])
        args.cameraDepthNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('camera', {}).get('depth', {}).get('names', [])
        args.cameraPointCloudNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('camera', {}).get('pointCloud', {}).get('names', [])
        args.armJointStateNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('arm', {}).get('jointState', {}).get('names', [])
        args.armEndPoseNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('arm', {}).get('endPose', {}).get('names', [])
        args.localizationPoseNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('localization', {}).get('pose', {}).get('names', [])
        args.gripperEncoderNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('gripper', {}).get('encoder', {}).get('names', [])
        args.imu9AxisNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('imu', {}).get('9axis', {}).get('names', [])
        args.lidarPointCloudNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('lidar', {}).get('pointCloud', {}).get('names', [])
        args.robotBaseVelNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('robotBase', {}).get('vel', {}).get('names', [])
        args.liftMotorNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('lift', {}).get('motor', {}).get('names', [])
        # args.armJointStateNames = []
        # args.armEndPoseNames = []
        # args.localizationPoseNames = []
        args.armJointStateDims = [7 for _ in range(len(args.armJointStateNames))]
        args.armEndPoseDims = [7 for _ in range(len(args.armEndPoseNames))]
    return args


def main():
    args = get_arguments()
    process(args)


if __name__ == "__main__":
    main()
