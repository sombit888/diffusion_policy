# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any, Dict, Union

import datasets

import numpy as np
import safetensors.torch
import torch
from backports.strenum import StrEnum
from torchvision import transforms
from dataclasses import dataclass
from PIL import Image
from functools import partial

# from barrel.pipes.vlams.data.robotics.hf.utils.lerobot_utils import (
#     filter_episode_metadata,
#     load_episode_data_index,
#     load_episode_metadata,
#     load_hf_dataset,
#     load_info,
#     np_column,
# )
from diffusion_policy.dataset.lerobot_utils import (
    filter_episode_metadata,
    load_episode_data_index,
    load_episode_metadata,
    load_hf_dataset,
    load_info,
    np_column,
)
from diffusion_policy.dataset.filter import add_target_joint_position, filter_episodes , add_target_joint_delta, select_hf_dataset_episodes
from diffusion_policy.dataset.base_dataset import BaseImageDataset

# For maintainers, see lerobot/common/datasets/push_dataset_to_hub/CODEBASE_VERSION.md
CODEBASE_VERSION = "v1.6"


# class ControlReference(StrEnum):
#     # TODO: Inherit from enum.StrEnum in python 3.11
#     """Indicates how control values are expressed"""
#     RELATIVE = "relative"  # Translation / rotation expressed w.r.t. the current end-effector position
#     WORLD = "world"  # Translation / rotation expressed w.r.t. fixed world frame
#     UKNOWN = "unknown"


class ReferenceFrame(StrEnum):
    # TODO: Inherit from enum.StrEnum in python 3.11
    """Indicates the frame w.r.t. which control values are expressed"""
    ROBOT_BASE = "robot_base"  # Translation / rotation expressed in robot base frame
    WORLD = (
        "world"  # Translation / rotation expressed in a world frame (not robot base)
    )
    END_EFFECTOR = (
        "end_effector"  # Translation / rotation expressed in end-effector frame
    )
    CAMERA = "camera"  # Translation / rotation expressed in camera frame
    UNKNOWN = "unknown"


@dataclass
class EpisodeMetadata:
    """
    Packs episodes metadata for a given dataframe. Example:
        episode_ids: [1, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4]
        indices_of_first_frames: [0, 3, 5, 9]
        indices_of_last_frames: [2, 4, 8, 10]
        inverse_indices: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]
    """

    episode_ids: (
        np.ndarray
    )  # Unique episode ids in the order they appear in the dataframe
    indices_of_first_frames: (
        np.ndarray
    )  # First indices of the episode ids in the dataframe
    indices_of_last_frames: (
        np.ndarray
    )  # Last indices of the episode ids. These are **including**
    inverse_indices: (
        np.ndarray
    )  # Array of indices that reconstructs the episode id array in the dataframe


def extract_episode_metadata(episode_ids: np.ndarray) -> EpisodeMetadata:
    # Note that np.unique returns the values in a sorted order, even if the source is not sorted

    # np.unique returns the values in a sorted order, even if the source is not sorted. Thus, if
    # episode_ids isn't sorted, we need to apply np.unique twice to get indices w.r.t. original array.
    # Example:
    #   episode_ids: [1, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4]
    #   indices: [5, 0, 3, 9]
    #   inverse: [1, 1, 1, 2, 2, 0, 0, 0, 0, 3, 3]
    #   indices_of_first_frames: [0, 3, 5, 9]
    #   inverse_indices: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]

    _, indices, inverse = np.unique(episode_ids, return_index=True, return_inverse=True)

    # Re-adjust such that the order matches the order in the source
    _, indices_of_first_frames, inverse_indices, counts = np.unique(
        indices[inverse], return_index=True, return_inverse=True, return_counts=True
    )

    _assert_episodes_are_contiguous(inverse_indices, episode_ids)

    indices_of_last_frames = indices_of_first_frames + counts - 1

    episode_ids = episode_ids[indices_of_first_frames]

    return EpisodeMetadata(
        episode_ids=episode_ids,
        indices_of_first_frames=indices_of_first_frames,
        indices_of_last_frames=indices_of_last_frames,
        inverse_indices=inverse_indices,
    )


def _assert_episodes_are_contiguous(
    inverse_indices: np.ndarray, episode_ids: np.ndarray
):
    if np.any((diff := np.diff(inverse_indices)) < 0):
        offending_indices = np.where(diff < 0)[0] + 1
        offending_episode_ids = np.unique(episode_ids[offending_indices])

        with np.printoptions(threshold=np.iinfo(np.int32).max):
            raise ValueError(
                f"Episodes not contiguoues in the dataframe. Likely, separate dataframes have been "
                f"concatenated and episodes from different dataframes got the same episode id. "
                f"Offending episode_ids: {offending_episode_ids.size} / {np.unique(episode_ids).size}"
                f"\n{offending_episode_ids}"
            )


class LeRobotDataset:
    """
    Fork of LeRobotDataset from lerobot/lerobot/common/datasets/lerobot_dataset.py

    Explicitly forking for the time being since:
        - Official `LeRobotDataset` seems to be going through rapid prototyping and updates and doesn't seem
        to have reached a stable version yet (no official releases or cycles). We want to maintain stability
        - Official `LeRobotDataset` doesn't have support for some features we need, e.g.
            - Train/val split based on %
            - Removing troch transform on the dataset
            - Faster episode indexing

    Once official `LeRobotDataset` has reached stable format and API, this class can be
    removed or implemented as a wrapper around it

    bridge/
    ├── meta_data
    │   ├── info.json
    │   ├── test (optional)
    │   │   ├── episode_data_index.safetensors
    │   │   └── episode_metadata
    │   │       ├── data-00000-of-00001.arrow
    │   │       ├── dataset_info.json
    │   │       └── state.json
    │   └── train
    │       ├── episode_data_index.safetensors
    │       └── episode_metadata
    │           ├── data-00000-of-00001.arrow
    │           ├── dataset_info.json
    │           └── state.json
    ├── test (optional)
    │   ├── data-00000-of-00001.arrow
    │   ├── dataset_info.json
    │   └── state.json
    └── train
        ├── data-00000-of-00001.arrow
        ├── dataset_info.json
        └── state.json
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
    ):
        """
        Args:
            dataset_path: Either a full dataset path on disk or HF dataset ID
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split

        # Load the core dataset
        self.hf_dataset: datasets.Dataset = load_hf_dataset(dataset_path, split=split)

        # Make sure all episodes are contiguous and ordered in increasing order
        if not are_values_increasing(self.np_column("episode_index")):
            raise ValueError(f"Dataset {dataset_path} doesn't have contiguous episodes")

        # Load the episode metadata
        # TODO: Merge episode_metadata in the main dataset. Columns can be dropped if not needed
        episode_metadata = load_episode_metadata(dataset_path, split)
        if episode_metadata is None:
            episode_metadata = datasets.Dataset.from_dict(
                {"episode_index": np.arange(len(self.hf_dataset))}
            )
        self.episode_metadata = episode_metadata

        if split in ["train", "test"]:
            # Contains keys 'from' and 'to'
            self.episode_data_index: Dict[str, np.ndarray] = load_episode_data_index(
                dataset_path, split
            )
        else:
            self.episode_data_index = calculate_episode_data_index(self.hf_dataset)

        self.info: Dict[str, Any] = load_info(dataset_path)

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image stream from cameras."""
        keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, datasets.Image):
                keys.append(key)
        return keys

    @property
    def num_episodes(self) -> int:
        return len(self.episode_data_index["from"])

    @property
    def episode_lengths(self) -> np.ndarray:
        return self.episode_data_index["to"] - self.episode_data_index["from"]

    def np_column(self, column_name: str) -> np.ndarray:
        return np_column(self.hf_dataset, column_name)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    @property
    def dataset_name(self) -> str:
        return os.path.basename(os.path.normpath(self.dataset_path))

    def episode_index_from_index(
        self, index: Union[int, np.ndarray]
    ) -> Union[int, np.ndarray]:
        assert 0 <= index < len(self), f"0 <= {index} < {len(self)} not true"
        episode_index = (
            np.searchsorted(self.episode_data_index["from"], index, side="right") - 1
        )
        return episode_index

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Dataset path: '{self.dataset_path}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {len(self)},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: image (.png),\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Codebase Version: {self.info.get('codebase_version', '< v1.6')},\n"
            f")"
        )

    def replace_hf_dataset(self, hf_dataset: datasets.Dataset) -> "LeRobotDataset":
        obj = self.__class__.__new__(self.__class__)

        # Propagate all attributes
        obj.dataset_path = self.dataset_path
        obj.split = self.split
        obj.info = self.info

        # Update the hf_dataset and the episode data index
        obj.hf_dataset = hf_dataset
        obj.episode_data_index = calculate_episode_data_index(hf_dataset)
        obj.episode_metadata = filter_episode_metadata(
            hf_dataset, self.episode_metadata
        )

        return obj

    @property
    def control_reference_frame(self) -> ReferenceFrame:
        """
        Let the end-effector control at each timestep `k` be expressed in a frame [R_k | t_k].
        This gives the reference frame for the transform, e.g. robot_base, eef, etc.
        Think of it as the reference frame for the end-effector pose after executing the control
        NOTE:
            - `t_k` is NOT the translation control, but might be equal to it when:
                - Translation control is deltas and the reference frame is the end-effector and R is
                    fixed for every timestep
                - Translation control is absolute cartesian coordinates (in robot base), the reference frame
                    is the robot base and R is fixed for every timestep
            - `R_k` is NOT the rotation control
            - `R_k` specifies the XYZ axes alignment for translation control
            - `R_k` might be in different reference frame from `t_k`, e.g. `t_k` is expressed in end-effector
                frame, but the XYZ axes are always fixed and in robot base frame (the most common case)
        """
        return ReferenceFrame(self.info.get("control_reference_frame", "unknown"))


def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> Dict[str, np.ndarray]:
    """
    Calculates episode boundaries for `hf_dataset` using 'episode_index' column.
    NOTE: Assumes 'episode_index' is unique. This can break if you concatenate two datasets together
    """
    episode_ids = np_column(hf_dataset, "episode_index")

    # np.unique returns the values in a sorted order, even if the source is not sorted. Thus, if
    # episode_ids isn't sorted, we need to apply np.unique twice to get indices w.r.t. original array.
    # Example:
    #   episode_ids: [1, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4]
    #   indices: [5, 0, 3, 9]
    #   inverse: [1, 1, 1, 2, 2, 0, 0, 0, 0, 3, 3]
    #   indices_of_first_frames: [0, 3, 5, 9]
    #   inverse_indices: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]

    _, indices, inverse = np.unique(episode_ids, return_index=True, return_inverse=True)
    _, indices_of_first_frames, inverse_indices = np.unique(
        indices[inverse], return_index=True, return_inverse=True
    )
    indices_of_last_frames = np.concatenate(
        [indices_of_first_frames[1:], [len(hf_dataset)]]
    )

    if np.any(np.diff(inverse_indices) < 0):
        raise ValueError(
            f"Dataset 'episode_index' not contiguous. Likely, separate datasets have been concatenated and "
            f"episodes from different datasets got the same episode_index.\nDataset: {hf_dataset.info}\n"
            f"indices_of_first_frames: {indices_of_first_frames}\n"
            f"indices_of_last_frames: {indices_of_last_frames}\n"
            f"groups: {inverse_indices}"
        )

    return {
        "from": np.asarray(indices_of_first_frames, dtype=np.int64),
        "to": np.asarray(indices_of_last_frames, dtype=np.int64),
    }


def are_values_increasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) >= 0))


def save_to_disk(dataset: LeRobotDataset, output_path: str) -> None:
    """Save the dataset to disk"""

    split = "train" if dataset.split.startswith("train") else "test"
    hf_output_path = os.path.join(output_path, split)
    meta_data_path = os.path.join(output_path, "meta_data")

    if os.path.exists(hf_output_path):
        raise FileExistsError(f"Output path {hf_output_path} already exists")
    else:  # noqa: RET506
        os.makedirs(hf_output_path)
        os.makedirs(meta_data_path, exist_ok=True)
        os.makedirs(os.path.join(meta_data_path, split), exist_ok=True)

    # Save the hf dataset
    dataset.hf_dataset.save_to_disk(hf_output_path)

    # Save the episode metadata dataset
    if dataset.episode_metadata.column_names != ["episode_index"]:
        dataset.episode_metadata.save_to_disk(
            os.path.join(output_path, "meta_data", split, "episode_metadata")
        )

    # Save the info
    with open(os.path.join(meta_data_path, "info.json"), "w") as f:
        json.dump(dataset.info, f)

    # Save the episode data index
    episode_data_index = {
        key: torch.tensor(value) for key, value in dataset.episode_data_index.items()
    }
    safetensors.torch.save_file(
        episode_data_index,
        os.path.join(meta_data_path, split, "episode_data_index.safetensors"),
    )


class LeRobotDatasetDiffusion(LeRobotDataset, BaseImageDataset):
    """
    A dataset class for LeRobotDataset that inherits from BaseImageDataset.
    This class is specifically designed to work with image data in the Diffusion CodeBase format.
    """

    def __init__(self, dataset_path: str, split: str = "train"):
        super().__init__(dataset_path=dataset_path, split=split)
        self._init_args = {
            "dataset_path": dataset_path,
            "split": split,
        }

        self.keys_to_add = [
            "observation.robot_state.cartesian_position",
            "observation.robot_state.gripper_position",
            "observation.robot_state.joint_positions",
            "observation.robot_state.joint_velocities",
            "action.cartesian_position",
            "action.cartesian_velocity",
            "action.gripper_position",
            "action.gripper_velocity",
            "action.joint_position",
            "action.joint_velocity",
            "action.robot_state.cartesian_position",
            "action.robot_state.gripper_position",
            "action.joint_position",
            "action.robot_state.joint_velocities",
            "action.target_cartesian_position",
            "action.target_gripper_position",
            "action.target.joint_position",
            "action.target.joint_position_delta",
            # "task",
            # "episode_id",
            "observation.images.main",
            "observation.images.secondary",
            # 'observation.images.wrist_camera',
            # 'observation.timestamp.cameras.main',
            # 'observation.timestamp.cameras.secondary',
            # 'observation.timestamp.cameras.wrist_camera',
            # 'observation.has_wrist',
            # 'timestamp',
            # "episode_index",
            # "frame_index",
            # "index",
        ]
        self.img_keys = [
            "observation.images.main",
            "observation.images.secondary",
            # 'observation.images.wrist_camera',
        ]
        self.to_tensor = transforms.ToTensor()
        # breakpoint()
        # self._add_target_keys(horizon=5)  # Add target keys with horizon of 5
        # breakpoint()

    def filter_episodes(self, drop_columns: list, horizon: int = 5,task_str: str = ""):
        """
        Filter dataset to only include 'open' tasks, drop unused columns, and add future action targets.
        Modifies the dataset in-place.
        """
        # Step 1: Identify open episodes
        if task_str:
            task_list = self.hf_dataset[self.episode_data_index["from"]]["task"]
            open_episode_mask = np.array([ task_str in task for task in task_list])

            # Step 2: Select only 'task_str' episodes
            selected_indices = np.arange(self.num_episodes)[open_episode_mask]
            self.hf_dataset = self.select_episodes_by_index(selected_indices)

            # Step 3: Drop unused columns
        if drop_columns:
            self.hf_dataset = self.hf_dataset.remove_columns(drop_columns)

        # Step 4: Add future targets
        add_target_joint_position(self, horizon=horizon)

        add_target_joint_delta(self, horizon=horizon)
        self.hf_dataset.set_format(type="torch", columns=self.keys_to_add)
        
        # custom_transform = partial(transform_fnc, keys_to_add=self.keys_to_add, img_keys=self.img_keys, to_tensor=self.to_tensor)
        # self.hf_dataset = self.hf_dataset.with_transform(
        #     custom_transform
        # )
        # custom_transform = partial(eager_transform, keys_to_add=self.keys_to_add, img_keys=self.img_keys, to_tensor=self.to_tensor)
        # self.hf_dataset = self.hf_dataset.map(custom_transform, num_proc=8,batched=False)  # batched=True if you want batch processing

    def select_episodes_by_index(self, episode_indices: np.ndarray):
        """
        Replace current dataset with a subset of episodes.
        """
        self.episode_indices = episode_indices
        return select_hf_dataset_episodes(self.hf_dataset,self.episode_data_index, episode_indices)

  
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        # for k in self.img_keys:
        #     item[k] = self.to_tensor(item[k])  # CxHxW

        return item
        
# def transform_fnc(example,keys_to_add,img_keys, to_tensor):
#     item = {}
#     for key in list(example.keys()):
#         if key in keys_to_add:
#             if key in img_keys:
#                 item[key] = to_tensor(np.array(example[key]))
#             elif key not in item:
#                 # If the key is not present, we can add a default value
#                 # Here we assume the default value is None, but it can be changed based on requirements
#                 item[key] = None
#             else:
#                 # If the key is present, we can convert it to a tensor if it's not already
#                 if isinstance(item[key], np.ndarray):
#                     item[key] = torch.tensor(item[key])
#                 elif isinstance(item[key], list):
#                     item[key] = torch.tensor(np.array(item[key]))
#     return item
# def transform_fnc(example, keys_to_add, img_keys, to_tensor):
#     item = {}
#     for key in keys_to_add:
#         if key in example:
#             val = example[key]
#             if key in img_keys:
#                 # Pass PIL image directly to to_tensor
#                 if isinstance(val, Image.Image):
#                     item[key] = to_tensor(val)
#                 elif isinstance(val, np.ndarray):
#                     item[key] = to_tensor(val)
#                 else:
#                     raise TypeError(f"Expected PIL.Image or np.ndarray for {key}, got {type(val)}")
#             else:
#                 # Convert other data types to tensor
#                 if isinstance(val, np.ndarray):
#                     item[key] = torch.tensor(val)
#                 elif isinstance(val, list):
#                     item[key] = torch.tensor(val)
#                 else:
#                     item[key] = torch.tensor([val]) if isinstance(val, (int, float)) else val
#         else:
#             # Default if key not in example
#             item[key] = None
#     return item
def eager_transform(example, keys_to_add, img_keys, to_tensor):
    item = {}
    for key in keys_to_add:
        if key in example:
            val = example[key]
            if key in img_keys:
                # Convert PIL or ndarray to tensor
                if isinstance(val, Image.Image) or isinstance(val, np.ndarray):
                    item[key] = to_tensor(val)
                else:
                    raise TypeError(f"Expected PIL.Image or np.ndarray for {key}, got {type(val)}")
            else:
                if isinstance(val, np.ndarray):
                    item[key] = torch.tensor(val)
                elif isinstance(val, list):
                    item[key] = torch.tensor(val)
                else:
                    item[key] = torch.tensor([val]) if isinstance(val, (int, float)) else val
        else:
            item[key] = None
    return item
def transform_fnc(example, keys_to_add, img_keys, to_tensor):
    # Remove keys that are not needed
    keys_to_remove = set(example.keys()) - set(keys_to_add)
    for key in keys_to_remove:
        del example[key]

    # Process keys in-place
    for key in keys_to_add:
        if key in example:
            val = example[key]
            if key in img_keys:
                # Pass PIL image directly to to_tensor
                if isinstance(val, Image.Image):
                    example[key] = to_tensor(val)
                elif isinstance(val, np.ndarray):
                    example[key] = to_tensor(val)
                else:
                    raise TypeError(f"Expected PIL.Image or np.ndarray for {key}, got {type(val)}")
            else:
                # Convert other data types to tensor
                if isinstance(val, np.ndarray):
                    example[key] = torch.tensor(val)
                elif isinstance(val, list):
                    example[key] = torch.tensor(val)
                else:
                    example[key] = torch.tensor([val]) if isinstance(val, (int, float)) else val
        else:
            # Add key with default value None if not present
            example[key] = None

    return example
def save_rerun_viz_id(dataset:LeRobotDatasetDiffusion, episode_index: int, output_dir: str = '/scratch/sombit_dey/output_rerun/',output_filename:str = 'rerun_viz.rrd'):
    """
    Save a rerun visualization of the dataset for a specific episode index.
    Args:
        dataset (LeRobotDatasetDiffusion): The dataset to visualize.
        episode_index (int): The index of the episode to visualize.
        output_dir (str): The directory to save the visualization.
        output_filename (str): The filename for the visualization.
    """
    import rerun as rr
    from scipy.spatial.transform import Rotation as R
    import time 
    import ipdb; ipdb.set_trace()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the episode data
    episode_data = dataset.hf_dataset[dataset.episode_data_index['from'][episode_index]:dataset.episode_data_index['to'][episode_index]]

    dataset_name = dataset.dataset_name
    axis_length = 0.3
    fps = dataset.fps
    # EEF pose in key observation.robot_state.cartesian_position N,6 D vector, in X,Y,Z, R_x,R_y,R_z
    rr.init("visualize_oxe", recording_id=f"{dataset_name}_{episode_index}")
    cartesian_pose = episode_data['observation.robot_state.cartesian_position'] 
    rpy = episode_data['observation.robot_state.cartesian_position'][:, 3:6]
    rot_mats_np = R.from_euler('xyz', rpy).as_matrix() #  [N, 3, 3] D vector

    # rr.log(
    #     "robot/ee_pose",
    #     rr.Transform3D(
    #         translation=cartesian_pose[:, :3].numpy(),
    #         mat3x3=rot_mats_np,
    #         axis_length=axis_length,
    #     ),
    # )   
    
    # translation_control = dataset.np_column('control.translation')
    # rotation_control = dataset.np_column('control.rotation')
    # gripper_control = dataset.np_column('control.gripper')

    # rotation_obs = rotmat_as_3x3(
    #     convert_rotation(torch.from_numpy(rotation_obs), RotationFormat.ROTMAT)
    # ).numpy()
    # rotation_control = rotmat_as_3x3(
    #     convert_rotation(torch.from_numpy(rotation_control), RotationFormat.ROTMAT)
    # ).numpy()

    # if use_urdf and dataset_name in DATASET_TO_URDF:
    #     urdf_translation, urdf_rotation = compute_fk_poses(dataset, **DATASET_TO_URDF[dataset_name])
    # else:
    #     urdf_translation, urdf_rotation = None, None

    # # Find the episode index for the given episode_id
    # episode_ids = dataset.np_column('episode_index')[dataset.episode_data_index['from']]
    # episode_index: int = np.where(episode_ids == episode_id)[0][0]

    low = dataset.episode_data_index['from'][episode_index]
    high = dataset.episode_data_index['to'][episode_index]

    # language_instruction = str(dataset[int(low)]['observation.language_instruction']).replace(' ', '_')
    start_timestamp = time.time()


    # Set up gripper series
    # rr.log(
    #     "gripper/state",
    #     rr.SeriesPoints(
    #         colors=[255, 0, 0],
    #         markers="circle",
    #         marker_sizes=3,
    #     ),
    #     static=True,
    # )
    # rr.log(
    #     "gripper/control",
    #     rr.SeriesPoints(
    #         colors=[0, 255, 0],
    #         markers="circle",
    #         marker_sizes=3,
    #     ),
    #     static=True,
    # )

    for frame_index in range(low, high):
        # rr.set_time("timestamp", timestamp=(start_timestamp + (frame_index - low) * 1 / fps))
        rr.set_time("frame_index", sequence=frame_index - low)

        rr.log(
            "robot/base", rr.Transform3D(translation=np.zeros(3), mat3x3=np.eye(3), axis_length=axis_length)
        )

        data_point = dataset[frame_index]

        for key, value in data_point.items():
            if key.startswith('observation.images'):
                camera_name = key.replace('observation.images.', '')
                image_np = np.asarray(value)
                rr.log(f"camera/{camera_name}", rr.Image(image_np))

        rr.log(
            "robot/ee_pose",
            rr.Transform3D(
                translation=cartesian_pose[frame_index, :3].numpy(),
                mat3x3=rot_mats_np[frame_index],
                axis_length=axis_length,
            ),
        )   
    
        # EEF pose
        # rr_log_pose("eef_pose", translation_obs[frame_index], rotation_obs[frame_index], axis_length)

        # # EEF pose gripper
        # rr.log("gripper/state", rr.Scalars(gripper_obs[frame_index]))

        # # EEF control
        # rr_log_pose(
        #     "eef_control", translation_control[frame_index], rotation_control[frame_index], axis_length
        # )

        # EEF control gripper
        # rr.log("gripper/control", rr.Scalars(gripper_control[frame_index]))

        # if urdf_translation is not None and urdf_rotation is not None:
        #     rr_log_pose("eef_urdf", urdf_translation[frame_index], urdf_rotation[frame_index], axis_length)

    rr.save(os.path.join(output_dir, f"{output_filename}.rrd"))

if __name__ == "__main__":

    # dataset = LeRobotDatasetDiffusion("/scratch/sombit_dey/insait_droid/insait_droid/")
    # open_episode_mask =  ["open" in task for task in dataset.hf_dataset[dataset.episode_data_index['from']]['episode_index']]
    # dataset_open = filter_episodes(dataset, episode_mask=open_episode_mask)
    # dataset = LeRobotDatasetDiffusion("/scratch/sombit_dey/insait_droid/insait_droid/")
    # open_episode_mask =  ["open" in task for task in dataset.hf_dataset[dataset.episode_data_index['from']]['task']]
    # dataset_open = filter_episodes(dataset=dataset,
    #                                episodes_mask=np.array(open_episode_mask))
    # # delete the orig dataset
    # del dataset
    # breakpoint()
    # # print columns
    # print(dataset_open.hf_dataset.column_names)
    # drop_columns= [ 'observation.images.wrist_camera']
    # dataset_open.hf_dataset = dataset_open.hf_dataset.remove_columns(
    #             drop_columns)
    # # print columns after removing wrist camera
    # print(dataset_open.hf_dataset.column_names)

    # filter columns , remove wrist camera
    # add_target_joint_position(dataset_open, horizon=5)
    
    dataset = LeRobotDatasetDiffusion("/scratch/sombit_dey/insait_droid/insait_droid/")
    dataset.filter_episodes(
            drop_columns=['observation.images.wrist_camera'],
            task_str = 'close',
            horizon=5
        )
    save_rerun_viz_id(dataset, episode_index=0, output_dir='/scratch/sombit_dey/output_rerun/', output_filename='rerun_viz.rrd')