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
from typing import Any, Dict

import datasets
import numpy as np
import safetensors
import torch
from backports.strenum import StrEnum

from barrel.pipes.vlams.data.robotics.hf.utils.lerobot_utils import (
    filter_episode_metadata,
    load_episode_data_index,
    load_episode_metadata,
    load_hf_dataset,
    load_info,
    np_column,
)

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
    ROBOT_BASE = 'robot_base'  # Translation / rotation expressed in robot base frame
    WORLD = "world"  # Translation / rotation expressed in a world frame (not robot base)
    END_EFFECTOR = "end_effector"  # Translation / rotation expressed in end-effector frame
    CAMERA = "camera"  # Translation / rotation expressed in camera frame
    UNKNOWN = 'unknown'


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
        if not are_values_increasing(self.np_column('episode_index')):
            raise ValueError(f"Dataset {dataset_path} doesn't have contiguous episodes")

        # Load the episode metadata
        # TODO: Merge episode_metadata in the main dataset. Columns can be dropped if not needed
        episode_metadata = load_episode_metadata(dataset_path, split)
        if episode_metadata is None:
            episode_metadata = datasets.Dataset.from_dict({'episode_index': np.arange(len(self.hf_dataset))})
        self.episode_metadata = episode_metadata

        if split in ["train", "test"]:
            # Contains keys 'from' and 'to'
            self.episode_data_index: Dict[str, np.ndarray] = load_episode_data_index(dataset_path, split)
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
        return len(self.episode_data_index['from'])

    @property
    def episode_lengths(self) -> np.ndarray:
        return self.episode_data_index['to'] - self.episode_data_index['from']

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

    def episode_index_from_index(self, index: int | np.ndarray) -> int | np.ndarray:
        assert 0 <= index < len(self), f"0 <= {index} < {len(self)} not true"
        episode_index = np.searchsorted(self.episode_data_index['from'], index, side='right') - 1
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
        obj.episode_metadata = filter_episode_metadata(hf_dataset, self.episode_metadata)

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
        return ReferenceFrame(self.info.get('control_reference_frame', 'unknown'))


def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> Dict[str, np.ndarray]:
    """
    Calculates episode boundaries for `hf_dataset` using 'episode_index' column.
    NOTE: Assumes 'episode_index' is unique. This can break if you concatenate two datasets together
    """
    episode_ids = np_column(hf_dataset, 'episode_index')

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
    indices_of_last_frames = np.concatenate([indices_of_first_frames[1:], [len(hf_dataset)]])

    if np.any(np.diff(inverse_indices) < 0):
        raise ValueError(
            f"Dataset 'episode_index' not contiguous. Likely, separate datasets have been concatenated and "
            f"episodes from different datasets got the same episode_index.\nDataset: {hf_dataset.info}\n"
            f"indices_of_first_frames: {indices_of_first_frames}\n"
            f"indices_of_last_frames: {indices_of_last_frames}\n"
            f"groups: {inverse_indices}"
        )

    return {
        'from': np.asarray(indices_of_first_frames, dtype=np.int64),
        'to': np.asarray(indices_of_last_frames, dtype=np.int64),
    }


def are_values_increasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) >= 0))


def save_to_disk(dataset: LeRobotDataset, output_path: str) -> None:
    """Save the dataset to disk"""

    split = 'train' if dataset.split.startswith('train') else 'test'
    hf_output_path = os.path.join(output_path, split)
    meta_data_path = os.path.join(output_path, 'meta_data')

    if os.path.exists(hf_output_path):
        raise FileExistsError(f"Output path {hf_output_path} already exists")
    else:  # noqa: RET506
        os.makedirs(hf_output_path)
        os.makedirs(meta_data_path, exist_ok=True)
        os.makedirs(os.path.join(meta_data_path, split), exist_ok=True)

    # Save the hf dataset
    dataset.hf_dataset.save_to_disk(hf_output_path)

    # Save the episode metadata dataset
    if dataset.episode_metadata.column_names != ['episode_index']:
        dataset.episode_metadata.save_to_disk(
            os.path.join(output_path, 'meta_data', split, 'episode_metadata')
        )

    # Save the info
    with open(os.path.join(meta_data_path, 'info.json'), 'w') as f:
        json.dump(dataset.info, f)

    # Save the episode data index
    episode_data_index = {key: torch.tensor(value) for key, value in dataset.episode_data_index.items()}
    safetensors.torch.save_file(
        episode_data_index, os.path.join(meta_data_path, split, 'episode_data_index.safetensors')
    )

if __name__ == "__main__":
    # Test
    breakpoint()
    dataset = LeRobotDataset(
        "/work/sombit_dey/insait_droid/insait_droid/"
    )
    
    print(dataset)
    print(dataset[0])