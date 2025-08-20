from typing import Dict, Optional
from typing import Tuple
import datasets
import numpy as np

# from diffusion_policy.dataset.lerobot_dataset import (
#     EpisodeMetadata,
#     extract_episode_metadata,
#     LeRobotDataset,
#     LeRobotDatasetDiffusion,
#     ReferenceFrame,
# )

# from barrel.components.geometry.rotation_transforms import RotationFormat, convert_rotation
# from barrel.core.common.logger import get_logger
# from barrel.pipes.vlams.data.robotics.hf.lerobot_dataset import LeRobotDataset, ReferenceFrame
from diffusion_policy.dataset.lerobot_utils import add_columns
from tqdm import tqdm


def np_unique(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unique elements in data and corresponding indices.

    np.unique returns the values in a sorted order, even if the source is not sorted. Thus, if you simply
    run np.unique on unsorted data, the indices you will get will be invalid.

    """

    # `data` might not be sorted. Thus we need to apply np.unique twice to get indices w.r.t. original array.
    # Example:
    #   data: [1, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4]
    #   indices: [5, 0, 3, 9]
    #   inverse: [1, 1, 1, 2, 2, 0, 0, 0, 0, 3, 3]
    #   indices_of_first_occurence: [0, 3, 5, 9]
    #   inverse_indices: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]
    #   counts: [3, 2, 4, 2]
    #   unique_ids: [1, 2, 0, 4]

    _, indices, inverse = np.unique(data, return_index=True, return_inverse=True)

    # Re-adjust such that the order matches the order in the source
    _, indices_of_first_occurence, inverse_indices, counts = np.unique(
        indices[inverse], return_index=True, return_inverse=True, return_counts=True
    )

    unique_ids = data[indices_of_first_occurence]

    return unique_ids, indices_of_first_occurence, inverse_indices, counts


def np_nearest_neighbour_indices(
    values: np.ndarray, queries: np.ndarray, side: str = "left"
) -> np.ndarray:
    """
    Find the indices of the nearest neighbours of `queries` inside `values`
    Args:
        values: 1D *sorted* array in ascending order
        queries: 1D array of queries for which to find nearest neighbour indices inside `values`
        side: 'left' or 'right'
            left: selects the left nearest neighbour if query is equidistant from its two neighbours
            right: selects the right nearest neighbour if query is equidistant from its two neighbours
    Returns:
        1D array of same size as queries containing the indices of the nearest neighbour in `values`
    """
    # Clip the insert indices to the range of the values array
    insert_indices = np.clip(np.searchsorted(values, queries), 0, len(values) - 1)

    if side == "left":
        nearest_neighbour_indices = np.where(
            np.abs(values[insert_indices] - queries)
            < np.abs(values[insert_indices - 1] - queries),
            insert_indices,
            insert_indices - 1,
        )
    elif side == "right":
        nearest_neighbour_indices = np.where(
            np.abs(values[insert_indices] - queries)
            <= np.abs(values[insert_indices - 1] - queries),
            insert_indices,
            insert_indices - 1,
        )
    else:
        raise ValueError(f"Invalid side: {side}")

    nearest_neighbour_indices = np.clip(nearest_neighbour_indices, 0, len(values) - 1)

    return nearest_neighbour_indices


def np_ranges(lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """
    Create a single array of ranges between lows and highs.
    Example:
        lows = [1, 4, 8]
        highs = [3, 5, 11]
        output -> [1, 2, 4, 8, 9, 10]
    Args:
        lows: Sorted 1D array of start values
        highs: Sorted 1D array of end values
    Returns:
        1D array of the ranges between lows and highs
    """
    assert lows.ndim == highs.ndim == 1, f"{lows.ndim}, {highs.ndim}"
    assert lows.shape == highs.shape, f"{lows.shape}, {highs.shape}"
    assert np.all(highs - lows > 0), f"{highs}, {lows}"

    # Calculate the length of each range
    lengths = highs - lows
    total_size = np.sum(lengths)

    # Create an array of repeated start values
    starts = np.repeat(lows, lengths)  # [total_size]

    # Create an array of repeated sizes of the previous range
    range_sizes = np.repeat(np.cumsum(np.concatenate([[0], lengths[:-1]])), lengths)

    # Create an array of offsets for each range
    offsets = np.arange(total_size) - range_sizes

    # Add the offsets to starts to get the output array
    output = starts + offsets

    return output


def filter_episodes_with_frames(dataset, frame_mask: np.ndarray):
    """
    Filter dataset at episode level. Any episode which contains a value which should be filtered is
    entirely removed (rather than removing just a single entry from the episode)
    Args:
        dataset: LeRobotDataset to be filtered
        frame_mask: 1D np.ndarray of type np.bool_ and the same size as dataset. Values of False indicate
            that the row should be removed from the dataset
    Returns:
        The filtered dataset
    """
    if len(dataset) != len(frame_mask):
        raise ValueError(
            f"Can't apply mask of size {len(frame_mask)} to dataset of size {len(dataset)}"
        )

    if np.all(frame_mask):
        return dataset

    boundaries = dataset.episode_data_index["to"][:-1]
    mask_per_episode = np.split(frame_mask, boundaries)
    episodes_mask = np.asarray([np.all(ep_mask) for ep_mask in mask_per_episode])

    assert (
        len(episodes_mask) == dataset.num_episodes
    ), f"{len(episodes_mask)} != {dataset.num_episodes}"

    # All episodes stay, return original dataset
    if np.all(episodes_mask):
        return dataset

    episode_indices = np.arange(dataset.num_episodes)[episodes_mask]

    return select_dataset_episodes_by_index(dataset, episode_indices)


def filter_episodes(dataset, episodes_mask: np.ndarray):
    """
    Remove episodes from dataset.
    Args:
        dataset: LeRobotDataset to be filtered
        episodes_mask: 1D np.ndarray, dtype np.bool_, shape [num_episodes]. Values of False indicate
            that the episode should be removed from the dataset
    Returns:
        The filtered dataset
    """
    if dataset.num_episodes != len(episodes_mask):
        raise ValueError(
            f"Can't apply mask of size {len(episodes_mask)} to dataset with {dataset.num_episodes} episodes"
        )
    episode_indices = np.arange(dataset.num_episodes)
    filtered_dataset = select_dataset_episodes_by_index(dataset, episode_indices)

    # Re-wrap with original dataset class if needed
    return dataset.__class__(**filtered_dataset._init_args)


def select_dataset_episodes_by_index(dataset, episode_indices: np.ndarray):
    """
    Cut `dataset` by selecting only episodes in `episode_indices`
    Args:
        dataset: LeRobotDataset to be cut
        episode_indices: Indices of the episodes to include in the resulting dataset
    Returns:
        The truncated dataset
    """
    hf_dataset = select_hf_dataset_episodes(
        dataset.hf_dataset, dataset.episode_data_index, episode_indices
    )
    return dataset.replace_hf_dataset(hf_dataset)


def select_dataset_episodes_by_id(
    dataset, episode_ids: np.ndarray, invert: bool = False
):
    """
    Cut `dataset` by selecting episodes according to their ids. Checks the `episode_index` column,
    which corresponds to the episode id. NOTE: an index in `dataset.episode_data_index['from']` is
    not guaranteed to correspond to the value in `episode_index` column if the dataset has been filtered
    in any way!

    Args:
        dataset: LeRobotDataset to be cut
        episode_ids: IDs of the episodes to include in the resulting dataset
        invert: If True, select episodes which are not in `episode_ids`
    Returns:
        The truncated dataset
    """
    # Get the episode ids present in this dataset. NOTE: the index in `dataset.episode_data_index['from']` is not
    # guaranteed to correspond to the value in `episode_index` column if the dataset has been filtered in any way!
    dataset_episode_ids = dataset.np_column("episode_index")[
        dataset.episode_data_index["from"]
    ]

    # Find the corresponding episode *indices* that we should keep
    episode_indices = np.nonzero(
        np.isin(dataset_episode_ids, episode_ids, invert=invert)
    )[0]

    return select_dataset_episodes_by_index(dataset, episode_indices)


def select_hf_dataset_episodes(
    hf_dataset: datasets.Dataset,
    episode_data_index: Dict[str, np.ndarray],
    episode_indices: np.ndarray,
) -> datasets.Dataset:
    """
    Cut `hf_dataset` by selecting only episodes in `episode_indices`. Note that episode_data_index
    will no longer be valid after the cut.
    Args:
        hf_dataset: HF dataset holding the data
        episode_data_index: Episode data index in lerobot format
        episode_indices: Indices of the episodes to include in the resulting dataset
    Returns:
        The truncated dataset
    """
    num_episodes = len(episode_data_index["from"])

    if np.any(episode_indices >= num_episodes) or np.any(
        episode_indices <= -num_episodes
    ):
        out_of_bounds = np.concatenate(
            [
                episode_indices[episode_indices >= num_episodes],
                episode_indices[episode_indices <= -num_episodes],
            ]
        )
        raise ValueError(
            f"Episode indices {out_of_bounds} out of bounds for dataset with {num_episodes} episodes"
        )

    # Remove any repeating indices
    episode_indices = np.unique(episode_indices)

    # Make all indices non-negative
    episode_indices = (episode_indices + num_episodes) % num_episodes

    # All episodes selected, return the same dataset
    if len(episode_indices) == num_episodes:
        return hf_dataset

    indices = np_ranges(
        episode_data_index["from"][episode_indices],
        episode_data_index["to"][episode_indices],
    )
    hf_dataset = hf_dataset.select(indices).with_format(**hf_dataset.format)

    return hf_dataset


def add_target_joint_position(dataset, horizon=5, from_observation=True):

    # Load necessary arrays once
    episode_indices = np.array(dataset.np_column("episode_index"))
    if not from_observation:
        joint_pos = np.array(dataset.np_column("action.joint_position"))
        gripper_pos = np.array(dataset.np_column("action.gripper_position"))
    else:
        joint_pos = np.array(
            dataset.np_column("observation.robot_state.joint_positions")
        )
        gripper_pos = np.array(
            dataset.np_column("observation.robot_state.gripper_position")
        )
    num_samples = joint_pos.shape[0]
    joint_dim = joint_pos.shape[1]
    gripper_pos = gripper_pos.reshape(num_samples, -1)  # Ensure 2D
    gripper_dim = gripper_pos.shape[1]

    # Combine joint and gripper into a single array
    action_array = np.concatenate([joint_pos, gripper_pos], axis=1)

    # Prepare output
    targets = np.zeros(
        (num_samples, horizon, joint_dim + gripper_dim), dtype=action_array.dtype
    )

    # Process episode by episode for clean slicing
    unique_episodes = np.unique(episode_indices)
    for ep in tqdm(unique_episodes, desc="Computing targets"):
        idxs = np.where(episode_indices == ep)[0]
        ep_actions = action_array[idxs]
        ep_len = len(idxs)

        for i in range(ep_len):
            # Grab up to horizon future actions
            end = min(i + horizon + 1, ep_len)
            seq = ep_actions[i + 1 : end]

            # If not enough future, pad with last valid or current
            if len(seq) < horizon:
                if len(seq) > 0:
                    pad = np.repeat(seq[-1][np.newaxis, :], horizon - len(seq), axis=0)
                else:
                    pad = np.repeat(ep_actions[i][np.newaxis, :], horizon, axis=0)
                seq = np.concatenate([seq, pad], axis=0)

            targets[idxs[i]] = seq[:horizon]

    # Add to dataset
    dataset.hf_dataset = dataset.hf_dataset.add_column(
        "action.target.joint_position", targets.tolist()
    )


def add_target_joint_delta_gripper_delta(dataset, horizon=5, from_observation=True):
    # Load necessary arrays once
    episode_indices = np.array(dataset.np_column("episode_index"))
    if not from_observation:
        joint_pos = np.array(dataset.np_column("action.joint_position"))
        gripper_pos = np.array(dataset.np_column("action.gripper_position"))
    else:
        joint_pos = np.array(
            dataset.np_column("observation.robot_state.joint_positions")
        )
        gripper_pos = np.array(
            dataset.np_column("observation.robot_state.gripper_position")
        )

    num_samples = joint_pos.shape[0]
    joint_dim = joint_pos.shape[1]
    gripper_pos = gripper_pos.reshape(num_samples, -1)  # Ensure 2D
    gripper_dim = gripper_pos.shape[1]

    # Combine joint and gripper into a single array
    action_array = np.concatenate([joint_pos, gripper_pos], axis=1)

    # Prepare output for deltas
    targets = np.zeros(
        (num_samples, horizon, joint_dim + gripper_dim), dtype=action_array.dtype
    )

    # Process episode by episode for clean slicing
    unique_episodes = np.unique(episode_indices)
    for ep in tqdm(unique_episodes, desc="Computing target deltas"):
        idxs = np.where(episode_indices == ep)[0]
        ep_actions = action_array[idxs]
        ep_len = len(idxs)

        for i in range(ep_len):
            # Current action
            current_action = ep_actions[i]

            # Grab up to horizon future actions
            end = min(i + horizon + 1, ep_len)
            seq = ep_actions[i + 1 : end]

            # If not enough future, pad with last valid or current
            if len(seq) < horizon:
                if len(seq) > 0:
                    pad = np.repeat(seq[-1][np.newaxis, :], horizon - len(seq), axis=0)
                else:
                    pad = np.repeat(current_action[np.newaxis, :], horizon, axis=0)
                seq = np.concatenate([seq, pad], axis=0)

            # Compute deltas: future_action - current_action
            deltas = seq[:horizon] - current_action
            targets[idxs[i]] = deltas

    # Add to dataset
    dataset.hf_dataset.add_column(
        "action.target.joint_position_delta", targets.tolist()
    )


def add_target_joint_delta_gripper_abs(dataset, horizon=5, from_observation=True):
    import numpy as np
    from tqdm import tqdm

    # Load necessary arrays once
    episode_indices = np.array(dataset.np_column("episode_index"))
    if not from_observation:
        joint_pos = np.array(dataset.np_column("action.joint_position"))
        gripper_pos = np.array(dataset.np_column("action.gripper_position"))
    else:
        joint_pos = np.array(
            dataset.np_column("observation.robot_state.joint_positions")
        )
        gripper_pos = np.array(
            dataset.np_column("observation.robot_state.gripper_position")
        )

    num_samples = joint_pos.shape[0]
    joint_dim = joint_pos.shape[1]
    gripper_pos = gripper_pos.reshape(num_samples, -1)  # Ensure 2D
    gripper_dim = gripper_pos.shape[1]

    # Prepare output
    targets = np.zeros(
        (num_samples, horizon, joint_dim + gripper_dim), dtype=joint_pos.dtype
    )

    # Process episode by episode for clean slicing
    unique_episodes = np.unique(episode_indices)
    for ep in tqdm(
        unique_episodes, desc="Computing target joint deltas + absolute gripper"
    ):
        idxs = np.where(episode_indices == ep)[0]
        ep_joint_pos = joint_pos[idxs]
        ep_gripper_pos = gripper_pos[idxs]
        ep_len = len(idxs)

        for i in range(ep_len):
            current_joint = ep_joint_pos[i]
            # Grab future joint positions and gripper positions
            end = min(i + horizon + 1, ep_len)
            future_joints = ep_joint_pos[i + 1 : end]
            future_grippers = ep_gripper_pos[i + 1 : end]

            # Padding if needed
            if len(future_joints) < horizon:
                if len(future_joints) > 0:
                    joint_pad = np.repeat(
                        future_joints[-1][np.newaxis, :],
                        horizon - len(future_joints),
                        axis=0,
                    )
                    gripper_pad = np.repeat(
                        future_grippers[-1][np.newaxis, :],
                        horizon - len(future_grippers),
                        axis=0,
                    )
                else:
                    joint_pad = np.repeat(current_joint[np.newaxis, :], horizon, axis=0)
                    gripper_pad = np.repeat(
                        ep_gripper_pos[i][np.newaxis, :], horizon, axis=0
                    )

                future_joints = np.concatenate([future_joints, joint_pad], axis=0)
                future_grippers = np.concatenate([future_grippers, gripper_pad], axis=0)

            # Compute joint deltas and use absolute gripper
            # joint_deltas = future_joints[:horizon] - current_joint
            joints_sequence = np.vstack([current_joint, future_joints[:horizon]])
            joint_deltas = np.diff(joints_sequence, axis=0)
            gripper_absolutes = future_grippers[:horizon]

            # Combine and assign
            targets[idxs[i]] = np.concatenate([joint_deltas, gripper_absolutes], axis=1)

    # Add to dataset
    dataset.hf_dataset.add_column(
        "action.target.joint_position_delta", targets.tolist()
    )


def add_target_joint_delta_gripper_abs_mask(dataset, horizon=5, from_observation=True):

    # Load necessary arrays once
    episode_indices = np.array(dataset.np_column("episode_index"))
    if not from_observation:
        joint_pos = np.array(dataset.np_column("action.joint_position"))
        gripper_pos = np.array(dataset.np_column("action.gripper_position"))
    else:
        joint_pos = np.array(
            dataset.np_column("observation.robot_state.joint_positions")
        )
        gripper_pos = np.array(
            dataset.np_column("observation.robot_state.gripper_position")
        )

    num_samples = joint_pos.shape[0]
    joint_dim = joint_pos.shape[1]
    gripper_pos = gripper_pos.reshape(num_samples, -1)  # Ensure 2D
    gripper_dim = gripper_pos.shape[1]

    # Prepare output
    targets = np.zeros(
        (num_samples, horizon, joint_dim + gripper_dim), dtype=joint_pos.dtype
    )
    target_mask = np.zeros((num_samples, horizon), dtype=np.int8)

    # Process episode by episode for clean slicing
    unique_episodes = np.unique(episode_indices)
    for ep in tqdm(
        unique_episodes, desc="Computing target joint deltas + absolute gripper"
    ):
        idxs = np.where(episode_indices == ep)[0]
        ep_joint_pos = joint_pos[idxs]
        ep_gripper_pos = gripper_pos[idxs]
        ep_len = len(idxs)

        for i in range(ep_len):
            current_joint = ep_joint_pos[i]
            end = min(i + horizon + 1, ep_len)
            future_joints = ep_joint_pos[i + 1 : end]
            future_grippers = ep_gripper_pos[i + 1 : end]
            valid_len = len(future_joints)

            # Padding if needed
            if valid_len < horizon:
                if valid_len > 0:
                    joint_pad = np.repeat(
                        future_joints[-1][np.newaxis, :], horizon - valid_len, axis=0
                    )
                    gripper_pad = np.repeat(
                        future_grippers[-1][np.newaxis, :], horizon - valid_len, axis=0
                    )
                else:
                    joint_pad = np.repeat(current_joint[np.newaxis, :], horizon, axis=0)
                    gripper_pad = np.repeat(
                        ep_gripper_pos[i][np.newaxis, :], horizon, axis=0
                    )

                future_joints = np.concatenate([future_joints, joint_pad], axis=0)
                future_grippers = np.concatenate([future_grippers, gripper_pad], axis=0)

            # Compute joint deltas and use absolute gripper
            # joint_deltas = future_joints[:horizon] - current_joint
            joints_sequence = np.vstack([current_joint, future_joints[:horizon]])
            joint_deltas = np.diff(joints_sequence, axis=0)
            gripper_absolutes = future_grippers[:horizon]

            # Combine and assign
            targets[idxs[i]] = np.concatenate([joint_deltas, gripper_absolutes], axis=1)
            target_mask[idxs[i], :valid_len] = 1  # mark valid steps

    # Add to dataset
    dataset.hf_dataset = dataset.hf_dataset.add_column(
        "action.target.joint_position_delta", targets.tolist()
    )
    dataset.hf_dataset = dataset.hf_dataset.add_column(
        "action.target.joint_position_delta_mask", target_mask.tolist()
    )


def add_target_cartesian_delta_gripper_abs_mask(
    dataset, horizon=5, from_observation=True
):
    # Load necessary arrays once
    episode_indices = np.array(dataset.np_column("episode_index"))
    if not from_observation:
        cartesian_pose = np.array(dataset.np_column("action.cartesian_position"))
        gripper_pos = np.array(dataset.np_column("action.gripper_position"))
    else:
        cartesian_pose = np.array(
            dataset.np_column("observation.robot_state.cartesian_position")
        )
        gripper_pos = np.array(
            dataset.np_column("observation.robot_state.gripper_position")
        )

    num_samples = cartesian_pose.shape[0]
    state_dim = cartesian_pose.shape[1]
    gripper_pos = gripper_pos.reshape(num_samples, -1)  # Ensure 2D
    gripper_dim = gripper_pos.shape[1]

    # Prepare output
    targets = np.zeros(
        (num_samples, horizon, state_dim + gripper_dim), dtype=cartesian_pose.dtype
    )
    target_mask = np.zeros((num_samples, horizon), dtype=np.int8)

    # Process episode by episode for clean slicing
    unique_episodes = np.unique(episode_indices)
    for ep in tqdm(
        unique_episodes, desc="Computing target joint deltas + absolute gripper"
    ):
        idxs = np.where(episode_indices == ep)[0]
        ep_cartesian_pose = cartesian_pose[idxs]
        ep_gripper_pos = gripper_pos[idxs]
        ep_len = len(idxs)

        for i in range(ep_len):
            current_cartesian = ep_cartesian_pose[i]
            end = min(i + horizon + 1, ep_len)
            future_cartesian = ep_cartesian_pose[i + 1 : end]
            future_grippers = ep_gripper_pos[i + 1 : end]
            valid_len = len(future_cartesian)

            # Padding if needed
            if valid_len < horizon:
                if valid_len > 0:
                    cartesian_pad = np.repeat(
                        future_cartesian[-1][np.newaxis, :], horizon - valid_len, axis=0
                    )
                    gripper_pad = np.repeat(
                        future_grippers[-1][np.newaxis, :], horizon - valid_len, axis=0
                    )
                else:
                    cartesian_pad = np.repeat(
                        current_cartesian[np.newaxis, :], horizon, axis=0
                    )
                    gripper_pad = np.repeat(
                        ep_gripper_pos[i][np.newaxis, :], horizon, axis=0
                    )

                future_cartesian = np.concatenate(
                    [future_cartesian, cartesian_pad], axis=0
                )
                future_grippers = np.concatenate([future_grippers, gripper_pad], axis=0)

            # Compute joint deltas and use absolute gripper
            # joint_deltas = future_joints[:horizon] - current_joint
            cartesian_sequence = np.vstack(
                [current_cartesian, future_cartesian[:horizon]]
            )
            cartesian_deltas = np.diff(cartesian_sequence, axis=0)
            gripper_absolutes = future_grippers[:horizon]

            # Combine and assign
            targets[idxs[i]] = np.concatenate(
                [cartesian_deltas, gripper_absolutes], axis=1
            )
            target_mask[idxs[i], :valid_len] = 1  # mark valid steps

    # Add to dataset
    dataset.hf_dataset = dataset.hf_dataset.add_column(
        "action.target.cartesian_position_delta", targets.tolist()
    )
    dataset.hf_dataset = dataset.hf_dataset.add_column(
        "action.target.cartesian_position_delta_mask", target_mask.tolist()
    )


if __name__ == "__main__":
    # Test
    pass
