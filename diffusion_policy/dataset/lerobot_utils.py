import functools
import json
import os
import re
from typing import Any, Dict, Tuple

import datasets
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def strip_split(split: str) -> str:
    if "[" in split:
        split, _ = split.split("[", 1)
    return split


def _slice_value_to_episode_index(value: str, num_episodes: int, side: str) -> int:
    """
    Take a slice value and convert it to an episode index. Follows python slicing convention:
        values before `:` are including, values after `:` are excluding
    Args:
        value: Slice value to use. Can be empty string, contain percentage or absolute number.
            Only non-negative numbers
        episode_data_index: Episode data index for the HF dataset, containing 'from' and 'to' keys and the
            boundaries of all episodes
        side: 'left' or 'right'. Which side of the slice operator this value comes from. Makes a difference
            only when value corresponds to 0% or 100%.
    Returns:
        Non-negative episode index
    """
    assert side in ["left", "right"]

    if value == "":
        if side == "left":
            return 0
        return num_episodes

    if "%" in value:
        percent = float(value.replace("%", ""))
        assert 100 >= percent >= 0, percent
        episode_index = round(percent / 100 * num_episodes)
    else:
        if not float(value).is_integer():
            raise ValueError(f"Absolute indices can only be integers, but got {value}")
        episode_index = int(value)
        if not num_episodes >= episode_index >= 0:
            raise ValueError(
                f"Index {episode_index} out of bounds for dataset containing {num_episodes} episodes"
            )

    episode_index = np.clip(episode_index, 0, num_episodes)

    # If the slice is on the left and corresponds to 0% of the dataset, raise error
    if (side == "right" and episode_index == 0) or (
        side == "left" and episode_index == num_episodes
    ):
        raise ValueError("Slice results in empty dataset")

    return episode_index


def episode_range_from_split(dataset_path: str, split: str) -> Tuple[int, int]:
    if "[" in split:
        split, slices = split.split("[", 1)
        slices = f"[{slices}"
    else:
        slices = ""

    episode_data_index: Dict[str, np.ndarray] = load_episode_data_index(
        dataset_path, split
    )
    num_episodes = len(episode_data_index["from"])

    if slices:
        # Regex that extracts start and end slice as strings. Includes % if present
        matches = re.search(r"\[((?:\d*\.?\d+%?)?):?((?:\d*\.?\d+%?)?)\]", slices)
        if matches is None:
            raise ValueError(f"{split}{slices} is not a valid dataset slice")

        match_low, match_high = matches.groups()

        if match_low != "" and match_high != "":
            if ("%" in match_low) != ("%" in match_high):
                raise ValueError(
                    f"Both start and end positions must be in % or not, but got {match_low}, {match_high}"
                )
            if float(match_low.replace("%", "")) >= float(match_high.replace("%", "")):
                raise ValueError(
                    f"Start position {match_low} must be < end position {match_high}"
                )

        episode_low = _slice_value_to_episode_index(
            match_low, num_episodes, side="left"
        )
        episode_high = _slice_value_to_episode_index(
            match_high, num_episodes, side="right"
        )
    else:
        episode_low = 0
        episode_high = num_episodes

    return episode_low, episode_high


def load_hf_dataset(dataset_path: str, split: str) -> datasets.Dataset:
    """
    Load the raw HF dataset used to store the robotics data
    Args:
        dataset_path: Path on disk or HF hub id
        split: One of 'train' or 'test'. Can also use absolute or percentage slices:
            - Absolute: 'train[123:450]', train[:4000]
            - Percent: 'train[:75%]', 'train[25%:75%]'. Fractional percents are also supported
            NOTE: Slices are executed on episode level rather than on individual transitions level. That
            means that the returned dataset always contains complete episodes. Absolute indices must be in
            the range of number of episodes
    Returns:
        The sliced dataset
    """
    split, slice_split = strip_split(split), split

    if os.path.exists(dataset_path):
        assert os.path.exists(
            split_path := os.path.join(dataset_path, split)
        ), f"{split_path} doesn't exist"
        hf_dataset = datasets.load_from_disk(split_path)
    else:
        hf_dataset = datasets.load_dataset(dataset_path, split=split)

    # Apply slicing
    if split != slice_split:
        split, slices = slice_split.split("[", 1)
        slices = f"[{slices}"

        episode_low, episode_high = episode_range_from_split(dataset_path, slice_split)

        episode_data_index: Dict[str, np.ndarray] = load_episode_data_index(
            dataset_path, slice_split
        )

        indices = np.arange(
            episode_data_index["from"][episode_low],
            episode_data_index["to"][episode_high - 1],
        )
        hf_dataset = hf_dataset.select(indices)

    return hf_dataset


def load_episode_data_index(dataset_path: str, split: str) -> Dict[str, np.ndarray]:
    """
    Loads episode data index for a given dataset split

    Args:
        dataset_path: Path to the root of the LeRobotDataset
        split: One of 'train' or 'test'
    Returns:
        Dict with keys 'from' and 'to' containing the range of indices for each episode
    """

    split = strip_split(split)

    if os.path.exists(dataset_path):
        path = os.path.join(
            dataset_path, "meta_data", split, "episode_data_index.safetensors"
        )
    else:
        path = hf_hub_download(
            dataset_path,
            f"meta_data/{split}/episode_data_index.safetensors",
            repo_type="dataset",
        )

    assert os.path.exists(path), f"File {path} doesn't exist"

    episode_data_index = load_file(path)

    return {
        "from": episode_data_index["from"].numpy(),
        "to": episode_data_index["to"].numpy(),
    }


def load_info(dataset_path: str) -> Dict[str, Any]:
    """
    info contains useful information regarding the dataset that are not stored elsewhere
    """
    if os.path.exists(dataset_path):
        path = os.path.join(dataset_path, "meta_data", "info.json")
    else:
        path = hf_hub_download(dataset_path, "meta_data/info.json", repo_type="dataset")

    with open(path) as f:
        info = json.load(f)
    return info


def load_episode_metadata(dataset_path: str, split: str) -> datasets.Dataset:
    """
    Loads episode metadata for a given dataset split. Each row in this dataset corresponds to the metadata
    for an episode in the core dataset. The size of this metadata dataset is the number of episodes and never
    emtpy, since the 'episode_index' field is always present. If split contains slicing, it will be applied
    to the resulting metadata dataset such that it matches the slicing applied to the core dataset.

    Args:
        dataset_path: Path to the root of the LeRobotDataset (not the metadata dataset)
        split: One of 'train' or 'test' and optionally containing slicing
    Returns:
        The loaded and optionally filtered dataset of episode metadata
    """

    load_path = os.path.join(
        dataset_path, "meta_data", strip_split(split), "episode_metadata"
    )

    if os.path.exists(dataset_path) and os.path.exists(load_path):
        episode_metadata_dataset = datasets.load_from_disk(load_path)
    else:
        return None
        episode_metadata_dataset = datasets.load_dataset(load_path)

    # Filter the episode metadata dataset
    if split != strip_split(split) and len(episode_metadata_dataset) > 0:
        episode_low, episode_high = episode_range_from_split(dataset_path, split)
        episode_indices = np.arange(episode_low, episode_high)
        episode_metadata_dataset = episode_metadata_dataset.select(episode_indices)

    return episode_metadata_dataset


def filter_episode_metadata(
    hf_dataset: datasets.Dataset, episode_metadata: datasets.Dataset
) -> datasets.Dataset:
    """
    Filter episode_metadata such that it contains only episodes present in hf_dataset
    """
    episode_ids = np.unique(np_column(hf_dataset, "episode_index"))
    metadata_episode_ids = np_column(episode_metadata, "episode_index")

    mask = np.isin(metadata_episode_ids, episode_ids)
    indices = np.arange(len(metadata_episode_ids))[mask]

    episode_metadata = episode_metadata.select(indices)

    return episode_metadata


@functools.lru_cache(maxsize=None)  # pylint:disable=cache-max-size-none
def np_column(hf_dataset: datasets.Dataset, column_name: str) -> np.ndarray:
    # Get the np column directly from the PyArrow table backing the dataset
    column = hf_dataset.data.column(column_name).to_numpy()

    # If the dataset is a subset of the PyArrow table, filter the output of the column
    if hf_dataset._indices is not None:  # noqa: SLF001
        indices = hf_dataset._indices.column("indices").to_numpy()  # noqa: SLF001
        column = column[indices]
    # If the column is array of arrays, make it a single multidimensional array
    if column.dtype == np.dtype("O"):
        column = np.vstack(column)

    assert len(column) == len(hf_dataset), f"{len(column)} != {len(hf_dataset)}"
    return column


def add_columns(
    hf_dataset: datasets.Dataset, columns: Dict[str, np.ndarray]
) -> datasets.Dataset:
    """
    Add new columns to hf_dataset in an optimized way. Specifically, whenever a `datasets.Dataset` has
    been 'indexed', e.g. by calling `datasets.Dataset.select`, it creates an internal index mapping. If
    one tries to add a new column to the 'indexed' dataset, the operation can become extremely slow for
    big datasets since `datasets.Dataset` tries to flatten the dataset internally. Instead, here we
    bypass this by accessing the pyarrow table backing the `datasets.Dataset` and adding the new columns
    directly to it
    Args:
        hf_dataset: Dataset
        columns: Dict of column names pointing to arrays containing the column values. Each array must
            be the same length as `hf_dataset`
    Returns:
        datasets.Dataset with the new columns added
    """

    # If `hf_dataset` hasn't been 'indexed' yet, use simply create a new datasets.Dataset and
    # concatenate the dataset objects
    indices_table: datasets.table.Table = getattr(hf_dataset, "_indices", None)
    if indices_table is None:
        columns_dataset = datasets.Dataset.from_dict(columns)
        return datasets.concatenate_datasets([hf_dataset, columns_dataset], axis=1)

    # `hf_dataset` has been 'indexed' -> we need to be smart and add to the pyarrow table directly
    hf_dataset_table: datasets.table.Table = hf_dataset.data
    indices = indices_table.column("indices").to_numpy()

    # Get the size of the original 'unindexed' dataset contained in `hf_dataset_table`
    expaneded_size = len(hf_dataset_table)
    assert len(indices) != expaneded_size  # Might be unnecessary

    # Expand the new columns to the size of the original table. The expanded values are zeros
    # and would be hidden, i.e. would never be 'visible' when using `datasets.Dataset` API
    expanded_columns = {}
    for column_name, column in columns.items():
        if len(indices) != len(column):
            raise ValueError(
                f"Column {column_name} with size {len(column)} != {len(indices)}"
            )

        expanded_column = np.zeros(
            (expaneded_size,) + column.shape[1:], dtype=column.dtype
        )
        expanded_column[indices] = column
        expanded_columns[column_name] = expanded_column

    # Create a table containing the expanded columns
    columns_dataset = datasets.Dataset.from_dict(expanded_columns)
    expanded_columns_table: datasets.table.Table = columns_dataset.data

    assert len(hf_dataset_table) == len(expanded_columns_table)

    info = datasets.DatasetInfo(
        # IMPORTANT: Need to pass Features, otherwise Images end up being decoded
        features=datasets.Features({**hf_dataset.features, **columns_dataset.features}),
    )
    fingerprint = datasets.fingerprint.update_fingerprint(
        "".join(
            [hf_dataset._fingerprint, columns_dataset._fingerprint]
        ),  # noqa: SLF001
        add_columns,
        {"info": info},
    )

    # Concatenate the tables for `hf_dataset` and `columns_dataset` and directly apply the indexing
    # by providing `indices_table`
    hf_dataset = datasets.Dataset(
        arrow_table=datasets.table.concat_tables(
            [hf_dataset_table, expanded_columns_table], axis=1
        ),
        indices_table=indices_table,
        info=datasets.DatasetInfo(
            # IMPORTANT: Need to pass Features, otherwise Images end up being decoded
            features=datasets.Features(
                {**hf_dataset.features, **columns_dataset.features}
            ),
        ),
        # IMPORTANT: Need to pass fingerprint, otherwise ends up computing a hash of the data inside
        # the table, which means loading the enitre dataset and likely running OOM
        fingerprint=fingerprint,
    )

    assert len(hf_dataset) == len(indices_table)

    return hf_dataset
