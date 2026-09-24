"""Validate USD/PhysX interfaces before feeding them into the task."""
from pathlib import PurePosixPath

import torch


def contact_filter_indices(sensor_paths, filter_paths, links):
    """Return per-environment columns in canonical finger/link order, or fail loudly."""
    if len(sensor_paths) != len(filter_paths):
        raise RuntimeError("Contact sensor/filter row counts differ")
    indices = []
    for sensor_path, row in zip(sensor_paths, filter_paths):
        env_path = PurePosixPath(sensor_path).parent.parent
        expected = [str(env_path / "Robot" / name) for name in links]
        if len(row) != len(expected) or len(set(row)) != len(row) or set(row) != set(expected):
            raise RuntimeError(f"Unexpected contact filters for {sensor_path}: {row}; expected {expected}")
        indices.append([row.index(path) for path in expected])
    return indices


def door_info_from_metadata(metadata_rows, device):
    """Original DoorMan's eight fields, read from the generated asset rather than constants."""
    rows = []
    for metadata in metadata_rows:
        lr, io = metadata["doorOpenLR"], metadata["doorOpenIO"]
        if lr != -1 or io != -1:
            raise ValueError("This stationary left-hand MDP supports right/out doors only; calibrate frames before changing direction")
        rows.append([metadata["doorWidth"], metadata["doorHeight"], metadata["doorHandleHeight"],
                     metadata["doorHandleWidth"], metadata["doorWeight"] / 100.0, lr, 1.0 - lr, io])
    values = torch.tensor(rows, device=device, dtype=torch.float32)
    if not torch.isfinite(values).all() or not (values[:, :5] > 0).all():
        raise ValueError("Door metadata must contain finite positive dimensions and mass")
    return values
