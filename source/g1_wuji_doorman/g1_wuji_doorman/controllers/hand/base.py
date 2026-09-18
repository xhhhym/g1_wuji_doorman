# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent hand-control interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class HandController(ABC):
    """Convert a backend-specific command into 20 Wuji joint targets."""

    target_dim = 20

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)

    @property
    @abstractmethod
    def command_dim(self) -> int:
        """Number of command dimensions consumed by this backend."""
        raise NotImplementedError

    def _validate_command(self, command: torch.Tensor) -> None:
        expected_shape = (self.num_envs, self.command_dim)
        if command.shape != expected_shape:
            raise ValueError(
                f"Hand command must have shape {expected_shape}, "
                f"got {tuple(command.shape)}."
            )
        if command.device != self.device:
            raise ValueError(
                f"Hand command is on {command.device}, expected {self.device}."
            )
        if not torch.isfinite(command).all():
            raise ValueError("Hand command contains NaN or Inf.")

    @abstractmethod
    def compute_joint_targets(self, command: torch.Tensor) -> torch.Tensor:
        """Return joint targets with shape ``(num_envs, 20)``."""
        raise NotImplementedError

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Reset backend state; stateless backends require no work."""
        del env_ids
