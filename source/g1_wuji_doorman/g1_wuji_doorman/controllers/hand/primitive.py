# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-dimensional open-to-grasp Wuji hand controller."""

from __future__ import annotations

import torch

from .base import HandController
from .poses import WUJI_HAND_GRASP_POSE, WUJI_HAND_OPEN_POSE


class PrimitiveHandController(HandController):
    """Map one normalized closure command to 20 joint targets."""

    command_dim = 1

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        super().__init__(num_envs=num_envs, device=device)
        self._open_pose = torch.tensor(
            WUJI_HAND_OPEN_POSE,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._grasp_pose = torch.tensor(
            WUJI_HAND_GRASP_POSE,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def compute_joint_targets(self, command: torch.Tensor) -> torch.Tensor:
        """Interpolate ``-1``/``0``/``1`` to open/mid/grasp targets."""
        self._validate_command(command)
        alpha = (command.clamp(-1.0, 1.0) + 1.0) * 0.5
        return torch.lerp(self._open_pose, self._grasp_pose, alpha)
