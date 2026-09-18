# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct Wuji joint-target hand controller."""

from __future__ import annotations

import torch

from .base import HandController


class JointTargetHandController(HandController):
    """Use a 20-dimensional physical joint-position command directly."""

    command_dim = HandController.target_dim

    def compute_joint_targets(self, command: torch.Tensor) -> torch.Tensor:
        self._validate_command(command)
        return command.clone()
