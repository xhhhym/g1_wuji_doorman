# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verified Wuji hand-pose targets in physical joint order."""


WUJI_HAND_REST_POSE = (
    0.12, 0.05, 0.15, 0.10,
    0.10, 0.00, 0.18, 0.12,
    0.10, 0.00, 0.18, 0.12,
    0.10, 0.00, 0.18, 0.12,
    0.10, 0.00, 0.18, 0.12,
)

WUJI_HAND_OPEN_POSE = (
    0.06, 0.00, 0.05, 0.03,
    0.02, 0.00, 0.04, 0.03,
    0.02, 0.00, 0.04, 0.03,
    0.02, 0.00, 0.04, 0.03,
    0.02, 0.00, 0.04, 0.03,
)

WUJI_HAND_GRASP_POSE = (
    0.80, 0.45, 0.80, 0.80,
    0.80, 0.18, 0.80, 0.80,
    0.80, 0.18, 0.80, 0.80,
    0.80, 0.18, 0.80, 0.80,
    0.80, 0.18, 0.80, 0.80,
)

_WUJI_HAND_POSES = {
    "rest": WUJI_HAND_REST_POSE,
    "open": WUJI_HAND_OPEN_POSE,
    "grasp": WUJI_HAND_GRASP_POSE,
}

for _pose_name, _pose in _WUJI_HAND_POSES.items():
    if len(_pose) != 20:
        raise RuntimeError(
            f"Wuji {_pose_name} pose must contain 20 joint targets, "
            f"got {len(_pose)}."
        )
