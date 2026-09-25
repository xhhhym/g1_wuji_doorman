"""Configuration constants for the frozen HOMIE standing controller."""

from dataclasses import dataclass, field
from pathlib import Path
import os


HOMIE_OBS_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# HOMIE outputs actions only for both legs and the waist.
HOMIE_ACTION_JOINT_ORDER = HOMIE_OBS_JOINT_ORDER[:15]

HOMIE_DEFAULT_JOINT_POS = (
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def default_homie_checkpoint() -> str:
    """Resolve the bundled checkpoint, or use ``G1_HOMIE_CHECKPOINT``."""
    override = os.environ.get("G1_HOMIE_CHECKPOINT")
    if override:
        return str(Path(override).expanduser())
    return str(Path(__file__).resolve().parents[2] / "models" / "model_stand.pt")


@dataclass(frozen=True)
class HomieControllerCfg:
    """Frozen parameters belonging to the HOMIE checkpoint contract."""

    checkpoint_path: str = field(default_factory=default_homie_checkpoint)
    action_scale: float = 0.25
    default_height: float = 0.74
    frame_dim: int = 86
    history_length: int = 6
    num_actions: int = 15
