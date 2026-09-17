"""Isaac Lab configuration for the G1 + Wuji hands robot."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# -----------------------------------------------------------------------------
# Asset paths
# -----------------------------------------------------------------------------

G1_WUJI_ASSET_DIR = Path(__file__).resolve().parent / "g1_wuji"
G1_WUJI_USD_PATH = G1_WUJI_ASSET_DIR / "g1_wuji_no_merge.usd"


if not G1_WUJI_USD_PATH.is_file():
    raise FileNotFoundError(
        f"G1+Wuji USD file was not found: {G1_WUJI_USD_PATH}"
    )


# -----------------------------------------------------------------------------
# Robot configuration
# -----------------------------------------------------------------------------

G1_WUJI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(G1_WUJI_USD_PATH),
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Temporary settings for the first visual inspection.
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Temporarily fix the floating base while inspecting the asset.
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),

        # Isaac Lab quaternion order: (w, x, y, z).
        rot=(1.0, 0.0, 0.0, 0.0),

        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),

        # Initial pose taken from the existing Doorman G1+Wuji configuration.
        # Joints not matched below remain at zero.
        joint_pos={
            # Legs
            ".*_hip_pitch_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,

            # Thumb / finger 1
            ".*_finger1_joint1": 0.12,
            ".*_finger1_joint2": 0.05,
            ".*_finger1_joint3": 0.15,
            ".*_finger1_joint4": 0.10,

            # Fingers 2–5
            ".*_finger[2-5]_joint1": 0.10,
            ".*_finger[2-5]_joint2": 0.0,
            ".*_finger[2-5]_joint3": 0.18,
            ".*_finger[2-5]_joint4": 0.12,
        },

        joint_vel={
            ".*": 0.0,
        },
    ),

    # For this asset-inspection stage, configure all 69 joints and use the
    # joint-drive stiffness and damping authored in the USD.
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },

    soft_joint_pos_limit_factor=0.9,
)

# -----------------------------------------------------------------------------
# Free-base configuration for HOMIE standing and door-opening training
# -----------------------------------------------------------------------------

# These names and parameter arrays are copied directly from the already
# validated GR00T-VisualSim2Real/viz_wuji_homie_standing.py integration.
DOORMAN_BODY_DOF_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
DOORMAN_WUJI_DOF_NAMES = tuple(
    f"{side}_finger{finger}_joint{joint}"
    for side in ("left", "right")
    for finger in range(1, 6)
    for joint in range(1, 5)
)
DOORMAN_ALL_DOF_NAMES = DOORMAN_BODY_DOF_NAMES + DOORMAN_WUJI_DOF_NAMES

_BODY_EFFORT = (
    88, 139, 88, 139, 35, 35, 88, 139, 88, 139, 35, 35, 88, 35, 35,
    25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5,
)
_BODY_VELOCITY = (
    32, 20, 32, 20, 30, 30, 32, 20, 32, 20, 30, 30, 32, 30, 30,
    37, 37, 37, 37, 37, 22, 22, 37, 37, 37, 37, 37, 22, 22,
)
_BODY_ARMATURE = (
    0.01017752004, 0.025101925, 0.01017752004, 0.025101925, 0.00721945, 0.00721945,
    0.01017752004, 0.025101925, 0.01017752004, 0.025101925, 0.00721945, 0.00721945,
    0.01017752004, 0.00721945, 0.00721945,
    0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.00425, 0.00425,
    0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.00425, 0.00425,
)
_BODY_STIFFNESS = (
    150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250,
    100, 100, 40, 40, 20, 20, 20, 100, 100, 40, 40, 20, 20, 20,
)
_BODY_DAMPING = (
    2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
    5, 5, 2, 2, 2, 2, 2, 5, 5, 2, 2, 2, 2, 2,
)

_effort: dict[str, float] = dict(zip(DOORMAN_BODY_DOF_NAMES, _BODY_EFFORT))
_velocity: dict[str, float] = dict(zip(DOORMAN_BODY_DOF_NAMES, _BODY_VELOCITY))
_armature: dict[str, float] = dict(
    zip(DOORMAN_BODY_DOF_NAMES, (value * 3.0 for value in _BODY_ARMATURE))
)
_stiffness: dict[str, float] = dict(
    zip(DOORMAN_BODY_DOF_NAMES, _BODY_STIFFNESS)
)
_damping: dict[str, float] = dict(zip(DOORMAN_BODY_DOF_NAMES, _BODY_DAMPING))
for _joint_name in DOORMAN_WUJI_DOF_NAMES:
    _effort[_joint_name] = 30.0
    _velocity[_joint_name] = 10.0
    _armature[_joint_name] = 0.003
    _stiffness[_joint_name] = 10.0
    _damping[_joint_name] = 0.2

_free_base_default_pos: dict[str, float] = {
    name: 0.0 for name in DOORMAN_ALL_DOF_NAMES
}
_free_base_default_pos.update(
    {
        "left_hip_pitch_joint": -0.1,
        "left_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "right_hip_pitch_joint": -0.1,
        "right_knee_joint": 0.3,
        "right_ankle_pitch_joint": -0.2,
        "left_finger1_joint1": 0.05,
        "right_finger1_joint1": 0.05,
    }
)

G1_WUJI_FREE_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(G1_WUJI_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos=_free_base_default_pos,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=list(DOORMAN_ALL_DOF_NAMES),
            effort_limit_sim=_effort,
            velocity_limit_sim=_velocity,
            stiffness=_stiffness,
            damping=_damping,
            armature=_armature,
            friction={name: 0.0 for name in DOORMAN_ALL_DOF_NAMES},
        ),
    },
)
