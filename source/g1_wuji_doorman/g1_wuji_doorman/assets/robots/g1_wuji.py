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
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,

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