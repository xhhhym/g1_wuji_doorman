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

G1_WUJI_FREE_BASE_CFG = G1_WUJI_CFG.copy()

# The robot must respond to gravity and balance itself through HOMIE.
G1_WUJI_FREE_BASE_CFG.spawn.rigid_props.disable_gravity = False
G1_WUJI_FREE_BASE_CFG.spawn.articulation_props.fix_root_link = False

# Match the PD gains used when training the HOMIE checkpoints.
G1_WUJI_FREE_BASE_CFG.actuators = {
    "body": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
            "waist_.*_joint",
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_.*_joint",
        ],
        stiffness={
            ".*_hip_.*_joint": 150.0,
            ".*_knee_joint": 200.0,
            ".*_ankle_.*_joint": 40.0,
            "waist_.*_joint": 250.0,
            ".*_shoulder_pitch_joint": 100.0,
            ".*_shoulder_roll_joint": 100.0,
            ".*_shoulder_yaw_joint": 40.0,
            ".*_elbow_joint": 40.0,
            ".*_wrist_.*_joint": 20.0,
        },
        damping={
            ".*_hip_.*_joint": 2.0,
            ".*_knee_joint": 4.0,
            ".*_ankle_.*_joint": 2.0,
            "waist_.*_joint": 5.0,
            ".*_shoulder_pitch_joint": 5.0,
            ".*_shoulder_roll_joint": 5.0,
            ".*_shoulder_yaw_joint": 2.0,
            ".*_elbow_joint": 2.0,
            ".*_wrist_.*_joint": 2.0,
        },

        effort_limit_sim={
            ".*_hip_pitch_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_yaw_joint": 88.0,
            ".*_knee_joint": 139.0,
            ".*_ankle_pitch_joint": 35.0,
            ".*_ankle_roll_joint": 35.0,
            "waist_yaw_joint": 88.0,
            "waist_roll_joint": 35.0,
            "waist_pitch_joint": 35.0,
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_hip_pitch_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_yaw_joint": 32.0,
            ".*_knee_joint": 20.0,
            ".*_ankle_pitch_joint": 30.0,
            ".*_ankle_roll_joint": 30.0,
            "waist_yaw_joint": 32.0,
            "waist_roll_joint": 30.0,
            "waist_pitch_joint": 30.0,
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        armature={
            ".*_hip_pitch_joint": 0.03053256012,
            ".*_hip_roll_joint": 0.075305775,
            ".*_hip_yaw_joint": 0.03053256012,
            ".*_knee_joint": 0.075305775,
            ".*_ankle_pitch_joint": 0.02165835,
            ".*_ankle_roll_joint": 0.02165835,
            "waist_yaw_joint": 0.03053256012,
            "waist_roll_joint": 0.02165835,
            "waist_pitch_joint": 0.02165835,
            ".*_shoulder_pitch_joint": 0.010829175,
            ".*_shoulder_roll_joint": 0.010829175,
            ".*_shoulder_yaw_joint": 0.010829175,
            ".*_elbow_joint": 0.010829175,
            ".*_wrist_roll_joint": 0.010829175,
            ".*_wrist_pitch_joint": 0.01275,
            ".*_wrist_yaw_joint": 0.01275,
        },
        friction=0.0,
    ),

        "hands": ImplicitActuatorCfg(
        joint_names_expr=[".*_finger[1-5]_joint[1-4]"],
        effort_limit_sim=30.0,
        velocity_limit_sim=10.0,
        stiffness=10.0,
        damping=0.2,
        armature=0.003,
        friction=0.0,
    ),
}

# Standing pose expected by the HOMIE checkpoint.
G1_WUJI_FREE_BASE_CFG.init_state.joint_pos.update(
    {
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
    }
)
