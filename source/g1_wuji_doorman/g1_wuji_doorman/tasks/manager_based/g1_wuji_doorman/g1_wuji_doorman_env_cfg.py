# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from g1_wuji_doorman.assets.door import DoorSpawnerCfg, spawn_door

from . import mdp

##
# Pre-defined configs
##

from g1_wuji_doorman.assets.robots import G1_WUJI_FREE_BASE_CFG


##
# Door Generation
##

DOOR_SPAWNER_CFG = DoorSpawnerCfg(
    func=spawn_door,

    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        fix_root_link=True,
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    ),

    build_latch=True,

    # 暂时关闭无关功能
    add_walls=False,
    add_floors=False,
    add_lights=False,
    add_ceiling=False,
    randomize_material=False,
    dynamic_material_randomization=False,

    # 固定开门方向
    door_open_lr=["right"],
    door_open_io=["out"],

    # 匹配 XML 门
    rand_door_width=0.85,
    rand_door_height=2.0,
    rand_door_handle_height=0.8,
    rand_door_handle_width=0.23,
    rand_door_weight=10.0,

    # 固定门框宽度
    wall_minimum_clearance_fblr=(3.0, 3.0, 0.505, 0.505),
    wall_maximum_clearance_fblr=(3.0, 3.0, 0.505, 0.505),

    # 匹配 XML 把手
    rand_axle_length=0.20,
    rand_handle_length=0.20,
    rand_handle_radius=0.02,
    rand_spawn_hook=False,

    # 固定动力学
    rand_hinge_drive_max_force=10.0,
    rand_hinge_drive_stiffness=0.0,
    rand_handle_drive_max_force=2.0,
)


##
# Scene definition
##


@configclass
class G1WujiDoormanSceneCfg(InteractiveSceneCfg):
    """Configuration for scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # Robot
    robot: ArticulationCfg = G1_WUJI_FREE_BASE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # Original DoorMan starting pose: the door root is at x=0.60 m,
    # while the robot root starts at x=0.04 m (about 0.56 m apart).
    robot.init_state.pos = (0.04, -0.20, 0.75)

    # Door
    door: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=DOOR_SPAWNER_CFG,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.04),
            joint_pos={
                ".*hinge.*": 0.0,
                ".*handle.*": 0.0,
                ".*latch.*": 0.0,
            },
            joint_vel={
                ".*": 0.0,
            },
        ),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=[".*hinge.*"],
                velocity_limit_sim=100.0,
                stiffness=None,
                damping=None,
            ),
            "handle": ImplicitActuatorCfg(
                joint_names_expr=[".*handle.*"],
                velocity_limit_sim=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )
    

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Composite high-level arm/hand action with frozen HOMIE standing."""

    doorman = mdp.G1WujiDoormanActionCfg(
        asset_name="robot",
        arm_action_scale=0.25,
        homie_decimation=4,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset the robot and door to their configured default states."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
   


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##


@configclass
class G1WujiDoormanEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: G1WujiDoormanSceneCfg = G1WujiDoormanSceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # 200 Hz physics, 50 Hz HOMIE, 25 Hz high-level policy.
        self.decimation = 8
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
