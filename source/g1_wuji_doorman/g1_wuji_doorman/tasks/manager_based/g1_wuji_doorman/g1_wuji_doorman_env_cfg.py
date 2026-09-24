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
from isaaclab.sensors import ContactSensorCfg
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
    activate_contact_sensors=True,

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
    rand_hook_length=0.05,
    # Freeze the remaining frame/decorative geometry for this single-door baseline.
    rand_total_wall_height=2.7,
    rand_door_cover_width=0.04,
    rand_spawn_keyhole=False,
    rand_keyhole_offset=0.075,
    rand_num_subpanels=0,
    rand_subpanel_frame_width=0.125,
    rand_subpanel_bottom=0.0,

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

    # A single handle body filtered against 25 separate hand links (one-to-many).
    handle_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Door/door_handle",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + link for link in mdp.FINGER_LINKS],
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
        homie_decimation=4,
        delta_action_scale=0.3,
        delta_action_clip=15.0,
        action_scale=0.25,
    )


@configclass
class CommandsCfg:
    door_task = mdp.DoorTaskStateCfg()


@configclass
class ObservationsCfg:
    """DoorMan-style teacher actor and asymmetric privileged critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        stage = ObsTerm(func=mdp.door_task_observation, params={"name": "stage"})
        gravity = ObsTerm(func=mdp.door_task_observation, params={"name": "gravity"})
        base_ang_vel = ObsTerm(func=mdp.door_task_observation, params={"name": "base_ang_vel"})
        base_lin_vel = ObsTerm(func=mdp.door_task_observation, params={"name": "base_lin_vel"})
        root_in_door = ObsTerm(func=mdp.door_task_observation, params={"name": "root_in_door"})
        arm_pos = ObsTerm(func=mdp.door_task_observation, params={"name": "arm_pos"})
        arm_vel = ObsTerm(func=mdp.door_task_observation, params={"name": "arm_vel"})
        hand_pos = ObsTerm(func=mdp.door_task_observation, params={"name": "hand_pos"})
        hand_vel = ObsTerm(func=mdp.door_task_observation, params={"name": "hand_vel"})
        target_in_palm = ObsTerm(func=mdp.door_task_observation, params={"name": "target_in_palm"})
        door_state = ObsTerm(func=mdp.door_task_observation, params={"name": "door_state"})
        tip_forces = ObsTerm(func=mdp.door_task_observation, params={"name": "tip_forces"})
        privileged_door_info = ObsTerm(func=mdp.door_task_observation, params={"name": "privileged_door_info"})
        finger_forces = ObsTerm(func=mdp.door_task_observation, params={"name": "finger_forces"})
        actions = ObsTerm(func=mdp.door_task_observation, params={"name": "actions"})
        delta_actions = ObsTerm(func=mdp.door_task_observation, params={"name": "delta_actions"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        full_handle_contact = ObsTerm(func=mdp.door_task_observation, params={"name": "full_handle_contact"})
        teacher_task_state = ObsTerm(func=mdp.door_task_observation, params={"name": "teacher_task_state"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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
    """Initial smoke-training weights; progress/bonus terms account for step_dt."""
    reach = RewTerm(func=mdp.door_task_reward, weight=4.0, params={"name": "reach"})
    align = RewTerm(func=mdp.door_task_reward, weight=1.0, params={"name": "align"})
    open_hand = RewTerm(func=mdp.door_task_reward, weight=0.5, params={"name": "open_hand"})
    contact = RewTerm(func=mdp.door_task_reward, weight=2.0, params={"name": "contact"})
    closure = RewTerm(func=mdp.door_task_reward, weight=0.5, params={"name": "closure"})
    handle_amount = RewTerm(func=mdp.door_task_reward, weight=1.0, params={"name": "handle_amount"})
    handle_progress = RewTerm(func=mdp.door_task_reward, weight=2.0, params={"name": "handle_progress"})
    handle_regression = RewTerm(func=mdp.door_task_reward, weight=-2.0, params={"name": "handle_regression"})
    latch = RewTerm(func=mdp.door_task_reward, weight=0.5, params={"name": "latch"})
    door_progress = RewTerm(func=mdp.door_task_reward, weight=5.0, params={"name": "door_progress"})
    door_amount = RewTerm(func=mdp.door_task_reward, weight=0.5, params={"name": "door_amount"})
    door_regression = RewTerm(func=mdp.door_task_reward, weight=-5.0, params={"name": "door_regression"})
    held_open = RewTerm(func=mdp.door_task_reward, weight=1.0, params={"name": "held_open"})
    transition = RewTerm(func=mdp.door_task_reward, weight=2.0, params={"name": "transition"})
    success = RewTerm(func=mdp.door_task_reward, weight=10.0, params={"name": "success"})
    failure = RewTerm(func=mdp.door_task_reward, weight=-5.0, params={"name": "failure"})
    upright = RewTerm(func=mdp.door_task_reward, weight=-2.0, params={"name": "upright"})
    base_drift = RewTerm(func=mdp.door_task_reward, weight=-1.0, params={"name": "base_drift"})
    right_arm_rest = RewTerm(func=mdp.door_task_reward, weight=-0.1, params={"name": "right_arm_rest"})
    action_rate = RewTerm(func=mdp.door_task_reward, weight=-0.01, params={"name": "action_rate"})
    joint_velocity = RewTerm(func=mdp.door_task_reward, weight=-0.0001, params={"name": "joint_velocity"})
    joint_limits = RewTerm(func=mdp.door_task_reward, weight=-1.0, params={"name": "joint_limits"})
    excess_force = RewTerm(func=mdp.door_task_reward, weight=-0.1, params={"name": "excess_force"})
    lost_contact = RewTerm(func=mdp.door_task_reward, weight=-0.5, params={"name": "lost_contact"})


@configclass
class TerminationsCfg:
    # Every task termination ensures the same idempotent state update; order is free.
    invalid_state = DoneTerm(func=mdp.task_invalid_state)
    fall = DoneTerm(func=mdp.task_fall)
    success = DoneTerm(func=mdp.task_success)
    stage_timeout = DoneTerm(func=mdp.task_stage_timeout, time_out=True)
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
    commands: CommandsCfg = CommandsCfg()
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
        self.episode_length_s = 30
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
