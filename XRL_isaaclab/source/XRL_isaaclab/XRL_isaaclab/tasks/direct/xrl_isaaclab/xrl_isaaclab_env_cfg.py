# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from XRL_isaaclab.robots.jackal_basic import JACKAL_BASIC_CONFIG

from copy import deepcopy

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass


XRL_ROLLING_TERRAINS_CFG = deepcopy(ROUGH_TERRAINS_CFG)
XRL_ROLLING_TERRAINS_CFG.size = (50.0, 50.0)
XRL_ROLLING_TERRAINS_CFG.num_rows = 1
XRL_ROLLING_TERRAINS_CFG.num_cols = 1
ROOT_SPAWN_PATCH_CFG = XRL_ROLLING_TERRAINS_CFG.sub_terrains["random_rough"].flat_patch_sampling["root_spawn"]
ROOT_SPAWN_PATCH_CFG.num_patches = 128
ROOT_SPAWN_PATCH_CFG.x_range = (-17.0, 17.0)
ROOT_SPAWN_PATCH_CFG.y_range = (-17.0, 17.0)


@configclass
class XrlIsaaclabEnvCfg(DirectRLEnvCfg):
    # env
    #seed = 5
    decimation = 2
    episode_length_s = 60.0
    # - spaces definition
    action_space = 2
    observation_space = 6 #roll, pitch, distance, forward velocity, orientation dot product, and orientation cross product
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JACKAL_BASIC_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.21),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)

    terrain = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="generator",
        terrain_generator=XRL_ROLLING_TERRAINS_CFG,
        collision_group=-1,
        debug_vis=False,
    )

    dof_names = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']

    # ray sensor
    ground_ray = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot",
        mesh_prim_paths=["/World/Terrain"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        pattern_cfg=GridPatternCfg(
            resolution=0.05,
            size=(0.01, 0.01),
            direction=(0.0, 0.0, -1.0),
        ),
        max_distance=5.0,
        ray_alignment="world",
    )
