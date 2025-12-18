# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from XRL_isaaclab.robots.jackal_basic import JACKAL_BASIC_CONFIG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import math
from isaaclab.sensors import RayCasterCfg


@configclass
class XrlIsaaclabEnvCfg(DirectRLEnvCfg):
    # env
    #seed = 5
    decimation = 2
    episode_length_s = 45.0
    # - spaces definition
    action_space = 4
    observation_space = 3 #x,y,z velocities and euclidean distance to the target location
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
    dof_names = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']
    ##################################################################
    # ray sensor
    ground_ray = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",  # cast from base
        offset=sim_utils.Transform(
            pos=(0.0, 0.0, 0.5),   # start ray above base
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        ray_directions=[(0.0, 0.0, -1.0)],  # straight down
        max_distance=5.0,
        debug_vis=False,
    )