# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from XRL_isaaclab.robots.jetbot import JETBOT_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import math


@configclass
class XrlIsaaclabEnvCfg(DirectRLEnvCfg):
    # env
    #seed = 5
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 2
    observation_space = 3 #x,y,z velocities and euclidean distance to the target location
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=5.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]