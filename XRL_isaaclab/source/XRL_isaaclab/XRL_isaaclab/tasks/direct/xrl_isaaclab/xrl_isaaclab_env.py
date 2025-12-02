# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.utils.math as math_utils

from .xrl_isaaclab_env_cfg import XrlIsaaclabEnvCfg

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config import ROUGH_TERRAINS_CFG

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""

    # Requires environmental variable ISAAC_ASSETS to be set to the path where Isaac Sim assets are located.  
    # This is brittle and should be replaced with a more robust asset management solution.
    isaac_assets = os.environ.get("ISAAC_ASSETS")
    if isaac_assets is None:
        raise EnvironmentError("ISAAC_ASSETS environment variable is not set. Please set it before running the application.")
    print(f"ISAAC_ASSETS environment variable: {isaac_assets}")
    x_arrow_path = f"{isaac_assets}/Assets/Isaac/4.5/Isaac/Props/UIElements/arrow_x.usd"
    disk_path = f"{isaac_assets}/Assets/Isaac/4.5/Isaac/Props/Shapes/disk.usd"


    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=x_arrow_path,
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                # "command": sim_utils.UsdFileCfg(
                #     usd_path=x_arrow_path,
                #     scale=(0.25, 0.25, 0.5),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                # ),
                "target": sim_utils.UsdFileCfg(
                    usd_path=disk_path,
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
        }
    )
    return VisualizationMarkers(cfg=marker_cfg)

class XrlIsaaclabEnv(DirectRLEnv):
    cfg: XrlIsaaclabEnvCfg

    def __init__(self, cfg: XrlIsaaclabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        #add background
        # cfg = sim_utils.UsdFileCfg(
        #     usd_path = "/home/jrshs79/isaacsim/isaacsim_assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Environments/Terrains/rough_plane.usd"
        # )

        # prim_path = '/World/background'
        # cfg.func(prim_path, cfg)
        #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX v
        # Terrain importer configuration
        terrain_importer_cfg = TerrainImporterCfg(
            prim_path="/World/Terrain",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,   # <-- REQUIRED; Adjustment made in rough.py script in the IsaacLab source files outside the current project
            #noise range = (-0.12, 0.12), noise step = 0.008, downsampled scale = 0.4
        )

        # Instantiate importer
        terrain_importer_cfg.class_type(terrain_importer_cfg)
        #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ^
        # Auto-import happens inside __init__, so NO further calls needed.
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX v
        # 3) Flat spawn patches
        flat_patch_cfg = sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.025),  # Lx, Ly, thickness
            collision_props=sim_utils.CollisionPropertiesCfg()
            # visual_material=sim_utils.PreviewSurfaceCfg(
            #     diffuse_color=(0.1, 0.8, 0.1)  # optional: make it green so you can see it
            # ),
        )
        flat_patch_cfg.func(
            "/World/envs/env_.*/flat_spawn",
            flat_patch_cfg,
            translation=(0.0, 0.0, 0.0125),      # ≈ thickness/2
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ^
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        # setting aside useful variables for later
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.pose_commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()#set to 3 to account for the x,y, and z position data
        #self.pose_commands = self.pose_commands/torch.linalg.norm(self.pose_commands, dim=1, keepdim=True)
        self.pose_commands[:, -1] = 0.0
        self.offsets = self.scene.env_origins[:,:3].clone() #save the individual environment offsets
        print(self.pose_commands)


        # offsets to account for atan range and keep things on [-pi, pi]
        #self.pose = self.robot.data.root_com_pose_w[:,0:3]
        ratio = self.pose_commands[:,1]/(self.pose_commands[:,0]+1E-8)
        gzero = torch.where(self.pose_commands > 0, True, False)
        lzero = torch.where(self.pose_commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.forward_marker_location = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.command_marker_location = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.target_marker_location = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.target_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

    def _visualize_markers(self):
        # get marker locations and orientations
        self.forward_marker_location = self.robot.data.root_pos_w
        self.command_marker_location = self.robot.data.root_pos_w
        self.target_marker_location = self.pose_commands
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # offset markers so they are above the jetbot
        forward_loc = self.forward_marker_location + self.marker_offset
        target_loc = self.target_marker_location
        target_loc = target_loc + self.marker_offset #offset target marker to be above ground plane
        loc = torch.vstack((forward_loc, target_loc))
        rots = torch.vstack((self.forward_marker_orientations, self.target_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        self.pose = self.robot.data.root_com_pose_w[:,0:3]
        pose_target = torch.sub(self.pose_commands, self.pose)
        self.forwards[:,-1] = 0.0
        pose_target[:,-1] = 0.0

        dot = torch.sum(self.forwards * pose_target, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, pose_target, dim=-1)[:,-1].reshape(-1,1)
        #forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)

        
        x_pose = self.pose[:,0] #column vector for all current x positions
        x_commands = self.pose_commands[:,0] #column vector for all x commands
        y_pose = self.pose[:,1] #column vector for all current y positions
        y_commands = self.pose_commands[:,1] #column vector for all x commands
        x_dif = torch.sub(x_commands,x_pose)
        y_dif = torch.sub(y_commands,y_pose)
        dist = torch.sqrt((torch.pow(x_dif,2) + torch.pow(y_dif,2))).reshape(-1,1)

        obs = torch.hstack((dot, cross, dist))

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        vel = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        #v_0 = 2.0
        #velocity_reward = (vel/v_0)-1


        pose_target = torch.sub(self.pose_commands, self.pose)
        alignment_reward = torch.sum(self.forwards * pose_target, dim=-1, keepdim=True)

        x_pose = self.pose[:,0] #column vector for all current x positions
        x_commands = self.pose_commands[:,0] #column vector for all x commands
        y_pose = self.pose[:,1] #column vector for all current y positions
        y_commands = self.pose_commands[:,1] #column vector for all x commands
        x_dif = torch.sub(x_commands,x_pose)
        y_dif = torch.sub(y_commands,y_pose)
        dist = torch.pow((torch.pow(x_dif,2) + torch.pow(y_dif,2)),0.5).reshape(-1,1)
        d_0 = 5.0
        distance_reward = 1-(dist/d_0) #minimize the distance from agent to target

        total_reward = vel*torch.exp(alignment_reward) + distance_reward
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # pick new commands for reset envs
        self.pose_commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda() + 5
        self.pose_commands[:,-1] = 0.0
        #self.pose_commands = self.pose_commands/torch.linalg.norm(self.pose_commands, dim=1, keepdim=True)
        self.pose_commands = self.pose_commands + self.offsets  #Apply the offsets pulled in the _setup_scene method to the commands to ensure that the commands are populated at the individual environment level.
        

        # recalculate the orientations for the command markers with the new commands
        ratio = self.pose_commands[:,1]/(self.pose_commands[:,0]+1E-8)
        gzero = torch.where(self.pose_commands > 0, True, False)
        lzero = torch.where(self.pose_commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        # set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()