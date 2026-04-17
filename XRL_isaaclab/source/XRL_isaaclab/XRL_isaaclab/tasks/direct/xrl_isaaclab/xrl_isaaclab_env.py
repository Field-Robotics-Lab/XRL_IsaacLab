# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import math
import numpy as np
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
from isaaclab.sensors import RayCaster
#from isaacsim.robot.wheeled_robots.controllers import DifferentialController

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
                "command": sim_utils.UsdFileCfg(
                    usd_path=x_arrow_path,
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
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
        self._controller = None

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        N = self.cfg.scene.num_envs
        device = self.device
        self._prev_dist = torch.zeros((N, 1), device=device)
        self._stuck_count = torch.zeros((N,), dtype=torch.int32, device=device)
        self._angle_count = torch.zeros((N,), dtype=torch.int32, device=device)
        self._success_count = torch.zeros((N,), dtype=torch.int32, device=device)
        self._turned_around = torch.zeros((N,), dtype=torch.bool, device=device)
        self._is_stuck = torch.zeros((N,), dtype=torch.bool, device=device)
        

    def _build_controller(self):
        if self._controller is None:
            from isaacsim.robot.wheeled_robots.controllers import DifferentialController
            wheel_radius = 0.2
            wheel_base = 0.3765
            self._controller = DifferentialController(
                name="diff_drive",
                wheel_radius=wheel_radius,
                wheel_base=wheel_base,
            )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        #add background
        #     # Terrain importer configuration
        # terrain_importer_cfg = TerrainImporterCfg(
        #     prim_path="/World/Terrain",
        #     terrain_type="generator",
        #     terrain_generator=ROUGH_TERRAINS_CFG,   # <-- REQUIRED; Adjustment made in rough.py script in the IsaacLab source files outside the current project
        #     #noise range = (-0.12, 0.12), noise step = 0.008, downsampled scale = 0.4; for jetbot
        #     #noise range = (-0.2, 0.2), noise step = 0.005, downsampled scale = 0.4; for jackal
        # )
        #     # Instantiate importer
        # self.terrain_importer = terrain_importer_cfg.class_type(terrain_importer_cfg)
        #     # Auto-import happens inside __init__, so NO further calls needed
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # # add raycaster for depth measurments
        # self.ground_ray = RayCaster(self.cfg.ground_ray)
        # self.scene.sensors["ground_ray"] = self.ground_ray

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        # setting aside useful variables for later
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.pose_commands = 2 * torch.randn((self.cfg.scene.num_envs, 3)).cuda()  #set to 3 to account for the x,y, and z position data
        #self.pose_commands = self.pose_commands/torch.linalg.norm(self.pose_commands, dim=1, keepdim=True)
        self.pose_commands[:, -1] = 0.0
        self.offsets = self.scene.env_origins[:,:3].clone() #save the individual environment offsets
        self.pose_commands += self.offsets

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
        #self.forward_marker_location = self.robot.data.root_pos_w
        self.command_marker_location = self.robot.data.root_pos_w
        self.target_marker_location = self.pose_commands
        #self.forward_marker_orientations = self.robot.data.root_quat_w
        cmd_vec = self.pose_commands - self.robot.data.root_pos_w
        cmd_vec[:, 2] = 0.0
        cmd_yaw = torch.atan2(cmd_vec[:, 1], cmd_vec[:, 0]).reshape(-1, 1)
        self.command_marker_orientations = math_utils.quat_from_angle_axis(cmd_yaw, self.up_dir).squeeze()
        self.target_marker_orientations = self.command_marker_orientations

        # offset markers so they are above the jetbot
        #forward_loc = self.forward_marker_location + self.marker_offset
        command_loc = self.command_marker_location + self.marker_offset
        target_loc = self.target_marker_location #+ self.marker_offset  # offset target marker to be above ground plane
        loc = torch.vstack((command_loc, target_loc))
        rots = torch.vstack((self.command_marker_orientations, self.target_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack(
            (torch.zeros_like(all_envs), torch.ones_like(all_envs))
        )
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX v
    def _apply_action(self) -> None:
        # RL outputs [v, omega]
        self._build_controller()

        commands = self.actions[:, :2].detach().cpu().numpy()
        wheel_targets = np.asarray(
            [self._controller.forward(command).joint_velocities for command in commands],
            dtype=np.float32,
        )
        wheel_targets = torch.as_tensor(wheel_targets, device=self.device, dtype=self.actions.dtype)
        left_vel = wheel_targets[:, 0:1]
        right_vel = wheel_targets[:, 1:2]
        #Setup for simplified e2e
        # left = self.actions[:,0:1]
        # right = self.actions[:,1:2]

        # joint_targets = torch.zeros((self.cfg.scene.num_envs, self.num_wheel_joints), device=self.device)
        # joint_targets[:, self.left_wheel_ids] = left_vel.unsqueeze(-1)
        # joint_targets[:, self.right_wheel_ids] = right_vel.unsqueeze(-1)

        # self.robot.set_joint_velocity_target(joint_targets)

        expanded = torch.cat([left_vel, right_vel, left_vel, right_vel], dim=1)
        zero_expanded = torch.zeros_like(expanded)
        target_vel = torch.where(
            self.dist <= self.dist_0,
            zero_expanded,
            expanded
        )
        self.robot.set_joint_velocity_target(target_vel, joint_ids=self.dof_idx)
        #self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx) #needed for straight e2e

    def _get_observations(self) -> dict:
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        self.forwards[:,-1] = 0.0
        denom_for = torch.linalg.norm(self.forwards, dim=1, keepdim=True).clamp_min(1e-6)
        self.forwards_unit = self.forwards / denom_for
        self.pose = self.robot.data.root_com_pose_w[:,0:3]
        self.pose_target = torch.sub(self.pose_commands, self.pose)
        self.pose_target[:,-1] = 0.0
        denom_targ = torch.linalg.norm(self.pose_target, dim=1, keepdim=True).clamp_min(1e-6)
        self.pose_target_unit = self.pose_target / denom_targ

        self.dot = torch.sum(self.forwards * self.pose_target, dim=-1, keepdim=True)
        self.dot_norm = torch.sum(self.forwards_unit * self.pose_target_unit, dim=-1, keepdim=True)
        self.cross = torch.cross(self.forwards, self.pose_target, dim=-1)[:,-1].reshape(-1,1)
        self.cross_norm = torch.cross(self.forwards_unit, self.pose_target_unit, dim=-1)[:,-1].reshape(-1,1)
        #self.forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        self.forward_speed = self.robot.data.root_com_lin_vel_w[:,0].reshape(-1,1)


        x_pose = self.pose[:,0] #column vector for all current x positions
        x_commands = self.pose_commands[:,0] #column vector for all x commands
        y_pose = self.pose[:,1] #column vector for all current y positions
        y_commands = self.pose_commands[:,1] #column vector for all x commands
        x_dif = torch.sub(x_commands,x_pose)
        y_dif = torch.sub(y_commands,y_pose)
        self.dist = torch.sqrt((torch.pow(x_dif,2) + torch.pow(y_dif,2))).reshape(-1, 1)
        self.dist_0 = 2.0
        self.success = self.dist <= self.dist_0

        self.dot = torch.sum(self.forwards * self.pose_target, dim=-1, keepdim=True)
        cos_psi = torch.sum(self.pose_target_unit * self.forwards_unit, dim=-1, keepdim=True) #dot prod/magnitudes
        self.cos_psi = torch.clamp(cos_psi, -1.0, 1.0)
        self.cross = torch.cross(self.forwards, self.pose_target, dim=-1)[:,-1].reshape(-1,1)
        self.forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)

        self.euler = math_utils.euler_xyz_from_quat(self.robot.data.root_link_quat_w)
        self.roll  = self.euler[0] #pull the real-time roll angle from the euler tensor
        self.roll_deg = torch.rad2deg(self.roll).abs().unsqueeze(-1) #convert the roll angle above to degrees
        self.pitch = self.euler[1]
        self.pitch_deg = torch.rad2deg(self.pitch).abs().unsqueeze(-1)

        # # Get the ray sensor (use ONE handle consistently)
        # ray = self.scene.sensors["ground_ray"]

        # # Update it each step (your env doesn't have self.dt)
        # ray.update(self.cfg.sim.dt)

        # # Hit positions (num_envs, num_rays, 3)
        # ray_hits_w = ray.data.ray_hits_w

        # # One ray → ground z
        # #ground_z = ray_hits_w[:, 0, 2]
        # ground_z = ray_hits_w[..., 0, 2].reshape(self.num_envs)


        # # Fallback if miss is encoded as NaN/inf
        # ground_z = torch.where(
        #     torch.isfinite(ground_z),
        #     ground_z,
        #     self.robot.data.root_pos_w[:, 2] - 1.0
        # )

        # self.ground_z = ground_z

        # com_z = self.robot.data.root_com_pos_w[:, 2]
        # h = torch.clamp(com_z - self.ground_z, min=1e-3)
        # track_width = 0.3765
        # track_width_t = torch.full_like(h, track_width) #create a tensor the same size as h with the trackwidth value
        # wheel_base = 0.430
        # wheel_base_t = torch.full_like(h, wheel_base)
        # roll_crit = torch.atan2(2.0*h, track_width_t) #tread (t) is the distance between the center point of both tires on one axle. 376.5 mm or 0.3765 m on the Jackal
        # self.roll_crit_deg = torch.rad2deg(roll_crit).abs().unsqueeze(-1)
        # pitch_crit = torch.atan2(2.0*h, wheel_base_t)
        # self.pitch_crit_deg = torch.rad2deg(pitch_crit).abs().unsqueeze(-1)

        obs = torch.hstack((self.forward_speed, self.dot, self.cross, self.dist, self.roll_deg, self.pitch_deg))
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # roll_0 = 0.2 * self.roll_crit_deg #set the threshold angle for the reward to 80% of the critical roll angle
        # pitch_0 = 0.2 * self.pitch_crit_deg
        # r = 1-(self.roll_deg/roll_0)
        # roll_sig = 1/(1+torch.exp(-r))
        # p = 1-(self.pitch_deg/pitch_0)
        # pitch_sig = 1/(1+torch.exp(-p))

        # not_rolling = self.roll_deg < roll_0
        # roll_reward = torch.where(
        #     not_rolling,
        #     #roll_sig,
        #     torch.zeros_like(roll_sig),
        #     -1*roll_sig
        # )

        # not_pitching = self.pitch_deg < pitch_0
        # pitch_reward = torch.where(
        #     not_pitching,
        #     #pitch_sig,
        #     torch.zeros_like(pitch_sig),
        #     -1*pitch_sig
        # )

        alignment_reward = self.dot_norm

        is_aligned = alignment_reward >= 0.0
        scale = 0.001
        dist_delta = (self._prev_dist - self.dist)/scale
        distance_reward = dist_delta
        # distance_reward = torch.where(
        #     is_aligned,
        #     dist_delta,
        #     0.5 * dist_delta
        # )
        self._prev_dist = self.dist.detach()

        speed_reward = torch.sigmoid(self.forward_speed)
        #speed_reward = torch.tanh(self.forward_speed)

        success_sig = torch.full((self.cfg.scene.num_envs,1), 100, device=self.device)
        success_reward = torch.where(
            self.success,
            success_sig,
            torch.zeros_like(self.dist)
        )


        total_reward = (alignment_reward) + distance_reward + success_reward
        print(f'A:{alignment_reward[0][0]} S:{speed_reward[0][0]} D:{distance_reward[0][0]} Tot:{total_reward[0][0]}')
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        N = self.num_envs

        #calculate necessary data to determine if the vehicle is  stuck
        no_change = 0.0005
        #no_change = 1e-2 #threshold to determine if there is no change from the previous timestep.  currently set to 1mm
        steps_required = 500 # X consecutive timesteps
        delta = (self._prev_dist - self.dist).abs().squeeze(-1)
        not_moving = delta < no_change

        #update counters
        self._stuck_count = torch.where(not_moving, self._stuck_count + 1, torch.zeros_like(self._stuck_count))
        self._is_stuck = self._stuck_count >= steps_required

        # R_crit = self.roll_deg.abs() >= 0.7*self.roll_crit_deg
        # R_crit = R_crit.squeeze(-1)
        # P_crit = self.pitch_deg.abs() >= 0.7*self.pitch_crit_deg
        # P_crit = P_crit.squeeze(-1)
        psi_target = torch.acos(self.cos_psi.squeeze(-1))
        psi_target_deg = torch.rad2deg(psi_target).abs()
        A_crit = psi_target_deg > 135
        #self._turned_around = turned_around.squeeze(-1)
        self._angle_count = torch.where(
            A_crit,
            self._angle_count + 1,
            torch.zeros_like(self._angle_count)
        )
        self._turned_around = self._angle_count > steps_required
        #terminated = R_crit | P_crit | A_crit | self._is_stuck
        self._success_count = torch.where(
            self.success.squeeze(-1),
            self._success_count + 1,
            torch.zeros_like(self._angle_count)
        )
        self.goal = self._success_count > steps_required

        terminated = self._turned_around | self._is_stuck | self.goal 

        terminated = terminated.to(torch.bool).reshape(N)
        time_out  = time_out.to(torch.bool).reshape(N)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # pick new commands for reset envs
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.reshape(-1)
        self.pose = self.robot.data.root_com_pose_w[:,0:3]

        # # Old rejection sampling (kept for reference)
        # min_r = torch.full((env_ids.numel(),), 3.0, device=self.device)
        # xy = 2 * torch.randn((env_ids.numel(), 2), device=self.device)
        # for _ in range(10):
        #     r = torch.linalg.norm(xy, dim=1)
        #     mask = r < min_r
        #     if not mask.any():
        #         break
        #     xy[mask] = 2 * torch.randn((mask.sum(), 2), device=self.device)
        
        # self.pose_commands[env_ids, :2] = xy + self.offsets[env_ids, :2]
        # self.pose_commands[env_ids, 2] = 0.0

        # Annulus sampling: always respects minimum radius
        min_r = 3.0
        max_r = 8.0
        theta = 2 * torch.pi * torch.rand((env_ids.numel(),), device=self.device)
        rad = torch.sqrt(
            torch.rand((env_ids.numel(),), device=self.device) * (max_r**2 - min_r**2) + min_r**2
        )
        xy = torch.stack((rad * torch.cos(theta), rad * torch.sin(theta)), dim=1)
        self.pose_commands[env_ids, :2] = xy + self.pose[env_ids, :2]
        self.pose_commands[env_ids, 2] = 0.0

        #calculate distance to target
        x_pose = self.pose[:,0] #column vector for all current x positions
        x_commands = self.pose_commands[:,0] #column vector for all x commands
        y_pose = self.pose[:,1] #column vector for all current y positions
        y_commands = self.pose_commands[:,1] #column vector for all x commands
        x_dif = torch.sub(x_commands,x_pose)
        y_dif = torch.sub(y_commands,y_pose)
        dist_all = torch.sqrt((torch.pow(x_dif,2) + torch.pow(y_dif,2))).reshape(-1, 1)
        dist = dist_all[env_ids]

        #reset the environment buffers for determining if the robot is stuck
        self._prev_dist[env_ids] = dist.detach()
        self._stuck_count[env_ids] = 0
        self._is_stuck[env_ids] = False
        self._angle_count[env_ids] = 0
        self._turned_around[env_ids] = False

        # recalculate the orientations for the command markers with the new commands
        cmds = self.pose_commands[env_ids]
        ratio = cmds[:,1]/(cmds[:,0]+1E-8)
        x = cmds[:,0]
        y = cmds[:,1]
        plus = (x < 0) & (y > 0)
        minus = (x < 0) & (y < 0)
        #offsets = torch.pi*plus - torch.pi*minus
        offsets = (torch.pi * plus.to(cmds.dtype)) - (torch.pi * minus.to(cmds.dtype))  #
        #self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
        self.yaws[env_ids, 0] = torch.atan(ratio) + offsets

        # # set the root state for the reset envs
        # fp = self.terrain_importer.flat_patches
        # default_root_state = self.robot.data.default_root_state[env_ids]
        # fp_root_state = default_root_state.clone()

        # root_spawn = fp.get("root_spawn", None)
        # if root_spawn is not None:
        #     if root_spawn.ndim >= 3:
        #         points = root_spawn[env_ids]
        #         centers = points.mean(dim=2)
        #         #centers_w = centers + self.scene.env_origins[env_ids].unsqueeze(1)
        #         points_dist = centers[..., 0] ** 2 + centers[..., 1] ** 2
        #         best_idx = torch.argmin(points_dist, dim=1)
        #         patch_local = points[torch.arange(env_ids.numel()), best_idx, 0]
        #     elif root_spawn.ndim == 2:
        #         patch_local = root_spawn[env_ids, 0]
        #     else:
        #         patch_local = root_spawn[env_ids]
        #     patch_local = patch_local.reshape(-1, 3)
        #     fp_root_state[:, :3] += patch_local
        #     fp_root_state[:, :3] += self.scene.env_origins[env_ids]
        #     fp_root_state[:, 2] += 0.25  # put it above ground a bit (don’t trust patch_local[2])
        #     # Spawn target near the flat patch (world coords)
        #     new_cmds = torch.randn((env_ids.numel(), 3), device=self.device)
        #     new_cmds[:, :2] = 3.0 * new_cmds[:, :2]  # +/- ~2m in XY
        #     new_cmds[:, 2] = 0.0
        #     new_cmds[:, :2] += fp_root_state[:, :2]
        #     self.pose_commands[env_ids] = new_cmds
        #     self.robot.write_root_state_to_sim(fp_root_state, env_ids)
        # else:
        #     # Fallback: keep target near env origin
        #     new_cmds = torch.randn((env_ids.numel(), 3), device=self.device) + 3
        #     new_cmds[:, :2] = 3.0 * new_cmds[:, :2]
        #     new_cmds[:, 2] = 0.0
        #     new_cmds[:, :2] += self.scene.env_origins[env_ids, :2]
        #     self.pose_commands[env_ids] = new_cmds
        #     default_root_state[:, :3] += self.scene.env_origins[env_ids]
        #     self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()
