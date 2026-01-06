# XRL_IsaacLab
Framework for training ground based and submersible autonomous systems using Explainable Reinforcement Learning (XRL) using IsaacLab.

## Setup:  Isaacsim Default Assets
The isaaaclab documentation pulls the majority of its terrain and robot assets from the default assets stored on NVIDIA's Nucleus server. However, the Nucleus server is being deprectaed and will no longer be supported come October, 2025.  The easiest solution will be to download the assets and save them locally.  The nucleus server called in the prim_path is then just replaced by the local path. See below for further instructions or visit the Isaacsim 4.5 installation tips page (https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack) - see the *Assets* tab.

1.  Download the 3 asset packs that correspond to your version of isaacsim from the following link (currently using isaacsim 4.5):
    https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html#isaac-sim-latest-release

2.  Make a new directory called `isaacsim_assets`.  This can be anywhere, but for the example we'll have it in the user's home directory.

    '''
    mkdir ~/isaacsim_assets
    '''
    

3.  Unzip the 3 downloaded asset packs which should still be in your downloads folder.

    '''
    cd ~/Downloads
    unzip "isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005.zip" -d ~/isaacsim_assets
    unzip "isaac-sim-assets-2@4.5.0-rc.36+release.19112.f59b3005.zip" -d ~/isaacsim_assets
    unzip "isaac-sim-assets-3@4.5.0-rc.36+release.19112.f59b3005.zip" -d ~/isaacsim_assets
    '''

4.  Run Isaac Sim with the flag below to use the local assets. This is not persistent - probably should find a way to put this setting in a persistent way.

    '''
    ./isaac-sim.sh --/persistent/isaac/asset_root/default="${HOME}/isaacsim_assets/Assets/Isaac/4.5"
    '''

    This will bring up the IS GUI and may require you to log in with your local credentials.


## Usage Examples

### Conda Set Up
See https://github.com/Field-Robotics-Lab/isaac_lab_walkthrough for additional details.

Activate conda environment
```
eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)" 
conda activate env_isaaclab
```
### Environmental Variable

To avoid hardcoding asset paths, we set an environmental variable to declare the local location of the asset packs

```
export ISAAC_ASSETS=${HOME}$/isaacsim_assets 
```

### Install Verification Tests

**List environments**:
This verifies the install.

```
python ./XRL_isaaclab/scripts/list_envs.py 
```

Should see a task named `Template-Xrl-Isaaclab-Direct-v0`


**Zero Agent**:
Test of the training environment configuration with an agent that doesn't do anything.

```
export ISAAC_ASSETS=/home/bsb/isaacsim_assets 
python ./XRL_isaaclab/scripts/zero_agent.py --num_envs=10 --task=Template-Xrl-Isaaclab-Direct-v0
```

## Working Branches

### Tutorial

Initial setup of an skrl single agent environment that was derived from the official IsaacLab walkthrough pages found below:
https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/index.html

See the following link for instructions on how to swap in different robot assets and terrain packages: 
https://github.com/Field-Robotics-Lab/isaac_lab_walkthrough

### end_to_end

Modify the resulting walkthrough enviornment to teach the specified robot to drive to a randomly generated target location

1.  Update Configuration: In XRL_isaaclab_env_cfg, update the parameters for the environment depending on the robot you choose to use.  For this iteration, we are still using the jetbot but spawn in the four wheel drive jackal later in the rough_terrain extension.  For the jetbot, the action space is 2 (both wheel) and the observation space becomes three representing the dot product and cross product of the forward vector and the euclidean distance to the desired target.

...

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
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10, env_spacing=5.0, replicate_physics=True)
        dof_names = ["left_wheel_joint", "right_wheel_joint"]
...

2.  Update markers: To change the marker setup to account for a red target marker while keeping the forward pointing green arrow, comment out the command marker segment in define_markers.
  - Add a red disk by copying the forward marker setup, name it, change the color values, and point it to the disk asset in the previously downloaded assets folder.

...

    def define_markers() -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                    "forward": sim_utils.UsdFileCfg(
                        usd_path="/home/jrshs79/isaacsim/isaacsim_assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Props/UIElements/arrow_x.usd",
                        scale=(0.25, 0.25, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    # "command": sim_utils.UsdFileCfg(
                    #     usd_path="/home/jrshs79/isaacsim/isaacsim_assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Props/UIElements/arrow_x.usd",
                    #     scale=(0.25, 0.25, 0.5),
                    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    # ),
                    "target": sim_utils.UsdFileCfg(
                        usd_path="/home/jrshs79/isaacsim/isaacsim_assets/isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005/Assets/Isaac/4.5/Isaac/Props/Shapes/disk.usd",
                        scale=(0.25, 0.25, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    )
            }
        )
        return VisualizationMarkers(cfg=marker_cfg)
...

  - Under _setup_scene add target_marker_location and target_marker_orientations placeholders similar to the other two marker types.  You can keep, comment out, or delete the command marker placeholders.
  - Under _visualize_markers set the target_marker_location values equal to the pose_commands and leave the target_marker_orientations values as zeros.  sub the target marker values for the command marker values in rots and loc.

3.  Update Observations: Under _get_observations, use the math_utils function quat_apply to apply the root_link_quat_w quaternion to the FORWARD_VEC_B value in the articulation data and set this to the self.forwards value.
    - Apply the root_com_pose_w value to self.pose and subtract the pose vector from pose_command vector to get the pose_target vector.
    - Zero out the z component of the forwards and pose_target vectors.
    - Substitute these two vectors into the dot and cross values from the tutorial
    - Separate out the x and y components of the command vectors and the pose vectors.  Apply these values to the euclidean distance formular using the math operations provided by pytorch
  
...

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
...

4. Update Rewards: Under _get_rewards re-calulate the distance value and pose_target values from above.
    - Identify a threshold distance value, d_0
    - calulate the distance reward by dividing the current distance by the threshold and subtracting from 1 to encourage the robot to minimize the value.
    - Keep the reward signal from the tutorial and add the new distance value

### rough_terrain

Modify end_to_end environment with procedually generated terrain using the rough.py method provided with the basic enviornment setup.

### update_reward

Changed the reward structure from that provided in the walkthrough tutorial to include forward velocity, roll, pitch, distance, and alignment signals.
