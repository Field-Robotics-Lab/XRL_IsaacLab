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

### end_to_end

Modify the resulting walkthrough enviornment to teach the specified robot to drive to a randomly generated target location

### rough_terrain

Modify end_to_end environment with procedually generated terrain using the rough.py method provided with the basic enviornment setup.

### update_reward

Changed the reward structure from that provided in the walkthrough tutorial to include forward velocity, roll, pitch, distance, and alignment signals.
