# XRL_IsaacLab
Framework for training ground based and submersible autonomous systems using Explainable Reinforcement Learning (XRL) using IsaacLab.

# Isaacsim Default Assets
The isaaaclab documentation pulls the majority of its terrain and robot assets from the default assets stored on NVIDIA's Nucleus server.
However, the Nucleus server is being deprectaed and will no longer be supported come October, 2025.  The easiest solution will be to download
the assets and save them locally.  The nucleus server called in the prim_path is then just replaced by the local path. See below for further
instructions or visist the Isaacsim 4.5 installation tips page (https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_faq.html)

1.  Download the 3 asset packs that correspond to your version of isaacsim from the following link (currently using isaacsim 4.5):
    https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html#isaac-sim-latest-release

2.  Make a new directory in your isaacsim source file called isaacsim_assets

    mkdir ~/isaacsim_assets

3.  Unzip the 3 downloaded asset packs which should still be in your downloads folder.

    cd ~/Downloads
    unzip "isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005.zip" -d ~/isaacsim_assets
    unzip "isaac-sim-assets-2@4.5.0-rc.36+release.19112.f59b3005.zip" -d ~/isaacsim_assets
    unzip "isaac-sim-assets-3@4.5.0-rc.36+release.19112.f59b3005.zip" -d ~/isaacsim_assets

4.  Run Isaac Sim with the flag below to use the local assets.

    ./isaac-sim.sh --/persistent/isaac/asset_root/default="/home/<username>/isaacsim_assets/Assets/Isaac/4.5"
