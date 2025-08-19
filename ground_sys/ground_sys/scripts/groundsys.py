import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
#from isaaclab.sim.converters import spawn_from_usd
from isaaclab.assets import RigidObjectCfg

@configclass
class groundsysCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path = '/World/groundsysPlane',
        #spawn = spawn_from_usd(usd_path = '/home/jrshs79/Desktop/safety_park_high_res_textured.usd')
        spawn = sim_utils.UsdFileCfg(usd_path = '/home/jrshs79/Desktop/ground_sys/safety_park_high_res_textured.usd')
        #init_state = AssetInitStateCFG(rot=(0.6959,0.56077,0.0,0.44862))
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=75.0, color=(0.75, 0.75, 0.75))
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        #if count % 500 == 0:
        #    # reset counter
        #    count = 0
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=None)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = groundsysCfg(num_envs=1,env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()