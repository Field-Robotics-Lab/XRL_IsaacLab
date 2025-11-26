import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Check the ISAAC_ASSETS environment variable
isaac_assets = os.environ.get("ISAAC_ASSETS")  
if isaac_assets is None:
    raise EnvironmentError("ISAAC_ASSETS environment variable is not set. Please set it before running the application.")

print(f"ISAAC_ASSETS environment variable: {isaac_assets}")

jetbot_path = f"{isaac_assets}/Assets/Isaac/4.5/Isaac/Robots/Jetbot/jetbot.usd"
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=jetbot_path),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)