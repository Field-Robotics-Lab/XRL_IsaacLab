# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import shlex
import sys
import textwrap

from isaaclab.app import AppLauncher

import inspect

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime
from pprint import pformat

import skrl
import numpy as np
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
    from skrl.trainers.torch import StepTrainer, SequentialTrainer, ParallelTrainer
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import XRL_isaaclab.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


# ADDED: Helpers for run metadata, reward-source capture, and success-count reporting.
def _get_method_source(obj, method_name: str) -> str:
    method = getattr(type(obj), method_name, None)
    if method is None:
        return f"[missing] {type(obj).__name__}.{method_name}"

    try:
        return textwrap.dedent(inspect.getsource(method)).strip()
    except (OSError, TypeError):
        return f"[unavailable] {type(obj).__name__}.{method_name}"


def _write_training_command(
    log_dir: str, early_stopping_params: dict, training_cfg: dict, reward_details: str
) -> None:
    command = shlex.join(sys.orig_argv)
    early_stopping_args = " ".join(
        f"{name}={value}" for name, value in early_stopping_params.items()
    )
    command_with_early_stopping = f"{command}  # early_stop: {early_stopping_args}"

    with open(os.path.join(log_dir, "command.txt"), "w", encoding="utf-8") as file:
        file.write(
            "\n".join(
                [
                    "Command",
                    "-------",
                    command_with_early_stopping,
                    "",
                    "Early Stopping",
                    "--------------",
                    *(f"{name}: {value}" for name, value in early_stopping_params.items()),
                    "",
                    "Training Config",
                    "---------------",
                    pformat(training_cfg, sort_dicts=False),
                    "",
                    "Reward Function (_get_rewards)",
                    "------------------------------",
                    reward_details,
                ]
            )
            + "\n"
        )


def _append_success_summary(
    log_dir: str,
    success_iteration_count: int,
    success_env_iteration_count: int,
    completed_iterations: int,
    configured_num_envs: int,
) -> None:
    total_env_iterations = completed_iterations * configured_num_envs
    success_percentage = (
        100.0 * success_env_iteration_count / total_env_iterations if total_env_iterations > 0 else 0.0
    )

    with open(os.path.join(log_dir, "command.txt"), "a", encoding="utf-8") as file:
        file.write(
            "\n".join(
                [
                    "",
                    "Success Count",
                    "-------------",
                    "Criterion: success_reward mask from _get_rewards",
                    f"Successful training iterations: {success_iteration_count}",
                    f"Successful env iterations: {success_env_iteration_count}",
                    f"Total env iterations: {total_env_iterations}",
                    f"Successful env iteration percentage: {success_percentage:.2f}%",
                    f"Configured num envs: {configured_num_envs}",
                    f"Completed training iterations: {completed_iterations}",
                ]
            )
            + "\n"
        )
# END ADDED: Helpers for run metadata, reward-source capture, and success-count reporting.


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # ADDED: Keep the unwrapped env and capture the reward function source for command.txt.
    base_env = env.unwrapped
    reward_details = _get_method_source(env.unwrapped, "_get_rewards")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    agent = runner.agent

    # Hook TensorBoard write to capture the exact scalar value that TB logs
    _orig_write_tracking_data = agent.write_tracking_data

    def _write_tracking_data_hook(timestep: int, timesteps: int) -> None:
        tracked_total = agent.tracking_data.get("Reward / Total reward (mean)")
        if tracked_total:
            agent._last_tb_total_reward_mean = float(tracked_total[-1])
        else:
            agent._last_tb_total_reward_mean = None
        _orig_write_tracking_data(timestep, timesteps)

    agent.write_tracking_data = _write_tracking_data_hook

    agents_scope = [(0, env.num_envs)]  # one agent handles all envs
    trainer = StepTrainer(env,agent,agents_scope=agents_scope,cfg=agent_cfg["trainer"])


    # run training
    timesteps = agent_cfg["trainer"]["timesteps"]
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Original reward-based early stopping.
    plateau_window = min(1000, timesteps)
    plateau_patience = 50000
    plateau_rel_delta = 1e-6
    total_reward_means = []
    best_mean = -float("inf")
    plateau_counter = 0

    # Early stopping based on plateau of rolling success rate over completed episodes (kept for reference).
    # success_window = 300
    # success_patience = 5000
    # success_rate_delta = 0.001
    # episode_successes = []
    # best_success_rate = -float("inf")
    # success_plateau_counter = 0
    checkpoint_interval = max(1, timesteps // 100)
    early_stop_start = int(timesteps * 0.4)
    early_stopping_params = {
        "plateau_window": plateau_window,
        "plateau_patience": plateau_patience,
        "plateau_rel_delta": plateau_rel_delta,
        "early_stop_start": early_stop_start,
    }
    # ADDED: Write command metadata, training config, reward details, and initialize success counters.
    _write_training_command(log_dir, early_stopping_params, agent_cfg["trainer"], reward_details)
    success_iteration_count = 0
    success_env_iteration_count = 0
    completed_iterations = 0

    for i in range(timesteps):

        if isinstance(trainer.agents, list):
            trainer.agents = trainer.agents[0]

        next_states, rewards, terminated, truncated, infos = trainer.train()
        completed_iterations = i + 1

        # ADDED: Count successes using the same mask that drives success_reward in _get_rewards.
        success_mask = getattr(base_env, "_last_success_reward_mask", None)
        if success_mask is None:
            success_mask = getattr(base_env, "success", None)
        if success_mask is not None:
            success_mask = success_mask.reshape(-1).bool()
            successful_envs = int(success_mask.sum().item())
            success_env_iteration_count += successful_envs
            if successful_envs > 0:
                success_iteration_count += 1
        
        # Original reward-based early stopping.
        tracked_total = getattr(agent, "_last_tb_total_reward_mean", None)
        if tracked_total is not None:
            total_reward_means.append(tracked_total)

            if len(total_reward_means) >= early_stop_start:
                rolling_mean = sum(total_reward_means[-plateau_window:]) / plateau_window
                improvement_threshold = 0.0
                if best_mean != -float("inf"):
                    improvement_threshold = plateau_rel_delta * max(1.0, abs(best_mean))

                if best_mean == -float("inf") or rolling_mean > best_mean + improvement_threshold:
                    best_mean = rolling_mean
                    plateau_counter = 0
                else:
                    plateau_counter += 1

                if plateau_counter >= plateau_patience:
                    print(
                        f"[INFO] Early Stop (plateau); rolling_mean = {rolling_mean:.4f}, "
                        f"best_mean = {best_mean:.4f}, patience = {plateau_patience}"
                    )
                    break

        # Success-rate early stopping (kept for reference).
        # completed_episodes = (terminated | truncated).reshape(-1)
        # if completed_episodes.any():
        #     success_flags = base_env.goal.reshape(-1)[completed_episodes]
        #     episode_successes.extend(success_flags.float().tolist())
        #
        #     if i + 1 >= early_stop_start and len(episode_successes) >= success_window:
        #         rolling_success_rate = sum(episode_successes[-success_window:]) / success_window
        #
        #         if best_success_rate == -float("inf") or rolling_success_rate > best_success_rate + success_rate_delta:
        #             best_success_rate = rolling_success_rate
        #             success_plateau_counter = 0
        #         else:
        #             success_plateau_counter += 1
        #
        #         if success_plateau_counter >= success_patience:
        #             print(
        #                 f"[INFO] Early Stop (success plateau); rolling_success_rate = {rolling_success_rate:.4f}, "
        #                 f"best_success_rate = {best_success_rate:.4f}, patience = {success_patience}"
        #             )
        #             break

        if i % checkpoint_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{i}.pt"))

    # ADDED: Persist and print the final success-count summary.
    configured_num_envs = int(env_cfg.scene.num_envs)
    total_env_iterations = completed_iterations * configured_num_envs
    success_percentage = (
        100.0 * success_env_iteration_count / total_env_iterations if total_env_iterations > 0 else 0.0
    )
    _append_success_summary(
        log_dir,
        success_iteration_count,
        success_env_iteration_count,
        completed_iterations,
        configured_num_envs,
    )
    print(
        f"[INFO] Success count: {success_iteration_count} training iterations, "
        f"{success_env_iteration_count}/{total_env_iterations} env iterations "
        f"({success_percentage:.2f}%)"
    )

    #close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
