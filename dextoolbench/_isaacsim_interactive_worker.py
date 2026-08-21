"""Isaac Sim child-process worker for the DexToolBench interactive demo.

Lives in its own module with stdlib-only top-level imports: multiprocessing
``spawn`` re-imports the target function's module in the child, and Kit
segfaults at boot if heavy C extensions (viser's trimesh/embreex stack from
eval_interactive) are already loaded. Keep this module import-clean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_Z = 0.38
Z_OFFSET = 0.03
CONTROL_DT = 1.0 / 60.0


def _checkpoint_observation_dim(checkpoint_path: str) -> int:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = payload.get(0, payload)
    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError(f"checkpoint has no model state dict: {checkpoint_path}")
    dims = {
        int(value.numel())
        for key, value in model.items()
        if key.endswith("running_mean_std.running_mean") and value.ndim == 1
    }
    if len(dims) != 1:
        raise RuntimeError(
            "could not infer exactly one actor observation dimension from "
            f"{checkpoint_path}; found {sorted(dims)}"
        )
    return dims.pop()


def _write_isaac_trajectory(goals_xyzw, out_dir: Path) -> str:
    """gym-format goals ([x,y,z,qx,qy,qz,qw] list) -> fixed_trajectory_file
    format (pos (1,K,3), quat wxyz (1,K,4))."""
    pos = [[g[:3] for g in goals_xyzw]]
    quat_wxyz = [[[g[6], g[3], g[4], g[5]] for g in goals_xyzw]]
    path = out_dir / "trajectory_isaac_format.json"
    with open(path, "w") as f:
        json.dump({"pos": pos, "quat_wxyz": quat_wxyz}, f)
    return str(path)


def _sim_get_state(inner, obs):
    """Visualisation state (joint_pos, object pose, goal pose) in the format
    the gym worker sends: poses env-local, quats xyzw."""
    import numpy as np

    obs_np = obs["policy"][0].detach().cpu().numpy()
    lower = inner._joint_lower_canon.detach().cpu().numpy()
    upper = inner._joint_upper_canon.detach().cpu().numpy()
    joint_pos = 0.5 * (obs_np[:29] + 1.0) * (upper - lower) + lower

    origin = inner.scene.env_origins[0].detach().cpu().numpy()

    def _pose7(rigid):
        pos = rigid.data.root_pos_w[0].detach().cpu().numpy() - origin
        wxyz = rigid.data.root_quat_w[0].detach().cpu().numpy()
        xyzw = wxyz[[1, 2, 3, 0]]
        return np.concatenate([pos, xyzw]).astype(np.float32)

    return joint_pos, _pose7(inner.object), _pose7(inner.goal_viz)


def _write_manual_goal(inner, pose_xyzw):
    """Write one unrestricted env-local goal pose into the Isaac Lab scene."""
    import torch

    pose = torch.as_tensor(pose_xyzw, device=inner.device, dtype=torch.float32)
    if pose.shape != (7,) or not bool(torch.isfinite(pose).all()):
        raise ValueError(f"manual goal must be one finite 7D pose, got {pose_xyzw!r}")
    quat_xyzw = pose[3:]
    quat_norm = torch.linalg.vector_norm(quat_xyzw)
    if float(quat_norm.item()) < 1.0e-8:
        raise ValueError("manual goal quaternion has zero norm")
    quat_xyzw = quat_xyzw / quat_norm
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    world_pos = pose[:3] + inner.scene.env_origins[0]
    world_pose = torch.cat((world_pos, quat_wxyz)).unsqueeze(0)
    env_id = torch.zeros(1, device=inner.device, dtype=torch.long)
    inner.goal_viz.write_root_pose_to_sim(world_pose, env_ids=env_id)
    inner.goal_viz.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=inner.device), env_ids=env_id
    )


def _set_manual_goal_mode(inner, enabled: bool, n_goals: int) -> None:
    term = inner.cfg.termination
    term.success_steps = 1_000_000_000 if enabled else 1
    term.max_consecutive_successes = 0 if enabled else n_goals


def _restore_task_trajectory(inner, n_goals: int) -> None:
    import torch

    from isaacsimenvs.tasks.simtoolreal.utils.reset_utils import (
        _clear_goal_trackers,
        _reset_goal_pose,
    )

    _set_manual_goal_mode(inner, False, n_goals)
    env_id = torch.zeros(1, device=inner.device, dtype=torch.long)
    _reset_goal_pose(inner, env_id, mode="absolute")
    _clear_goal_trackers(inner, env_id)
    inner._successes[env_id] = 0


def _sim_episode(
    conn, env, inner, player, n_act, n_goals, rl_device, manual_goal_pose
):
    """Run one episode, streaming state to the parent via *conn*."""
    import time

    import torch

    player.player.init_rnn()
    obs, _ = env.reset()
    if manual_goal_pose is not None:
        _write_manual_goal(inner, manual_goal_pose)
    obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))

    step, done, paused = 0, False, False
    goals_reached = 0

    while not done:
        while conn.poll(0):
            cmd = conn.recv()
            if isinstance(cmd, tuple) and cmd[0] == "set_goal":
                manual_goal_pose = cmd[1]
                _set_manual_goal_mode(inner, True, n_goals)
                _write_manual_goal(inner, manual_goal_pose)
                conn.send(("goal_set", manual_goal_pose))
            elif cmd == "clear_goal":
                manual_goal_pose = None
                _restore_task_trajectory(inner, n_goals)
                conn.send(("trajectory_restored",))
            elif cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return manual_goal_pose

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()

        state = _sim_get_state(inner, obs)
        policy_obs = obs["policy"].to(rl_device)
        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
        done = bool(terminated[0].item() or truncated[0].item())
        if done:
            goals_reached = max(goals_reached, int(inner._prev_episode_successes[0].item()))
        else:
            goals_reached = max(goals_reached, int(inner._successes[0].item()))
        step += 1

        displayed_goal_count = 0 if manual_goal_pose is not None else n_goals
        conn.send(("state", state, goals_reached, displayed_goal_count, step))

        elapsed = time.time() - t0
        if (sleep := CONTROL_DT - elapsed) > 0:
            time.sleep(sleep)

    goal_pct = 100 * goals_reached / n_goals
    conn.send(("done", goal_pct, step))
    return manual_goal_pose


def sim_worker_isaacsim(conn, category, object_name, task_name, table_urdf,
                        config_path, checkpoint_path, table_height):
    """Child process entry-point: boots Kit, creates the env, waits for commands."""
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        launcher_args = parser.parse_args([])
        launcher_args.headless = True
        app = AppLauncher(launcher_args).app  # noqa: F841

        import gymnasium as gym
        import torch

        import isaacsimenvs  # noqa: F401  registers gym envs
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
        from deployment.rl_player import RlPlayer
        from dextoolbench.objects import NAME_TO_OBJECT

        rl_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Trajectory (gym format) -> fixed_trajectory_file (isaac format).
        traj_path = (
            REPO_ROOT / "dextoolbench" / "trajectories"
            / category / object_name / f"{task_name}.json"
        )
        with open(traj_path) as f:
            traj_data = json.load(f)
        table_delta = float(table_height) - TABLE_Z
        traj_data["start_pose"][2] += Z_OFFSET + table_delta
        for goal in traj_data["goals"]:
            goal[2] += table_delta
        n_goals = len(traj_data["goals"])
        tmp_dir = Path(tempfile.mkdtemp(prefix="dextoolbench_interactive_"))
        traj_file = _write_isaac_trajectory(traj_data["goals"], tmp_dir)

        obj = NAME_TO_OBJECT[object_name]

        cfg = SimToolRealEnvCfg()
        cfg.scene.num_envs = 1
        cfg.assets.object_urdf = str(obj.decomposed_urdf_path)
        cfg.assets.object_scale = tuple(obj.scale)
        # table_urdf arrives relative to assets/ (gym convention).
        cfg.assets.table_urdf = str(REPO_ROOT / "assets" / table_urdf)

        rs = cfg.reset
        rs.reset_position_noise_x = 0.0
        rs.reset_position_noise_y = 0.0
        rs.reset_position_noise_z = 0.0
        rs.reset_dof_pos_random_interval_arm = 0.0
        rs.reset_dof_pos_random_interval_fingers = 0.0
        rs.reset_dof_vel_random_interval = 0.0
        rs.table_reset_z = float(table_height)
        rs.table_reset_z_range = 0.0
        rs.start_arm_higher = True
        sp = traj_data["start_pose"]
        rs.fixed_start_pose = (sp[0], sp[1], sp[2], sp[6], sp[3], sp[4], sp[5])
        rs.fixed_trajectory_file = traj_file

        dr = cfg.domain_randomization
        dr.use_obs_delay = False
        dr.use_action_delay = False
        dr.use_object_state_delay_noise = False
        dr.object_scale_noise_multiplier_range = (1.0, 1.0)
        dr.joint_velocity_obs_noise_std = 0.0
        dr.force_scale = 0.0
        dr.torque_scale = 0.0
        dr.force_prob_range = (0.0001, 0.0001)
        dr.torque_prob_range = (0.0001, 0.0001)

        term = cfg.termination
        term.eval_success_tolerance = 0.01
        term.success_steps = 1
        term.max_consecutive_successes = n_goals
        cfg.episode_length_s = 3600.0

        env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        inner._replay_target_lab_order = None

        checkpoint_obs_dim = _checkpoint_observation_dim(checkpoint_path)
        if checkpoint_obs_dim != int(inner.cfg.observation_space):
            raise RuntimeError(
                f"selected policy expects {checkpoint_obs_dim} actor observations, "
                f"but the interactive base environment produces "
                f"{inner.cfg.observation_space}. Select a compatible no-tactile "
                "pose-policy checkpoint; tactile or stable-scrape checkpoints need "
                "their task-specific evaluator."
            )

        n_act = cfg.action_space
        player = RlPlayer(
            num_observations=inner.cfg.observation_space,
            num_actions=n_act,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=rl_device,
            num_envs=1,
        )

        player.player.init_rnn()
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))
        init_state = _sim_get_state(inner, obs)

        conn.send(("ready", init_state))

        manual_goal_pose = None
        while True:
            cmd = conn.recv()
            if cmd == "run":
                manual_goal_pose = _sim_episode(
                    conn, env, inner, player, n_act, n_goals, rl_device,
                    manual_goal_pose,
                )
            elif isinstance(cmd, tuple) and cmd[0] == "set_goal":
                manual_goal_pose = cmd[1]
                _set_manual_goal_mode(inner, True, n_goals)
                _write_manual_goal(inner, manual_goal_pose)
                conn.send(("goal_set", manual_goal_pose))
            elif cmd == "clear_goal":
                manual_goal_pose = None
                _restore_task_trajectory(inner, n_goals)
                conn.send(("trajectory_restored",))
            elif cmd == "quit":
                break

    except Exception as exc:
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except (BrokenPipeError, OSError):
            pass

    try:
        conn.close()
    except OSError:
        pass
    # Skip Kit teardown (it hangs); the parent already has everything it needs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    # Standalone entry point: launched via subprocess by
    # eval_interactive_isaacsim.py (Kit segfaults when booted inside a
    # multiprocessing-spawned child, so we use a plain process + socket).
    from multiprocessing.connection import Client

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True, help="host:port of the parent Listener")
    parser.add_argument("--authkey", required=True, help="hex auth key")
    parser.add_argument("--category", required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--table_urdf", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--table_height", type=float, required=True)
    cli = parser.parse_args()

    host, port_str = cli.address.rsplit(":", 1)
    conn = Client((host, int(port_str)), authkey=bytes.fromhex(cli.authkey))
    sim_worker_isaacsim(
        conn, cli.category, cli.object_name, cli.task_name, cli.table_urdf,
        cli.config_path, cli.checkpoint_path, cli.table_height,
    )
