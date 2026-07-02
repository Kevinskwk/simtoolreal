"""Task metric publishing for SimToolReal."""

from __future__ import annotations

import torch


def log_step_metrics(env) -> None:
    """Publish step-level extras consumed by RL-Games observers."""
    term_cfg = env.cfg.termination
    if term_cfg.max_consecutive_successes > 0:
        all_goals_hit = env._successes >= term_cfg.max_consecutive_successes
    else:
        all_goals_hit = torch.zeros_like(env._successes, dtype=torch.bool)

    episode_final = {
        "successes": env._successes.float(),
        "all_goals_hit": all_goals_hit.float(),
    }
    episode_final.update(
        {
            f"done_{name}": value.float()
            for name, value in env._termination_reasons.items()
        }
    )

    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)
    env.extras["curriculum/current_success_tolerance"] = float(
        env._current_success_tolerance
    )
    env.extras["curriculum/start_success_tolerance"] = float(
        term_cfg.success_tolerance
    )
    env.extras["curriculum/target_success_tolerance"] = float(
        term_cfg.target_success_tolerance
    )
    env.extras["curriculum/success_mean"] = float(
        getattr(env, "_curriculum_success_mean", 0.0)
    )
    default_success_threshold = term_cfg.tolerance_curriculum_success_threshold
    env.extras["curriculum/success_threshold"] = float(
        getattr(env, "_curriculum_success_threshold_value", default_success_threshold)
    )
    env.extras["curriculum/eligible_count"] = float(
        getattr(env, "_curriculum_eligible_count", env.num_envs)
    )
    env.extras["curriculum/updated_this_step"] = float(
        getattr(env, "_curriculum_updated_this_step", False)
    )
    env.extras["curriculum/update_count"] = float(
        getattr(env, "_curriculum_update_count", 0)
    )
    env.extras["curriculum/frames_since_update"] = float(
        env._frame_counter - env._last_curriculum_update
    )
    env.extras["curriculum/frame_counter"] = float(env._frame_counter)


__all__ = ["log_step_metrics"]
