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

    episode_cumulative = dict(env._reward_terms)
    extra_cumulative = getattr(env, "_episode_cumulative_terms", {})
    duplicate = set(episode_cumulative).intersection(extra_cumulative)
    if duplicate:
        raise RuntimeError(
            "duplicate episode cumulative metric names: "
            f"{sorted(duplicate)}"
        )
    episode_cumulative.update(extra_cumulative)
    reward = env._reward_terms.get("total_reward")
    if reward is None:
        raise RuntimeError("reward terms must include total_reward")
    episode_cumulative["episode_step_count"] = torch.ones_like(reward)
    env.extras["episode_cumulative"] = episode_cumulative
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)
    env.extras["curriculum/success_tolerance"] = float(env._current_success_tolerance)
    env.extras["curriculum/current_success_tolerance"] = float(
        env._current_success_tolerance
    )
    env.extras["curriculum/success_mean"] = float(
        getattr(env, "_curriculum_success_mean", 0.0)
    )

    for name, value in env._reward_terms.items():
        env.extras[f"reward/{name}"] = value.float().mean()


__all__ = ["log_step_metrics"]
