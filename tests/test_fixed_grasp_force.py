from pathlib import Path
import ast

import torch


def _load_pure_functions():
    path = (
        Path(__file__).resolve().parents[1]
        / "isaacsimenvs/tasks/simtoolreal/simtoolreal_fixed_grasp_force_env.py"
    )
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and (
            not isinstance(node, ast.FunctionDef)
            or node.name
            in {
                "fixed_force_observation_lists",
                "fixed_force_reward_terms",
                "integrate_normal_action",
            }
        )
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch, "contact_force_reward": _contact_force_reward}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _contact_force_reward(normal_force, target_force, force_sigma, max_force):
    error = torch.abs(normal_force - target_force)
    return torch.exp(-error / force_sigma), torch.clamp(normal_force - max_force, min=0.0)


_functions = _load_pure_functions()
fixed_force_observation_lists = _functions["fixed_force_observation_lists"]
fixed_force_reward_terms = _functions["fixed_force_reward_terms"]
integrate_normal_action = _functions["integrate_normal_action"]


def test_feedback_modes_have_expected_actor_and_critic_dimensions():
    force, critic = fixed_force_observation_lists("force")
    blind, blind_critic = fixed_force_observation_lists("blind")
    tactile, tactile_critic = fixed_force_observation_lists("tactile")
    assert len(force) == 6
    assert len(blind) == 4
    assert tactile[:-1] == blind and tactile[-1] == "tacmap"
    assert critic == blind_critic == tactile_critic == force


def test_unknown_feedback_mode_fails_loudly():
    try:
        fixed_force_observation_lists("invalid")
    except ValueError as exc:
        assert "force/tactile/blind" in str(exc)
    else:
        raise AssertionError("invalid feedback mode was accepted")


def test_force_reward_is_maximal_at_target_and_penalizes_action_changes():
    measured = torch.tensor([4.0, 2.0])
    target = torch.tensor([4.0, 4.0])
    reward, terms = fixed_force_reward_terms(
        measured,
        target,
        torch.tensor([0.0, 1.0]),
        sigma=2.0,
        soft_force_limit=12.0,
        max_force=20.0,
        force_weight=1.0,
        over_force_weight=0.1,
        action_rate_weight=0.01,
    )
    assert reward[0] == 1.0
    assert reward[1] < torch.exp(torch.tensor(-1.0))
    assert terms["action_rate_penalty"][1] == -0.01


def test_soft_over_force_penalty_starts_above_soft_limit():
    _, terms = fixed_force_reward_terms(
        torch.tensor([12.0, 16.0]),
        torch.tensor([6.0, 6.0]),
        torch.zeros(2),
        sigma=2.0,
        soft_force_limit=12.0,
        max_force=20.0,
        force_weight=1.0,
        over_force_weight=0.1,
        action_rate_weight=0.01,
    )
    assert terms["over_force_penalty"][0] == 0.0
    assert terms["over_force_penalty"][1] < 0.0


def test_normal_action_integration_scales_and_clamps_velocity():
    next_offset, velocity = integrate_normal_action(
        torch.tensor([0.0, 0.049, -0.009]),
        torch.tensor([0.5, 2.0, -2.0]),
        velocity_limit=0.01,
        step_dt=0.2,
        offset_min=-0.01,
        offset_max=0.05,
    )
    assert torch.allclose(velocity, torch.tensor([0.005, 0.01, -0.01]))
    assert torch.allclose(next_offset, torch.tensor([0.001, 0.05, -0.01]))
