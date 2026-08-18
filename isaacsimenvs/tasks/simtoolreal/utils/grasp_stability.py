"""Pure result aggregation for simulated grasp-stability evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskStabilityGates:
    computed_feasible: bool
    grasp_survived: bool
    tracking_succeeded: bool
    contact_succeeded: bool

    @property
    def passed(self) -> bool:
        return all((
            self.computed_feasible,
            self.grasp_survived,
            self.tracking_succeeded,
            self.contact_succeeded,
        ))

    @property
    def failure_reasons(self) -> list[str]:
        return [
            name for name, passed in (
                ("computed_infeasible", self.computed_feasible),
                ("grasp_not_retained", self.grasp_survived),
                ("tracking_failed", self.tracking_succeeded),
                ("contact_failed", self.contact_succeeded),
            ) if not passed
        ]


def static_stable_fraction(*, failed: bool, failure_step: int, ramp_steps: int) -> float:
    if ramp_steps <= 0:
        raise ValueError("ramp_steps must be positive")
    if not failed:
        return 1.0
    if failure_step < 0:
        return 0.0
    return min(float(failure_step) / float(ramp_steps), 1.0)


def task_stability_gates(
    *, computed_feasible: bool, grasp_survived: bool,
    final_position_error_m: float, final_orientation_error_deg: float,
    geometric_contact_ratio: float, sensed_contact_ratio: float,
    position_tolerance_m: float, orientation_tolerance_deg: float,
    contact_ratio_threshold: float,
) -> TaskStabilityGates:
    if position_tolerance_m <= 0.0 or orientation_tolerance_deg <= 0.0:
        raise ValueError("tracking tolerances must be positive")
    if not 0.0 <= contact_ratio_threshold <= 1.0:
        raise ValueError("contact ratio threshold must be in [0, 1]")
    return TaskStabilityGates(
        computed_feasible=bool(computed_feasible),
        grasp_survived=bool(grasp_survived),
        tracking_succeeded=(
            final_position_error_m <= position_tolerance_m
            and final_orientation_error_deg <= orientation_tolerance_deg
        ),
        contact_succeeded=(
            geometric_contact_ratio >= contact_ratio_threshold
            and sensed_contact_ratio >= contact_ratio_threshold
        ),
    )


__all__ = ["TaskStabilityGates", "static_stable_fraction", "task_stability_gates"]
