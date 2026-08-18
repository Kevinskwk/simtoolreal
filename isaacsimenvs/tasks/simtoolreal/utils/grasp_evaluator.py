"""Deterministic trajectory-conditioned grasp evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ARM_JOINT_COUNT = 7
# Isaac Sim merges the fixed hand mount into link 7 and the grasp bank records
# this rigid-body frame, not the visual hand-root frame from the source URDF.
PALM_LINK_NAME = "iiwa14_link_7"


def _xyz(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    values = default if text is None else tuple(float(value) for value in text.split())
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError(f"expected a finite xyz triple, got {values}")
    return np.asarray(values, dtype=np.float64)


def pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("pose requires position (3,) and wxyz quaternion (4,)")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(position).all() or not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError("pose contains non-finite values or a non-unit quaternion")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = position
    return transform


def matrix_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return transform[:3, 3].copy(), xyzw[[3, 0, 1, 2]]


def pose_error(current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = target[:3, 3] - current[:3, 3]
    rotation = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    return translation, rotation


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


class UrdfKinematics:
    """Small URDF FK/Jacobian implementation for reproducible offline scoring."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        links = {element.get("name") for element in root.findall("link")}
        if None in links or not links:
            raise ValueError(f"URDF has invalid links: {self.urdf_path}")
        self.joints: dict[str, JointSpec] = {}
        self.child_to_joint: dict[str, JointSpec] = {}
        for element in root.findall("joint"):
            name = element.get("name")
            joint_type = element.get("type")
            parent_element, child_element = element.find("parent"), element.find("child")
            if not name or joint_type not in ("fixed", "revolute", "continuous"):
                continue
            if parent_element is None or child_element is None:
                raise ValueError(f"joint {name} has no parent or child")
            parent, child = parent_element.get("link"), child_element.get("link")
            if not parent or not child:
                raise ValueError(f"joint {name} has an invalid parent or child")
            origin_element = element.find("origin")
            xyz = _xyz(None if origin_element is None else origin_element.get("xyz"), (0, 0, 0))
            rpy = _xyz(None if origin_element is None else origin_element.get("rpy"), (0, 0, 0))
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            origin[:3, 3] = xyz
            axis_element = element.find("axis")
            axis = _xyz(None if axis_element is None else axis_element.get("xyz"), (1, 0, 0))
            axis_norm = np.linalg.norm(axis)
            if joint_type == "fixed":
                axis = np.array([1.0, 0.0, 0.0])
            elif axis_norm <= 0.0:
                raise ValueError(f"joint {name} has a zero axis")
            else:
                axis /= axis_norm
            limit = element.find("limit")
            if joint_type == "fixed":
                lower = upper = velocity = 0.0
            elif joint_type == "continuous":
                lower, upper = -math.pi, math.pi
                velocity = float("inf") if limit is None else float(limit.get("velocity", "inf"))
            else:
                if limit is None:
                    raise ValueError(f"revolute joint {name} has no limits")
                lower = float(limit.get("lower", "nan"))
                upper = float(limit.get("upper", "nan"))
                velocity = float(limit.get("velocity", "nan"))
            spec = JointSpec(name, parent, child, joint_type, origin, axis, lower, upper, velocity)
            self.joints[name] = spec
            if child in self.child_to_joint:
                raise ValueError(f"link {child} has multiple parent joints")
            self.child_to_joint[child] = spec
        children = set(self.child_to_joint)
        roots = links - children
        if len(roots) != 1:
            raise ValueError(f"URDF must contain one root link, got {sorted(roots)}")
        self.root_link = next(iter(roots))
        self.actuated_joint_names = tuple(
            element.get("name") for element in root.findall("joint")
            if element.get("type") in ("revolute", "continuous")
        )
        if len(self.actuated_joint_names) < ARM_JOINT_COUNT:
            raise ValueError("URDF has fewer than seven actuated joints")
        self.arm_joint_names = self.actuated_joint_names[:ARM_JOINT_COUNT]
        self.arm_lower = np.asarray([self.joints[name].lower for name in self.arm_joint_names])
        self.arm_upper = np.asarray([self.joints[name].upper for name in self.arm_joint_names])
        self.arm_velocity = np.asarray([self.joints[name].velocity for name in self.arm_joint_names])

    def chain(self, link_name: str) -> list[JointSpec]:
        chain: list[JointSpec] = []
        current = link_name
        while current != self.root_link:
            joint = self.child_to_joint.get(current)
            if joint is None:
                raise ValueError(f"link {link_name!r} is not connected to {self.root_link!r}")
            chain.append(joint)
            current = joint.parent
        chain.reverse()
        return chain

    @lru_cache(maxsize=None)
    def link_graph_distance(self, first: str, second: str) -> int:
        first_path = [self.root_link] + [joint.child for joint in self.chain(first)]
        second_path = [self.root_link] + [joint.child for joint in self.chain(second)]
        shared = 0
        for first_link, second_link in zip(first_path, second_path):
            if first_link != second_link:
                break
            shared += 1
        return (len(first_path) - shared) + (len(second_path) - shared)

    def forward(
        self,
        joint_positions: dict[str, float],
        *,
        base_transform: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
        base = np.eye(4) if base_transform is None else np.asarray(base_transform, dtype=np.float64)
        if base.shape != (4, 4):
            raise ValueError("base transform must be 4x4")
        link_transforms = {self.root_link: base.copy()}
        joint_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pending = list(self.joints.values())
        while pending:
            progressed = False
            for joint in pending[:]:
                parent = link_transforms.get(joint.parent)
                if parent is None:
                    continue
                before_motion = parent @ joint.origin
                axis_world = before_motion[:3, :3] @ joint.axis
                joint_frames[joint.name] = (before_motion[:3, 3].copy(), axis_world)
                motion = np.eye(4)
                if joint.joint_type != "fixed":
                    value = float(joint_positions.get(joint.name, 0.0))
                    motion[:3, :3] = Rotation.from_rotvec(joint.axis * value).as_matrix()
                link_transforms[joint.child] = before_motion @ motion
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("URDF joint graph is disconnected or cyclic")
        return link_transforms, joint_frames

    def palm_fk_jacobian(
        self, arm_q: np.ndarray, hand_q: np.ndarray, base_transform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        arm_q = np.asarray(arm_q, dtype=np.float64)
        hand_q = np.asarray(hand_q, dtype=np.float64)
        if arm_q.shape != (7,) or hand_q.shape != (len(self.actuated_joint_names) - 7,):
            raise ValueError("arm or hand configuration has the wrong shape")
        values = dict(zip(self.actuated_joint_names, np.concatenate((arm_q, hand_q)), strict=True))
        links, frames = self.forward(values, base_transform=base_transform)
        palm = links[PALM_LINK_NAME]
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for index, name in enumerate(self.arm_joint_names):
            position, axis = frames[name]
            jacobian[:3, index] = np.cross(axis, palm[:3, 3] - position)
            jacobian[3:, index] = axis
        return palm, jacobian, links


@dataclass(frozen=True)
class GraspEvaluatorThresholds:
    joint_violation_rad: float = 5.0e-4
    ik_position_m: float = 0.005
    ik_orientation_deg: float = 5.0
    velocity_ratio: float = 1.0
    penetration_tolerance_m: float = 5.0e-4
    max_ik_iterations: int = 200
    ik_damping: float = 0.03
    ik_step_limit_rad: float = 0.08


@dataclass(frozen=True)
class TrajectoryEvaluation:
    metrics: dict[str, float | int]
    gates: dict[str, bool]
    arm_trajectory: list[list[float]]
    failure_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def solve_arm_ik(
    kinematics: UrdfKinematics,
    target_palm: np.ndarray,
    initial_arm_q: np.ndarray,
    hand_q: np.ndarray,
    base_transform: np.ndarray,
    thresholds: GraspEvaluatorThresholds,
) -> tuple[np.ndarray, float, float, np.ndarray, dict[str, np.ndarray]]:
    q = np.asarray(initial_arm_q, dtype=np.float64).copy()
    for _ in range(thresholds.max_ik_iterations):
        palm, jacobian, links = kinematics.palm_fk_jacobian(q, hand_q, base_transform)
        position_error, orientation_error = pose_error(palm, target_palm)
        if (
            np.linalg.norm(position_error) <= thresholds.ik_position_m
            and np.linalg.norm(orientation_error) <= math.radians(thresholds.ik_orientation_deg)
        ):
            break
        error = np.concatenate((position_error, orientation_error))
        regularizer = thresholds.ik_damping**2 * np.eye(6)
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, error)
        delta = np.clip(delta, -thresholds.ik_step_limit_rad, thresholds.ik_step_limit_rad)
        q = np.clip(q + delta, kinematics.arm_lower, kinematics.arm_upper)
    palm, jacobian, links = kinematics.palm_fk_jacobian(q, hand_q, base_transform)
    position_error, orientation_error = pose_error(palm, target_palm)
    return (
        q,
        float(np.linalg.norm(position_error)),
        math.degrees(float(np.linalg.norm(orientation_error))),
        jacobian,
        links,
    )


def sphere_table_clearance(
    link_transforms: dict[str, np.ndarray], sphere_path: str | Path,
    table_point: np.ndarray, table_normal: np.ndarray,
) -> float:
    payload = _load_sphere_payload(str(Path(sphere_path).resolve()))
    minimum = float("inf")
    for link_name, geometry in payload.items():
        transform = link_transforms.get(link_name)
        if transform is None:
            raise ValueError(f"collision sphere link is absent from URDF FK: {link_name}")
        centers = np.asarray(geometry["centers"], dtype=np.float64)
        radii = np.asarray(geometry["radii"], dtype=np.float64)
        world = centers @ transform[:3, :3].T + transform[:3, 3]
        clearance = (world - table_point) @ table_normal - radii
        minimum = min(minimum, float(clearance.min()))
    return minimum


def sphere_box_clearance(
    link_transforms: dict[str, np.ndarray], sphere_path: str | Path,
    box_transform: np.ndarray, box_extent: np.ndarray,
) -> float:
    """Return minimum sphere-to-oriented-box signed distance."""
    payload = _load_sphere_payload(str(Path(sphere_path).resolve()))
    box_transform = np.asarray(box_transform, dtype=np.float64)
    half_extent = 0.5 * np.asarray(box_extent, dtype=np.float64)
    if box_transform.shape != (4, 4) or half_extent.shape != (3,) or bool((half_extent <= 0.0).any()):
        raise ValueError("box transform or extent is invalid")
    minimum = float("inf")
    for link_name, geometry in payload.items():
        transform = link_transforms.get(link_name)
        if transform is None:
            raise ValueError(f"collision sphere link is absent from URDF FK: {link_name}")
        centers = np.asarray(geometry["centers"], dtype=np.float64)
        radii = np.asarray(geometry["radii"], dtype=np.float64)
        world = centers @ transform[:3, :3].T + transform[:3, 3]
        local = (world - box_transform[:3, 3]) @ box_transform[:3, :3]
        q = np.abs(local) - half_extent
        signed = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(
            np.max(q, axis=-1), 0.0
        ) - radii
        minimum = min(minimum, float(signed.min()))
    return minimum


def sphere_self_clearance(
    kinematics: UrdfKinematics, link_transforms: dict[str, np.ndarray],
    sphere_path: str | Path, *, excluded_graph_distance: int = 2,
) -> float:
    """Return minimum clearance between non-neighboring collision spheres."""
    payload = _load_sphere_payload(str(Path(sphere_path).resolve()))
    world: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for link_name, geometry in payload.items():
        transform = link_transforms.get(link_name)
        if transform is None:
            raise ValueError(f"collision sphere link is absent from URDF FK: {link_name}")
        centers = np.asarray(geometry["centers"], dtype=np.float64)
        radii = np.asarray(geometry["radii"], dtype=np.float64)
        world[link_name] = (
            centers @ transform[:3, :3].T + transform[:3, 3], radii
        )
    minimum = float("inf")
    names = list(world)
    for first_index, first_name in enumerate(names):
        first_centers, first_radii = world[first_name]
        for second_name in names[first_index + 1:]:
            if kinematics.link_graph_distance(first_name, second_name) <= excluded_graph_distance:
                continue
            second_centers, second_radii = world[second_name]
            distances = np.linalg.norm(
                first_centers[:, None, :] - second_centers[None, :, :], axis=-1
            ) - first_radii[:, None] - second_radii[None, :]
            minimum = min(minimum, float(distances.min()))
    if not math.isfinite(minimum):
        raise ValueError("self-collision sphere model contains no testable pairs")
    return minimum


@lru_cache(maxsize=8)
def _load_sphere_payload(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"collision sphere file is empty or invalid: {path}")
    return payload


def tool_box_table_clearance(
    tool_pose: np.ndarray, bounds: tuple[float, float, float, float, float, float],
    table_point: np.ndarray, table_normal: np.ndarray,
) -> float:
    lo, hi = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    corners = np.asarray([
        [x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])
    ])
    world = corners @ tool_pose[:3, :3].T + tool_pose[:3, 3]
    return float(((world - table_point) @ table_normal).min())


def evaluate_grasp_trajectory(
    *,
    kinematics: UrdfKinematics,
    sphere_path: str | Path,
    joint_positions: np.ndarray,
    palm_to_tool: np.ndarray,
    tool_trajectory: np.ndarray,
    dt: float,
    table_transform: np.ndarray,
    table_extent: np.ndarray,
    tool_bounds: tuple[float, float, float, float, float, float],
    base_transform: np.ndarray,
    thresholds: GraspEvaluatorThresholds = GraspEvaluatorThresholds(),
) -> TrajectoryEvaluation:
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    tool_trajectory = np.asarray(tool_trajectory, dtype=np.float64)
    if joint_positions.shape != (29,) or tool_trajectory.ndim != 3 or tool_trajectory.shape[1:] != (4, 4):
        raise ValueError("expected 29 joints and a (T, 4, 4) tool trajectory")
    if len(tool_trajectory) < 2 or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("trajectory needs at least two poses and positive dt")
    table_transform = np.asarray(table_transform, dtype=np.float64)
    if table_transform.shape != (4, 4):
        raise ValueError("table transform must be 4x4")
    table_point = table_transform[:3, 3] + table_transform[:3, 2] * 0.5 * float(table_extent[2])
    normal = table_transform[:3, 2].copy()
    normal_norm = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError("table normal is invalid")
    normal /= normal_norm
    arm_q, hand_q = joint_positions[:7].copy(), joint_positions[7:].copy()
    initial_violation = float(np.maximum(
        kinematics.arm_lower - arm_q, arm_q - kinematics.arm_upper
    ).clip(min=0.0).max())
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    min_singular_values: list[float] = []
    condition_numbers: list[float] = []
    manipulabilities: list[float] = []
    robot_clearances: list[float] = []
    self_clearances: list[float] = []
    tool_clearances: list[float] = []
    arm_trajectory: list[np.ndarray] = []
    waypoint_feasible: list[bool] = []
    for tool_pose in tool_trajectory:
        target_palm = tool_pose @ np.linalg.inv(palm_to_tool)
        arm_q, pos_error, rot_error, jacobian, links = solve_arm_ik(
            kinematics, target_palm, arm_q, hand_q, base_transform, thresholds
        )
        singular = np.linalg.svd(jacobian, compute_uv=False)
        position_errors.append(pos_error)
        orientation_errors.append(rot_error)
        min_singular_values.append(float(singular[-1]))
        condition_numbers.append(float(singular[0] / max(singular[-1], 1e-12)))
        manipulabilities.append(float(np.prod(singular)))
        robot_clearances.append(sphere_box_clearance(
            links, sphere_path, table_transform, np.asarray(table_extent)
        ))
        self_clearances.append(sphere_self_clearance(kinematics, links, sphere_path))
        tool_clearances.append(tool_box_table_clearance(
            tool_pose, tool_bounds, np.asarray(table_point), normal
        ))
        feasible = (
            pos_error <= thresholds.ik_position_m
            and rot_error <= thresholds.ik_orientation_deg
            and robot_clearances[-1] >= -thresholds.penetration_tolerance_m
            and self_clearances[-1] >= -thresholds.penetration_tolerance_m
            and tool_clearances[-1] >= -thresholds.penetration_tolerance_m
        )
        waypoint_feasible.append(feasible)
        arm_trajectory.append(arm_q.copy())
    arm_array = np.asarray(arm_trajectory)
    velocities = np.diff(arm_array, axis=0) / dt
    velocity_ratio = np.abs(velocities) / kinematics.arm_velocity
    accelerations = np.diff(velocities, axis=0) / dt if len(velocities) > 1 else np.zeros((0, 7))
    max_velocity_ratio = float(velocity_ratio.max(initial=0.0))
    min_joint_margin = float(np.minimum(
        arm_array - kinematics.arm_lower, kinematics.arm_upper - arm_array
    ).min())
    gates = {
        "joint_limits": initial_violation <= thresholds.joint_violation_rad and min_joint_margin >= -thresholds.joint_violation_rad,
        "ik_position": max(position_errors) <= thresholds.ik_position_m,
        "ik_orientation": max(orientation_errors) <= thresholds.ik_orientation_deg,
        "arm_velocity": max_velocity_ratio <= thresholds.velocity_ratio,
        "robot_environment_collision": min(robot_clearances) >= -thresholds.penetration_tolerance_m,
        "robot_self_collision": min(self_clearances) >= -thresholds.penetration_tolerance_m,
        "tool_environment_collision": min(tool_clearances) >= -thresholds.penetration_tolerance_m,
        "trajectory_coverage": all(waypoint_feasible),
    }
    failures = [name for name, passed in gates.items() if not passed]
    metrics: dict[str, float | int] = {
        "waypoint_count": len(tool_trajectory),
        "feasible_waypoint_fraction": float(np.mean(waypoint_feasible)),
        "initial_joint_violation_rad": initial_violation,
        "minimum_joint_margin_rad": min_joint_margin,
        "maximum_ik_position_error_m": max(position_errors),
        "mean_ik_position_error_m": float(np.mean(position_errors)),
        "maximum_ik_orientation_error_deg": max(orientation_errors),
        "mean_ik_orientation_error_deg": float(np.mean(orientation_errors)),
        "maximum_arm_velocity_ratio": max_velocity_ratio,
        "maximum_arm_acceleration_rad_s2": float(np.abs(accelerations).max(initial=0.0)),
        "minimum_jacobian_singular_value": min(min_singular_values),
        "maximum_jacobian_condition_number": max(condition_numbers),
        "minimum_yoshikawa_manipulability": min(manipulabilities),
        "minimum_robot_table_clearance_m": min(robot_clearances),
        "minimum_robot_self_clearance_m": min(self_clearances),
        "minimum_tool_table_clearance_m": min(tool_clearances),
        "palm_to_functional_edge_clearance_m": float(
            bounds_edge_clearance(palm_to_tool, tool_bounds)
        ),
    }
    return TrajectoryEvaluation(
        metrics=metrics,
        gates=gates,
        arm_trajectory=arm_array.tolist(),
        failure_reasons=failures,
    )


def grasp_fingerprint(entry: dict) -> str:
    import hashlib

    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bounds_edge_clearance(
    palm_to_tool: np.ndarray,
    tool_bounds: tuple[float, float, float, float, float, float],
) -> float:
    """Distance along local X from the palm origin to the functional +X edge."""
    palm_in_tool = np.linalg.inv(np.asarray(palm_to_tool, dtype=np.float64))[:3, 3]
    return float(tool_bounds[3] - palm_in_tool[0])


__all__ = [
    "GraspEvaluatorThresholds", "TrajectoryEvaluation", "UrdfKinematics",
    "bounds_edge_clearance", "evaluate_grasp_trajectory", "grasp_fingerprint", "matrix_pose", "pose_matrix",
    "solve_arm_ik", "sphere_table_clearance", "tool_box_table_clearance",
    "sphere_box_clearance", "sphere_self_clearance",
]
