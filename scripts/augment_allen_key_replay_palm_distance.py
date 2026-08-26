#!/usr/bin/env python3
"""Add physical-palm/tool frames and live palm-handle distance to a replay."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from trimesh.transformations import quaternion_from_matrix, quaternion_matrix
from yourdfpy import URDF

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = (
    ROOT / "isaacsimenvs/utils/interactive_viewer/index.template.html"
)
ROBOT_URDF = (
    ROOT
    / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)
PALM_MESH_CENTER_LOCAL = np.array(
    [0.0003402513, -0.0010913294, 0.0432048663, 1.0], dtype=float
)
SCENE_PATTERN = re.compile(
    r'<script id="scene-json" type="application/json">(.*?)</script>', re.DOTALL
)


def render_template(scene: dict) -> str:
    return DEFAULT_TEMPLATE_PATH.read_text().replace(
        "__SCENE_JSON__", json.dumps(scene, separators=(",", ":"))
    )


def marker_urdf(name: str, color: str, *, sphere: bool = False) -> str:
    geometry = '<sphere radius="0.012"/>' if sphere else '<box size="0.065 0.045 0.008"/>'
    axes = "" if sphere else """
    <visual><origin xyz="0.035 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.0025" length="0.07"/></geometry><material name="x"><color rgba="1 0.1 0.1 1"/></material></visual>
    <visual><origin xyz="0 0.035 0" rpy="-1.5707963 0 0"/><geometry><cylinder radius="0.0025" length="0.07"/></geometry><material name="y"><color rgba="0.1 1 0.1 1"/></material></visual>
    <visual><origin xyz="0 0 0.035"/><geometry><cylinder radius="0.0025" length="0.07"/></geometry><material name="z"><color rgba="0.1 0.3 1 1"/></material></visual>
"""
    return f"""<?xml version="1.0"?>
<robot name="{name}"><link name="{name}">
  <visual><geometry>{geometry}</geometry><material name="body"><color rgba="{color}"/></material></visual>
  {axes}
</link></robot>"""


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    transform = quaternion_matrix(np.roll(pose[3:], 1))
    transform[:3, 3] = pose[:3]
    return transform


def matrix_pose(transform: np.ndarray) -> list[float]:
    quaternion_wxyz = quaternion_from_matrix(transform)
    return [
        *transform[:3, 3].tolist(),
        quaternion_wxyz[1],
        quaternion_wxyz[2],
        quaternion_wxyz[3],
        quaternion_wxyz[0],
    ]


def embedded_object_urdf(scene: dict) -> str:
    matches = [robot.get("urdf_text") for robot in scene["robots"] if robot["name"] == "object"]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError("replay must contain exactly one embedded object URDF")
    return matches[0]


def handle_center_local(urdf_text: str) -> np.ndarray:
    root = ET.fromstring(urdf_text)
    for visual in root.findall(".//visual"):
        material = visual.find("material")
        if material is None or material.get("name") != "grip":
            continue
        origin = visual.find("origin")
        if origin is None or not origin.get("xyz"):
            raise RuntimeError("Allen-key grip visual has no origin")
        xyz = np.asarray([float(value) for value in origin.get("xyz").split()])
        return np.r_[xyz, 1.0]
    raise RuntimeError("embedded object URDF has no Allen-key grip visual")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-html", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()

    match = SCENE_PATTERN.search(args.input_html.read_text())
    if match is None:
        raise RuntimeError("input HTML has no embedded scene JSON")
    scene = json.loads(match.group(1))
    trajectory = scene["trajectory"]
    joint_names = trajectory["joint_names"]
    joint_positions = np.asarray(trajectory["positions"], dtype=float)
    base_positions = np.asarray(trajectory["base_trajectory"]["positions"], dtype=float)
    base_quaternions = np.asarray(trajectory["base_trajectory"]["quats"], dtype=float)
    object_positions = np.asarray(
        trajectory["object_trajectories"]["object"]["positions"], dtype=float
    )
    object_quaternions = np.asarray(
        trajectory["object_trajectories"]["object"]["quats"], dtype=float
    )
    frame_count = len(joint_positions)
    arrays = (base_positions, base_quaternions, object_positions, object_quaternions)
    if any(len(array) != frame_count for array in arrays):
        raise RuntimeError("replay trajectory arrays have inconsistent lengths")

    robot = URDF.load(ROBOT_URDF, load_meshes=False)
    if set(joint_names) != set(robot.actuated_joint_names):
        raise RuntimeError("replay joints do not match the SHARPA robot URDF")
    handle_local = handle_center_local(embedded_object_urdf(scene))
    palm_poses = []
    handle_poses = []
    tool_poses = []
    distances = []
    for index in range(frame_count):
        robot.update_cfg(dict(zip(joint_names, joint_positions[index], strict=True)))
        base_pose = np.r_[base_positions[index], base_quaternions[index]]
        world_from_base = pose_matrix(base_pose)
        base_from_palm_link = robot.get_transform("left_hand_C_MC")
        world_from_palm = world_from_base @ base_from_palm_link
        palm_center = world_from_palm @ PALM_MESH_CENTER_LOCAL
        world_from_palm[:3, 3] = palm_center[:3]

        tool_pose = np.r_[object_positions[index], object_quaternions[index]]
        world_from_tool = pose_matrix(tool_pose)
        handle_center = world_from_tool @ handle_local
        world_from_handle = world_from_tool.copy()
        world_from_handle[:3, 3] = handle_center[:3]

        palm_poses.append(matrix_pose(world_from_palm))
        handle_poses.append(matrix_pose(world_from_handle))
        tool_poses.append(tool_pose.tolist())
        distances.append(float(np.linalg.norm(palm_center[:3] - handle_center[:3])))

    scene["robots"].extend([
        {"name": "current_palm", "urdf_text": marker_urdf("current_palm", "0.05 0.9 1 0.75"), "position": [0, 0, 0], "rpy": [0, 0, 0], "animated": False},
        {"name": "tool_frame", "urdf_text": marker_urdf("tool_frame", "1.0 0.55 0.05 0.85"), "position": [0, 0, 0], "rpy": [0, 0, 0], "animated": False},
        {"name": "handle_center", "urdf_text": marker_urdf("handle_center", "1.0 0.9 0.05 0.95", sphere=True), "position": [0, 0, 0], "rpy": [0, 0, 0], "animated": False},
    ])
    trajectory["object_trajectories"].update({
        "current_palm": {"positions": [pose[:3] for pose in palm_poses], "quats": [pose[3:] for pose in palm_poses]},
        "tool_frame": {"positions": [pose[:3] for pose in tool_poses], "quats": [pose[3:] for pose in tool_poses]},
        "handle_center": {"positions": [pose[:3] for pose in handle_poses], "quats": [pose[3:] for pose in handle_poses]},
    })
    trajectory["frame_scalars"] = {"Palm-to-handle center (m)": distances}
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render_template(scene))
    print(json.dumps({
        "output": str(args.output_html.resolve()),
        "frames": frame_count,
        "distance_min_m": min(distances),
        "distance_mean_m": float(np.mean(distances)),
        "distance_max_m": max(distances),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
