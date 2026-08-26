"""Geometric palm pose helpers for the fixed-link SHARPA hand import."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_mul


# Isaac Sim merges iiwa14_link_ee, sharpa_mount, and left_hand_C_MC into
# iiwa14_link_7. This is the center of the left_hand_C_MC visual-mesh AABB,
# transformed through that fixed URDF chain into the surviving link-7 frame.
PALM_CENTER_IN_IIWA_LINK_7_M = (-0.0009660822, -0.0006111104, 0.1382048663)
PALM_LINK_QUAT_IN_IIWA_LINK_7_WXYZ = (
    0.7933521042, 0.0, 0.0, -0.6087630399
)


def palm_center_pose_from_merged_wrist(
    wrist_position_w: torch.Tensor,
    wrist_quaternion_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the physical palm-center pose from merged link-7 poses."""
    if wrist_position_w.shape[-1] != 3 or wrist_quaternion_w.shape[-1] != 4:
        raise ValueError("wrist pose has invalid shape")
    if wrist_position_w.shape[:-1] != wrist_quaternion_w.shape[:-1]:
        raise ValueError("wrist position and quaternion shapes do not match")
    if not bool(torch.isfinite(wrist_position_w).all()) or not bool(
        torch.isfinite(wrist_quaternion_w).all()
    ):
        raise ValueError("wrist pose contains NaN or Inf")
    offset = wrist_position_w.new_tensor(PALM_CENTER_IN_IIWA_LINK_7_M)
    local_quaternion = wrist_quaternion_w.new_tensor(
        PALM_LINK_QUAT_IN_IIWA_LINK_7_WXYZ
    )
    offset = offset.expand_as(wrist_position_w)
    local_quaternion = local_quaternion.expand_as(wrist_quaternion_w)
    center = wrist_position_w + quat_apply(wrist_quaternion_w, offset)
    quaternion = quat_mul(wrist_quaternion_w, local_quaternion)
    return center, quaternion


__all__ = [
    "PALM_CENTER_IN_IIWA_LINK_7_M",
    "PALM_LINK_QUAT_IN_IIWA_LINK_7_WXYZ",
    "palm_center_pose_from_merged_wrist",
]
