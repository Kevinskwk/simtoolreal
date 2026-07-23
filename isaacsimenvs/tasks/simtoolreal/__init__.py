"""SimToolReal task registration.

Registers ``Isaacsimenvs-SimToolReal-Direct-v0`` with the gymnasium registry
for the DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → SimToolRealEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/SimToolReal.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/SimToolRealPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/SimToolRealSAPG.yaml (default)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .simtoolreal_env import SimToolRealEnv
from .simtoolreal_env_cfg import SimToolRealEnvCfg
from .simtoolreal_fixed_grasp_force_env import SimToolRealFixedGraspNormalForceEnv
from .simtoolreal_tacmap_env import SimToolRealTacMapEnv
from .simtoolreal_scrape_pose_env import SimToolRealTacMapScrapePoseEnv
from .simtoolreal_tacmap_env_cfg import (
    SimToolRealTacMapContactEnvCfg,
    SimToolRealTacMapEnvCfg,
    SimToolRealFixedGraspNormalForceEnvCfg,
    SimToolRealTacMapScrapePoseEnvCfg,
)

__all__ = [
    "SimToolRealEnv",
    "SimToolRealEnvCfg",
    "SimToolRealTacMapEnv",
    "SimToolRealTacMapContactEnvCfg",
    "SimToolRealTacMapEnvCfg",
    "SimToolRealTacMapScrapePoseEnv",
    "SimToolRealTacMapScrapePoseEnvCfg",
    "SimToolRealFixedGraspNormalForceEnv",
    "SimToolRealFixedGraspNormalForceEnvCfg",
]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-SimToolReal-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg:SimToolRealEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolReal.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)

gym.register(
    id="Isaacsimenvs-SimToolReal-TacMap-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env:SimToolRealTacMapEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg:SimToolRealTacMapEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolRealTacMap.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)

gym.register(
    id="Isaacsimenvs-SimToolReal-TacMap-Contact-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env:SimToolRealTacMapEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg:SimToolRealTacMapContactEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolRealTacMapContact.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)


gym.register(
    id="Isaacsimenvs-SimToolReal-TacMap-Scrape-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_scrape_pose_env:SimToolRealTacMapScrapePoseEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg:SimToolRealTacMapScrapePoseEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolRealTacMapScrape.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)

gym.register(
    id="Isaacsimenvs-SimToolReal-TacMap-FixedGrasp-NormalForce-Direct-v0",
    entry_point=(
        "isaacsimenvs.tasks.simtoolreal.simtoolreal_fixed_grasp_force_env:"
        "SimToolRealFixedGraspNormalForceEnv"
    ),
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg:"
            "SimToolRealFixedGraspNormalForceEnvCfg"
        ),
        "env_cfg_yaml_entry_point": str(
            _CFG_DIR / "task" / "SimToolRealFixedGraspNormalForce.yaml"
        ),
        "rl_games_cfg_entry_point": str(
            _CFG_DIR / "train" / "SimToolRealFixedGraspForceSAPG.yaml"
        ),
        "rl_games_sapg_cfg_entry_point": str(
            _CFG_DIR / "train" / "SimToolRealFixedGraspForceSAPG.yaml"
        ),
    },
)
