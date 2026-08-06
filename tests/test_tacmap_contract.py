from pathlib import Path
from types import SimpleNamespace

import torch

import pytest

pytest.importorskip("isaaclab.envs")


def test_tacmap_task_registration_and_cfg_contract():
    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (
        SimToolRealTacMapContactEnvCfg,
        SimToolRealTacMapEnvCfg,
    )
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import (
        compute_obs_dim,
        register_obs_field_size,
    )

    spec = gym.spec("Isaacsimenvs-SimToolReal-TacMap-Direct-v0")
    assert spec.kwargs["env_cfg_entry_point"].endswith("SimToolRealTacMapEnvCfg")
    assert spec.kwargs["rl_games_cfg_entry_point"].endswith("SimToolRealPPO.yaml")
    assert spec.kwargs["rl_games_sapg_cfg_entry_point"].endswith("SimToolRealSAPG.yaml")

    base_cfg = SimToolRealEnvCfg()
    tacmap_cfg = SimToolRealTacMapEnvCfg()
    assert compute_obs_dim(tacmap_cfg.obs.obs_list) == compute_obs_dim(base_cfg.obs.obs_list)
    assert compute_obs_dim(tacmap_cfg.obs.state_list) == compute_obs_dim(base_cfg.obs.state_list)
    assert tacmap_cfg.observation_space == base_cfg.observation_space
    assert tacmap_cfg.state_space == base_cfg.state_space

    contact_spec = gym.spec("Isaacsimenvs-SimToolReal-TacMap-Contact-Direct-v0")
    assert contact_spec.kwargs["env_cfg_entry_point"].endswith("SimToolRealTacMapContactEnvCfg")
    assert contact_spec.kwargs["env_cfg_yaml_entry_point"].endswith("SimToolRealTacMapContact.yaml")
    assert contact_spec.kwargs["rl_games_sapg_cfg_entry_point"].endswith("SimToolRealSAPG.yaml")

    contact_cfg = SimToolRealTacMapContactEnvCfg()
    register_obs_field_size("tacmap", contact_cfg.compute_tacmap_obs_size())
    assert contact_cfg.tacmap_history_len == 5
    assert contact_cfg.compute_tacmap_obs_size() == 125
    contact_cfg.tacmap_policy_include_depth = False
    assert contact_cfg.compute_tacmap_obs_size() == 75
    contact_cfg.tacmap_policy_include_depth = True
    assert compute_obs_dim(contact_cfg.obs.obs_list) == compute_obs_dim(base_cfg.obs.obs_list) + 125
    assert compute_obs_dim(contact_cfg.obs.state_list) == compute_obs_dim(base_cfg.obs.state_list)


def test_tacmap_cfg_resolves_assets_and_sensor_shapes():
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import SimToolRealTacMapEnvCfg

    cfg = SimToolRealTacMapEnvCfg()
    assert cfg.enable_vbts is True
    assert cfg.compute_tacmap_obs_size() == 5 * 12 * 12
    cfg.resolution_step = 10
    assert {sensor_cfg.resolution_step for sensor_cfg in cfg.vbts_sensor} == {20}

    from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env import SimToolRealTacMapEnv

    SimToolRealTacMapEnv._sync_vbts_sensor_cfgs(cfg)
    assert {sensor_cfg.resolution_step for sensor_cfg in cfg.vbts_sensor} == {10}
    assert cfg.compute_tacmap_obs_size() == 5 * 24 * 24
    for sensor_cfg in cfg.vbts_sensor:
        assert Path(sensor_cfg.points_npy).is_file()
        assert Path(sensor_cfg.normals_npy).is_file()
        assert sensor_cfg.prim_path.startswith("/World/envs/env_.*/Robot/left_")
        assert sensor_cfg.target_rigid_expr == "/World/envs/env_.*/Object/object_root"


def test_tacmap_contact_policy_obs_stacks_history_without_sim():
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env import SimToolRealTacMapEnv

    env = object.__new__(SimToolRealTacMapEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.cfg = SimpleNamespace(
        tacmap_obs_normalization=255.0,
        contact_threshold=0.05,
        disable_tactile_ids=[],
        binary_contact=False,
        contact_smooth=1.0,
        contact_latency=0.0,
        contact_sensor_noise=0.0,
        enable_tactile=True,
    )
    env.vbts_deform = torch.zeros(2, 5, 12, 12, dtype=torch.uint8)
    env.last_contacts = torch.zeros(2, 5)
    env._prev_raw_contacts = torch.zeros(2, 5)
    env._tacmap_policy_obs_history = torch.zeros(2, 5, 25)
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env._tacmap_grid_y, env._tacmap_grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 12),
        torch.linspace(-1.0, 1.0, 12),
        indexing="ij",
    )

    env.vbts_deform[0, 0, 0, 0] = 255
    first = env.get_tacmap_policy_obs()

    assert first.shape == (2, 125)
    assert torch.allclose(first[:, 0:25], first[:, 25:50])
    assert first[0, 0] > 0.0
    assert first[0, 1] == 1.0
    assert first[0, 2] == 1.0
    assert first[0, 3] == -1.0
    assert first[0, 4] == -1.0

    env.episode_length_buf[:] = 1
    env.vbts_deform.zero_()
    second = env.get_tacmap_policy_obs()

    assert second.shape == (2, 125)
    assert torch.all(second[:, 0:25] == 0.0)
    assert torch.allclose(second[:, 25:50], first[:, 0:25])


def test_tacmap_contact_policy_obs_supports_legacy_three_feature_layout():
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env import SimToolRealTacMapEnv

    env = object.__new__(SimToolRealTacMapEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env.cfg = SimpleNamespace(
        tacmap_obs_normalization=255.0,
        contact_threshold=0.05,
        disable_tactile_ids=[],
        binary_contact=False,
        contact_smooth=1.0,
        contact_latency=0.0,
        contact_sensor_noise=0.0,
        enable_tactile=True,
        tacmap_policy_include_depth=False,
    )
    env.vbts_deform = torch.zeros(1, 5, 12, 12, dtype=torch.uint8)
    env.vbts_deform[0, 0, 0, 0] = 255
    env.last_contacts = torch.zeros(1, 5)
    env._prev_raw_contacts = torch.zeros(1, 5)
    env._tacmap_policy_obs_history = torch.zeros(1, 5, 15)
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env._tacmap_grid_y, env._tacmap_grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 12),
        torch.linspace(-1.0, 1.0, 12),
        indexing="ij",
    )

    observation = env.get_tacmap_policy_obs()

    assert observation.shape == (1, 75)
    assert observation[0, 0] > 0.0
    assert observation[0, 1] == -1.0
    assert observation[0, 2] == -1.0
