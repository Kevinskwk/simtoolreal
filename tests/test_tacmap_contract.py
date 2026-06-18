from pathlib import Path

import pytest

pytest.importorskip("isaaclab.envs")


def test_tacmap_task_registration_and_cfg_contract():
    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import SimToolRealTacMapEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_obs_dim

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


def test_tacmap_cfg_resolves_assets_and_sensor_shapes():
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import SimToolRealTacMapEnvCfg

    cfg = SimToolRealTacMapEnvCfg()
    assert cfg.enable_vbts is True
    assert cfg.compute_tacmap_obs_size() == 5 * 12 * 12
    for sensor_cfg in cfg.vbts_sensor:
        assert Path(sensor_cfg.points_npy).is_file()
        assert Path(sensor_cfg.normals_npy).is_file()
        assert sensor_cfg.prim_path.startswith("/World/envs/env_.*/Robot/left_")
        assert sensor_cfg.target_rigid_expr == "/World/envs/env_.*/Object/.*"
