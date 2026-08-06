# Conda environment setup for SimToolReal + TacMap (`sharpa`)

This guide recreates the conda environment currently used for Isaac Sim / Isaac Lab training in this repo. It is based on the SimToolReal Isaac Sim setup in `docs/isaacsim_installation.md`, but uses a conda environment named `sharpa` instead of `.venv_isaacsim`, and includes the TacMap tactile-sensor requirements.

This environment is for the `isaacsimenvs/` pipeline, not the legacy `isaacgymenvs/` pipeline. Keep Isaac Gym and Isaac Sim in separate environments.

## Current known-good versions

The active `sharpa` environment on this machine uses:

- Python `3.11.15`
- Isaac Sim Python packages `5.1.0.0`
- Isaac Lab `2.3.2.post1`
- PyTorch `2.7.0+cu128`
- TorchVision `0.22.0+cu128`
- `warp-lang 1.12.1`
- `gym 0.23.1`
- `gymnasium 1.2.0`
- `hydra-core 1.3.2`
- `wandb 0.26.0`
- local SimToolReal repo installed editable with `--no-deps`
- repo-local vendored `rl_games/` used by `isaacsimenvs/train.py`

## System prerequisites

Use a recent NVIDIA driver that supports CUDA 12.x and Isaac Sim 5.x. On the current training machine, `nvidia-smi` reports driver `580.159.03` and CUDA runtime support `13.0`, which is sufficient for the CUDA 12.8 PyTorch wheels used by the env.

Basic system tools:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential libgl1 libglib2.0-0 libx11-6 libxi6 libxrender1 libxtst6
```

Use Miniforge or Mambaforge. The examples below use `conda`; `mamba` is fine too.

## Create the conda environment

```bash
conda create -n sharpa -c conda-forge python=3.11 pip -y
conda activate sharpa
python --version
python -m pip install --upgrade pip setuptools wheel
```

Expected major/minor version:

```text
Python 3.11.x
```

## Install PyTorch

The current environment uses CUDA 12.8 PyTorch wheels:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

If CUDA 12.8 wheels are unavailable on your mirror, CUDA 12.6 wheels should also work with Isaac Sim 5.x:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA from Python:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Install Isaac Lab and Isaac Sim

Install Isaac Lab with the Isaac Sim extras from NVIDIA PyPI. This is a large download.

```bash
pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```

The current env resolved Isaac Sim packages to `5.1.0.0`. Do not casually upgrade Isaac Lab: SimToolReal uses Isaac Lab direct-RL APIs and converter behavior that can change across releases.

## Clone the repo

For a fresh clone:

```bash
mkdir -p ~/sharpa
cd ~/sharpa
git clone https://github.com/tylerlum/simtoolreal.git simtoolreal
cd simtoolreal
```

For the existing local working tree:

```bash
cd /path/to/simtoolreal
```

## Install repo-local RL and SimToolReal packages

Install the repo-local `rl_games` first. This repo has a vendored `rl_games/` fork, and local training changes rely on that code path.

```bash
pip install -e ./rl_games/
```

Install the packages needed by SimToolReal / Isaac Sim training. Keep the root install as `--no-deps`: the root `pyproject.toml` contains legacy Isaac Gym pins such as old `numpy`, old `warp-lang`, and `isaacgym-stubs`, which conflict with Python 3.11 / Isaac Sim.

```bash
pip install \
  omegaconf hydra-core "gym==0.23.1" gymnasium scipy numpy yourdfpy requests tqdm tyro \
  "imageio[ffmpeg]" wandb termcolor tensorboard tensorboardX pytest pytest-mock flaky \
  matplotlib pandas opencv-python-headless trimesh shapely coacd "typing_extensions>=4.13"

pip install -e . --no-deps
```

## TacMap-specific requirements

TacMap is implemented inside this repo under `isaacsimenvs/sensors/tacmap/`. It does not need a separate external TacMap package, but it does need the following runtime pieces:

- Isaac Lab ray-caster APIs: provided by `isaaclab==2.3.2.post1`
- Isaac Sim / USD / PhysX Python modules: provided by the `isaacsim-*==5.1.0.0` packages
- `torch` and `torchvision`: TacMap uses tensors and `torchvision.transforms.GaussianBlur`
- `numpy`: TacMap loads tactile point/normal maps from `.npy` assets
- `warp-lang`: pulled in by Isaac Lab / Isaac Sim ray-casting stack
- TacMap assets in this repo: `assets/tacmap/*.npy`

Important files:

```text
isaacsimenvs/sensors/tacmap/sharpa_tacmap_cfg.py
isaacsimenvs/sensors/tacmap/sharpa_tacmap_vbts.py
assets/tacmap/tactileSensor_map_4F_point_origin.npy
assets/tacmap/tactileSensor_map_4F_normal_origin.npy
assets/tacmap/tactileSensor_map_TH_point.npy
assets/tacmap/tactileSensor_map_TH_normal.npy
```

Quick TacMap asset check:

```bash
python -c "from pathlib import Path; import numpy as np; root=Path.cwd(); files=['assets/tacmap/tactileSensor_map_4F_point_origin.npy','assets/tacmap/tactileSensor_map_4F_normal_origin.npy','assets/tacmap/tactileSensor_map_TH_point.npy','assets/tacmap/tactileSensor_map_TH_normal.npy']; [print(f, np.load(root/f).shape, np.load(root/f).dtype) for f in files]"
```

## Environment variables for running Isaac Sim

Accept the Omniverse EULA for non-interactive launches:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional but recommended: put the Kit cache on a local SSD rather than a slow network filesystem.

```bash
export OMNI_KIT_CACHE_PATH=/tmp/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

If you use W&B:

```bash
wandb login
```

## Verify the environment

Basic import check:

```bash
cd /path/to/simtoolreal
conda activate sharpa
python -c "import torch, isaaclab, isaacsim; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); print('isaaclab:', isaaclab.__file__); print('isaacsim:', isaacsim.__file__)"
```

Do not import `isaacsimenvs` in this top-level check. Task registration imports
`isaaclab.envs`, which requires Kit to be initialized through `AppLauncher`.
Use the Isaac Sim smoke test below to verify the local task registration.

Compile-check the local training and TacMap code without launching Isaac Sim:

```bash
python -m py_compile \
  isaacsimenvs/train.py \
  isaacsimenvs/tasks/simtoolreal/simtoolreal_tacmap_env.py \
  isaacsimenvs/sensors/tacmap/sharpa_tacmap_vbts.py \
  rl_games/rl_games/torch_runner.py
```

Run lightweight local unit tests:

```bash
python -m pytest tests/test_expand_obs_checkpoint.py
```

Some TacMap/Isaac Lab tests may skip unless Isaac Lab submodules are initialized through Isaac Sim's app launcher. That is normal for pure unit-test runs.

Run one Isaac Sim smoke test. This starts Kit and is slow the first time:

```bash
python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 --num_assets_per_type 2 --steps 10
```

## Training commands

Official SimToolReal Isaac Sim task:

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless
```

TacMap task with tactile simulation available but without compact tactile policy features:

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TacMap-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless
```

TacMap contact-feature task, using the current compact temporal tactile history setup:

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TacMap-Contact-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --wandb_activate
```

Current partial-load finetuning command from the official no-tactile checkpoint:

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TacMap-Contact-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --wandb_activate \
  --wandb_name tactile_history5_expandobs_tacres20_halfenv \
  --checkpoint pretrained_policy/model.pth \
  --checkpoint_load_mode expand_obs \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
```

## Notes on current TacMap training settings

The active contact-feature task config is:

```text
isaacsimenvs/cfg/task/SimToolRealTacMapContact.yaml
```

Important fields:

```yaml
resolution_step: 20        # 240 / 20 = 12x12 tactile maps per finger
scene:
  num_envs: 12288          # half of the original 24576 to reduce VRAM use
tacmap_history_len: 5      # 5-step compact tactile history
```

`resolution_step` is a stride through the 240x240 tactile map source. Larger values mean lower tactile resolution:

```text
resolution_step=10 -> 24x24 maps, high memory
resolution_step=20 -> 12x12 maps, default/current
resolution_step=40 -> 6x6 maps, lower memory
```

For the compact contact-feature task, the policy does not receive the full tactile image. Current checkpoints use per-finger compact features `[contact_area, depth_mean, depth_max, center_x, center_y]` over the configured history length. With 5 fingers and 5 history steps, the tactile actor observation adds `5 * 5 * 5 = 125` values. Set `tacmap_policy_include_depth=false` only for legacy three-feature checkpoints.

## Common problems

### Accidentally importing the wrong `rl_games`

The conda env may contain a PyPI or GitHub `rl_games` package, but this repo has a local fork under `rl_games/`. `isaacsimenvs/train.py` inserts the repo-local `rl_games` path before importing the runner. If you run custom scripts, make sure they import the local fork when using local checkpoint-loading changes.

Quick check:

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / 'rl_games')); import rl_games.torch_runner as tr; print(tr.__file__)"
```

Expected path begins with:

```text
/path/to/simtoolreal/rl_games/rl_games/
```

### Do not install root dependencies without `--no-deps`

Avoid this in the Isaac Sim conda env:

```bash
pip install -e .
```

Use this instead:

```bash
pip install -e . --no-deps
```

The unguarded root dependencies are still useful for the older Isaac Gym workflow but conflict with the Isaac Sim / Python 3.11 workflow.

### Isaac Lab imports before AppLauncher

Some `isaaclab.*` submodules are only valid after Isaac Sim's app launcher has initialized Kit. Training scripts handle this. Standalone scripts should follow the Isaac Lab pattern and launch the app before importing deep Isaac Lab submodules.

### CUDA OOM with TacMap

TacMap memory scales with:

```text
num_envs * num_sensors * map_height * map_width
```

For five sensors:

```text
24576 envs, step 20: 5 * 12 * 12 maps per env
12288 envs, step 20: half the env memory
24576 envs, step 10: 4x the TacMap map data vs step 20
```

If you hit OOM, first reduce `scene.num_envs`, then consider increasing `resolution_step`.

### PhysX scene corruption after OOM

After a CUDA OOM or PhysX scene corruption, stop the Python process completely before starting another run. Check for stale processes with:

```bash
nvidia-smi
```

Only run one Isaac Sim training process per GPU unless you have explicitly partitioned resources.

## Optional: export a lockfile snapshot

After setup, save an exact package snapshot:

```bash
conda list -n sharpa > docs/sharpa_conda_list.txt
conda run -n sharpa python -m pip freeze > docs/sharpa_pip_freeze.txt
```

These snapshots are useful for reproduction, but the setup commands above are easier to audit than installing directly from a full freeze.
