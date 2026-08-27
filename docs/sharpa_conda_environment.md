# Reproduce the extended SimToolReal environment with conda (`sharpa`)

This is the canonical setup guide for the research extensions in this fork: TacMap, scraping/contact experiments, probing tools, grasp adjustment, and Allen-key turning. It follows the original SimToolReal Isaac Sim setup in `docs/isaacsim_installation.md`, uses a conda environment named `sharpa`, and adds the packages required by the newer environments and analysis scripts.

This environment covers the active `isaacsimenvs/` training/evaluation pipeline, the local Viser demo, and experiment analysis. It does not attempt to combine the legacy `isaacgymenvs/` stack or robot deployment/ROS dependencies into one environment. Follow their dedicated documentation and keep those environments separate.

For an exact handoff to another machine, reproduce all three layers:

1. The same Git branch or commit.
2. The Python/Isaac environment in this guide.
3. Any ignored checkpoints, datasets, or run outputs needed for the work being resumed.

## Repository revision

The added research code is published on the fork and is not present in the original upstream `main` branch. Clone the current branch directly:

```bash
mkdir -p ~/sharpa
cd ~/sharpa
git clone --branch feature/allen-key-adjustment-gate --single-branch \
  https://github.com/Kevinskwk/simtoolreal.git simtoolreal
cd simtoolreal
git remote add upstream https://github.com/tylerlum/simtoolreal.git
git status -sb
git rev-parse HEAD
```

If the work moves to another branch later, use the branch or commit from the source machine instead. A Codex agent should inspect `git status -sb`, `git branch -vv`, and the requested experiment script before installing or running anything.

## Known-good versions

The reproducible package baseline recorded and exercised on this machine uses:

- Python `3.11.15`
- Isaac Sim Python packages `5.1.0.0`
- Isaac Lab `2.3.2.post1`
- PyTorch `2.7.0+cu128`
- TorchVision `0.22.0+cu128`
- `warp-lang 1.12.1`
- `gym 0.23.1`
- `gymnasium 1.2.0`
- `hydra-core 1.3.2` and `omegaconf 2.3.0`
- `numpy 1.26.0` and `scipy 1.15.3`
- `wandb 0.26.0`
- local SimToolReal repo installed editable with `--no-deps`
- repo-local vendored `rl_games/` used by `isaacsimenvs/train.py`

The remaining direct dependencies and their tested versions are tracked in `requirements-isaacsim-extra.txt`.

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

## Install repo-local RL and SimToolReal packages

Install the repo-local `rl_games` fork without resolving its generic dependencies, then install the tested Isaac Sim dependency set. Installing it with `--no-deps` avoids pulling a second GUI OpenCV wheel alongside the headless wheel.

```bash
cd ~/sharpa/simtoolreal
pip install -e ./rl_games/ --no-deps
pip install -r requirements-isaacsim-extra.txt
```

Register the repo packages. Keep this install as `--no-deps`: the root `pyproject.toml` contains legacy Isaac Gym pins such as old `numpy`, old `warp-lang`, and `isaacgym-stubs`, which conflict with Python 3.11 / Isaac Sim.

```bash
pip install -e . --no-deps
```

Do not replace the requirements file with `pip install -e .` or `pip install -e ./rl_games/` without `--no-deps`. Those dependency declarations cover other environments and are not a valid lock for this Isaac Sim setup.

## Download required external data

The original pretrained policy is intentionally ignored by Git. Download it before running evaluation or finetuning:

```bash
python download_pretrained_policy.py
test -s pretrained_policy/model.pth
test -s pretrained_policy/config.yaml
```

TacMap maps and the Allen-key assets are tracked in Git and require no separate download. DexToolBench datasets are optional unless running the benchmark; follow `docs/dextoolbench.md` for those files.

Git also ignores `outputs/`, downloaded checkpoints, generated grasp banks, W&B local run data, and rendered videos. Copy the specific artifacts separately if the new machine must resume an existing run. Prefer an absolute checkpoint path and verify it with `test -s /path/to/model.pth` before launch.

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

If you use W&B, authenticate once on the new machine:

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

Confirm that Python resolves the vendored RL fork:

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / 'rl_games')); import rl_games; print(rl_games.__file__)"
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

Run lightweight local unit tests without starting Isaac Sim:

```bash
python -m pytest -q tests
```

Some TacMap/Isaac Lab tests may skip unless Isaac Lab submodules are initialized through Isaac Sim's app launcher. That is normal for pure unit-test runs.

Run the AppLauncher and task-registration smoke tests. Each starts Kit and can be slow on first launch:

```bash
python isaacsimenvs/tests/test_load_isaacsim.py
python isaacsimenvs/tests/test_gym_register.py
```

Then construct and step a small environment:

```bash
python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 --num_assets_per_type 2 --steps 10
```

Only after these checks pass should a new machine launch a large environment count.

## Training commands

Download `pretrained_policy/model.pth` first. The original task remains the best end-to-end installation check:

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

The current Allen-key workflow has a deterministic preflight that does not start training or require W&B:

```bash
python scripts/validate_allen_key_turning_task.py \
  --output outputs/allen_key_turning_validation/install_smoke.html \
  --headless
```

For real training, use `scripts/run_allen_key_turning_training.sh` and set the intended `NUM_ENVS`, `MAX_EPOCHS`, and optional `CHECKPOINT`. Read the script first because its defaults evolve with the experiment. Do not run a second Isaac Sim process on a GPU already used by training.

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

## Record the completed installation

After setup, save an exact package snapshot:

```bash
conda list -n sharpa > docs/sharpa_conda_list.txt
conda run -n sharpa python -m pip freeze > docs/sharpa_pip_freeze.txt
```

These snapshots are useful for reproduction, but the setup commands above are easier to audit than installing directly from a full freeze.

Also record the source revision and GPU stack with the experiment:

```bash
mkdir -p outputs/reproduction
git rev-parse HEAD > outputs/reproduction/source_commit.txt
nvidia-smi > outputs/reproduction/nvidia_smi.txt
```

The snapshot files are ignored by Git because they contain machine-specific transitive packages. `requirements-isaacsim-extra.txt` is the reviewed, portable dependency list.

## Minimal handoff checklist for another Codex agent

Give the agent the repository URL, branch or commit, target GPU, and checkpoint/data locations. It should then:

1. Follow this guide without installing root dependencies.
2. Run the CUDA import check and all three Isaac Sim smoke tests.
3. Download the original pretrained policy or verify transferred checkpoints.
4. Run the task-specific deterministic preflight before using the production environment count.
5. Report exact failing commands and tracebacks; it must not silently skip missing sensors, assets, checkpoints, or validation failures.
