# TacMap IsaacSim Integration

This branch adds TacMap tactile simulation on top of the official
`isaacsimenvs` SimToolReal task. The policy and critic observations are
unchanged from `Isaacsimenvs-SimToolReal-Direct-v0`; TacMap buffers are updated
for debugging and future tactile-policy work.

## Task

Use the TacMap task id with the existing official PPO or SAPG agent configs:

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TacMap-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --wandb_activate
```

For a short smoke run, also override:

```bash
agent.params.config.max_epochs=1
agent.params.config.horizon_length=4
agent.params.config.minibatch_size=1024
agent.params.config.central_value_config.minibatch_size=1024
```

## Scope

Included:

- `SharpaTacmap` ray-cast tactile sensors.
- Five fingertip TacMap sensors attached to the official merged fingertip
  bodies.
- TacMap map assets under `assets/tacmap`.
- A registered proprio-only TacMap task:
  `Isaacsimenvs-SimToolReal-TacMap-Direct-v0`.

Deferred:

- Feeding tactile tensors into the policy.
- TacMap CNN policy fusion.
- rl_games structured-dict observation patches.
- Any replacement of official reward, reset, action, or termination code.
