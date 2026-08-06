#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

variant="${1:-}"
timestamp="$(date +%Y%m%d_%H%M%S)"

case "${variant}" in
  tactile)
    checkpoint="outputs/2026-07-22/08-24-28/0_simtoolreal_sapg/nn/last_0_simtoolreal_sapg_ep_27000_rew_32252.904.pth"
    run_name="tactile_intervalforce_onsetgate_huber_${timestamp}"
    observation_overrides=(
      "env.use_tacmap=true"
      "env.enable_vbts=true"
      "env.include_tacmap_in_policy=true"
      "env.tacmap_history_len=5"
      "env.obs.obs_list=[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,tacmap,scrape_target_contact_normal_force]"
    )
    ;;
  no-tactile)
    checkpoint="outputs/2026-07-21/11-48-11/0_simtoolreal_sapg/nn/last_0_simtoolreal_sapg_ep_27000_rew_30620.992.pth"
    run_name="no_tactile_intervalforce_onsetgate_huber_${timestamp}"
    observation_overrides=(
      "env.use_tacmap=false"
      "env.enable_vbts=false"
      "env.include_tacmap_in_policy=false"
      "env.obs.obs_list=[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,scrape_target_contact_normal_force]"
    )
    ;;
  *)
    echo "Usage: $0 {tactile|no-tactile}" >&2
    exit 2
    ;;
esac

python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TacMap-Scrape-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --checkpoint "${checkpoint}" \
  --checkpoint_load_mode weights \
  --wandb_activate \
  --wandb_project simtoolreal \
  --wandb_group contact_onset_filter_matched_ep27000 \
  --wandb_name "${run_name}" \
  "${observation_overrides[@]}" \
  env.enable_tool_table_contact_force_reward=true \
  env.contact_force_use_control_interval_average=true \
  env.tool_table_contact_sensor_update_period=0.0 \
  env.tool_table_contact_sensor_history_len=2 \
  env.contact_force_filter_alpha=0.2 \
  env.contact_force_onset_threshold_n=0.1 \
  env.contact_force_onset_grace_steps=3 \
  env.contact_force_reward_ramp_steps=6 \
  env.contact_force_huber_delta_n=1.0 \
  env.contact_force_grasp_min_fingertips=2 \
  env.contact_force_grasp_max_fingertip_distance_m=0.12 \
  env.scene.num_envs=12288 \
  agent.params.seed=42 \
  agent.params.config.max_epochs=12000 \
  agent.params.config.minibatch_size=98304 \
  agent.params.config.central_value_config.minibatch_size=98304 \
  agent.params.config.expl_coef_block_size=4096 \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
