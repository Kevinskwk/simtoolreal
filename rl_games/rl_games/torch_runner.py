import os
import time
import numpy as np
import random
from copy import deepcopy
import torch

from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
from rl_games.algos_torch import torch_ext


def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        load_mode = args.get('checkpoint_load_mode', 'resume')
        if load_mode == 'resume':
            agent.restore(args['checkpoint'])
        elif load_mode == 'weights':
            weights = _load_checkpoint_weights(agent, args['checkpoint'])
            agent.set_weights(weights)
            _try_load_central_value(agent, weights)
            print(f"=> initialized model weights from '{args['checkpoint']}'")
        elif load_mode == 'expand_obs':
            weights = _load_checkpoint_weights(agent, args['checkpoint'])
            weights = _expand_obs_checkpoint_weights(agent, weights)
            agent.set_weights(weights)
            _try_load_central_value(agent, weights)
            print(f"=> initialized expanded-observation model weights from '{args['checkpoint']}'")
        else:
            raise ValueError(f"checkpoint_load_mode must be resume/weights/expand_obs, got {load_mode!r}")


def _load_checkpoint_weights(agent, checkpoint_path):
    checkpoint = torch_ext.load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict):
        if getattr(agent, 'global_rank', None) in checkpoint:
            return checkpoint[agent.global_rank]
        if 0 in checkpoint:
            return checkpoint[0]
    return checkpoint


def _try_load_central_value(agent, weights):
    if getattr(agent, 'has_central_value', False) and 'assymetric_vf_nets' in weights:
        try:
            agent.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        except RuntimeError as exc:
            print(f"Skipping central value checkpoint weights: {exc}")


def _expand_obs_checkpoint_weights(agent, weights):
    weights = deepcopy(weights)
    if 'model' not in weights:
        raise KeyError("expand_obs checkpoint must contain a 'model' state_dict")

    source_model = weights['model']
    target_model = agent.model.state_dict()
    _expand_running_mean_std(source_model, target_model)
    _expand_actor_rnn_input(source_model, target_model)
    _adapt_sapg_block_params(source_model, target_model)
    return weights


def _expand_running_mean_std(source_model, target_model):
    mean_key = 'running_mean_std.running_mean'
    var_key = 'running_mean_std.running_var'
    for key, fill_value in ((mean_key, 0.0), (var_key, 1.0)):
        if key not in source_model or key not in target_model:
            continue
        source = source_model[key]
        target = target_model[key]
        if source.shape == target.shape:
            continue
        if source.ndim != 1 or target.ndim != 1 or source.shape[0] > target.shape[0]:
            raise RuntimeError(
                f"expand_obs cannot adapt {key}: checkpoint shape {tuple(source.shape)} "
                f"target shape {tuple(target.shape)}"
            )
        expanded = target.clone()
        expanded.fill_(fill_value)
        expanded[: source.shape[0]] = source
        source_model[key] = expanded
        print(f"=> expanded {key}: {tuple(source.shape)} -> {tuple(target.shape)}")


def _expand_actor_rnn_input(source_model, target_model):
    key = 'a2c_network.rnn.rnn.weight_ih_l0'
    mean_key = 'running_mean_std.running_mean'
    if key not in source_model or key not in target_model:
        return
    source = source_model[key]
    target = target_model[key]
    if source.shape == target.shape:
        return
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise RuntimeError(
            f"expand_obs cannot adapt {key}: checkpoint shape {tuple(source.shape)} "
            f"target shape {tuple(target.shape)}"
        )
    if mean_key not in target_model:
        raise KeyError(f"expand_obs needs target {mean_key} to infer observation dimensions")

    new_obs_dim = int(target_model[mean_key].shape[0])
    old_cols = int(source.shape[1])
    new_cols = int(target.shape[1])
    sapg_embed_dim = new_cols - new_obs_dim
    old_obs_dim = old_cols - sapg_embed_dim
    tactile_dim = new_obs_dim - old_obs_dim
    if sapg_embed_dim < 0 or old_obs_dim < 0 or tactile_dim < 0:
        raise RuntimeError(
            f"expand_obs inferred invalid dims: old_obs={old_obs_dim}, new_obs={new_obs_dim}, "
            f"old_cols={old_cols}, new_cols={new_cols}"
        )
    expected_new_cols = new_obs_dim + sapg_embed_dim
    if new_cols != expected_new_cols:
        raise RuntimeError(
            f"expand_obs expected target {key} to have {expected_new_cols} columns "
            f"(new_obs={new_obs_dim} + embed={sapg_embed_dim}), got {new_cols}"
        )

    expanded = target.clone()
    expanded.zero_()
    expanded[:, :old_obs_dim] = source[:, :old_obs_dim]
    expanded[:, new_obs_dim:new_cols] = source[:, old_obs_dim:old_cols]
    source_model[key] = expanded
    print(
        f"=> expanded {key}: {tuple(source.shape)} -> {tuple(target.shape)} "
        f"with tactile_dim={tactile_dim}, sapg_embed_dim={sapg_embed_dim}"
    )


def _adapt_sapg_block_params(source_model, target_model):
    for key in ('a2c_network.extra_params', 'a2c_network.sigma'):
        if key not in source_model or key not in target_model:
            continue
        source = source_model[key]
        target = target_model[key]
        if source.shape == target.shape:
            continue
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise RuntimeError(
                f"expand_obs cannot adapt {key}: checkpoint shape {tuple(source.shape)} "
                f"target shape {tuple(target.shape)}"
            )

        adapted = target.clone()
        rows = min(source.shape[0], target.shape[0])
        adapted[:rows] = source[:rows]
        if target.shape[0] > source.shape[0]:
            repeat_count = target.shape[0] - source.shape[0]
            adapted[source.shape[0]:] = source[-1:].expand(repeat_count, -1)
        source_model[key] = adapted
        print(f"=> adapted {key}: {tuple(source.shape)} -> {tuple(target.shape)}")


def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma == 'fixed':
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')


class Runner:

    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if params["config"].get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            # set different random seed for each GPU
            self.seed += self.global_rank

            print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size}")

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += self

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def set_vec_env(self, vec_env):
        self.params['config']['vec_env'] = vec_env

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)
        return agent.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        if args['train']:
            return self.run_train(args)
        elif args['play']:
            return self.run_play(args)
        else:
            return self.run_train(args)