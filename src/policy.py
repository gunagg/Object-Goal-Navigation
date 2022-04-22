import abc

import math
import sys
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from .models.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
)
from .models.rnn_state_encoder import RNNStateEncoder
from .models.running_mean_and_var import RunningMeanAndVar
from .models.common import CategoricalNet, CustomFixedCategorical

from habitat import Config


class Policy(nn.Module):

    # The following configurations are used in the trainer to create the appropriate rollout
    # As well as the appropriate auxiliary task wiring
    # Whether to use multiple beliefs
    IS_MULTIPLE_BELIEF = False
    # Whether to section a single belief for auxiliary tasks, keeping a single GRU core
    IS_SECTIONED = False
    # Whether the fusion module is an RNN (see RecurrentAttentivePolicy)
    IS_RECURRENT = False
    # Has JIT support
    IS_JITTABLE = False
    # Policy fuses multiple inputs
    LATE_FUSION = True

    def __init__(self, net, dim_actions, observation_space=None, config=None, **kwargs):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        actor_head_layers = getattr(config, "ACTOR_HEAD_LAYERS", 1)
        critic_head_layers = getattr(config, "CRITIC_HEAD_LAYERS", 1)

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions, layers=actor_head_layers
        )
        self.critic = CriticHead(self.net.output_size, layers=critic_head_layers)
        if "rgb" in observation_space.spaces:
            self.running_mean_and_var = RunningMeanAndVar(
                observation_space.spaces["rgb"].shape[-1]
                + (
                    observation_space.spaces["depth"].shape[-1]
                    if "depth" in observation_space.spaces
                    else 0
                ),
                initial_count=1e4,
            )
        else:
            self.running_mean_and_var = None

    def forward(self, *x):
        raise NotImplementedError

    def _preprocess_obs(self, observations):
        dtype = next(self.parameters()).dtype
        observations = {k: v.to(dtype=dtype) for k, v in observations.items()}
         # since this seems to be what running_mean_and_var is expecting
        observations = {k: v.permute(0, 3, 1, 2) if len(v.size()) == 4 else v for k, v in observations.items()}
        observations = _process_depth(observations)

        if "rgb" in observations:
            rgb = observations["rgb"].to(dtype=next(self.parameters()).dtype) / 255.0
            x = [rgb]
            if "depth" in observations:
                x.append(observations["depth"])

            x = self.running_mean_and_var(torch.cat(x, 1))
            # this preprocesses depth and rgb, but not semantics. we're still embedding that in our encoder
            observations["rgb"] = x[:, 0:3]
            if "depth" in observations:
                observations["depth"] = x[:, 3:]
        # ! Permute them back, because the rest of our code expects unpermuted
        observations = {k: v.permute(0, 2, 3, 1) if len(v.size()) == 4 else v for k, v in observations.items()}

        return observations

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs
    ):
        observations = self._preprocess_obs(observations)
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        observations = self._preprocess_obs(observations)
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        observations = self._preprocess_obs(observations)
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, None, None, None


class CriticHead(nn.Module):
    HIDDEN_SIZE = 32
    def __init__(self, input_size, layers=1):
        super().__init__()
        if layers == 1:
            self.fc = nn.Linear(input_size, 1)
            nn.init.orthogonal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)
        else: # Only support 2 layers max
            self.fc = nn.Sequential(
                nn.Linear(input_size, self.HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(self.HIDDEN_SIZE, 1)
            )
            nn.init.orthogonal_(self.fc[0].weight)
            nn.init.constant_(self.fc[0].bias, 0)

    def forward(self, x):
        return self.fc(x)

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class Seq2SeqNet(Net):
    r"""A baseline sequence to sequence network that concatenates
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
        self,
        observation_space: Space,
        model_config: Config,
        num_actions,
        goal_sensor_uuid=None,
        additional_sensors=["gps", "compass"]
    ):
        super().__init__()
        self.model_config = model_config

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
                trainable=model_config.DEPTH_ENCODER.trainable,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
            "ResnetRGBEncoder",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, model_config.RGB_ENCODER.output_size, device
            )
        elif model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.rgb_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.backbone,
                trainable=model_config.RGB_ENCODER.train_encoder,
            )

        # Init the RNN state decoder
        rnn_input_size = (
            model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )
        
        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors

        if "gps" in additional_sensors:
            input_gps_dim = observation_space.spaces[
                "gps"
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            # logger.info("\n\nSetting up GPS sensor")
        
        if "compass" in observation_space.spaces:
            assert (
                observation_space.spaces["compass"].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            # logger.info("\n\nSetting up Compass sensor")

        if self.goal_sensor_uuid is not None and self.goal_sensor_uuid != "no_sensor":
            self._n_object_categories = (
                int(
                    observation_space.spaces[self.goal_sensor_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            # logger.info("\n\nSetting up Object Goal sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = [depth_embedding, rgb_embedding]

        if "gps" in self.additional_sensors:
            obs_gps = observations["gps"]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))
        
        if "compass" in self.additional_sensors:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if self.goal_sensor_uuid is not None and self.goal_sensor_uuid != "no_sensor":
            object_goal = observations["objectgoal"].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config,
        goal_sensor_uuid=None,
        additional_sensors=["gps", "compass"]
    ):
        super().__init__()
        self.net = Seq2SeqNet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
            goal_sensor_uuid=goal_sensor_uuid,
            additional_sensors=additional_sensors,
        )
        self.action_distribution = CategoricalNet(
            self.net.output_size, action_space.n
        )
        self.train()
    
    def forward(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        return distribution.logits, rnn_hidden_states