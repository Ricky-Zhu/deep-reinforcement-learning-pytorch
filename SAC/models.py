import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros([1, action_dim]))

    def forward(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, -20., 2.)
        std = torch.exp(log_std)
        return mean, std

    def sample_action(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        pre_tanh_action = dist.rsample()

        action = torch.tanh(pre_tanh_action)

        log_probs = dist.log_prob(pre_tanh_action).sum(1, keepdims=True) - \
                    torch.log(1 - action * action + 1e-6).sum(1, keepdims=True)

        return pre_tanh_action, action, log_probs

    def eval_action(self, obs, actions):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions)
        return log_probs


class QNet(nn.Module):
    def __init__(self, obs_dim, action_dim, init_method=None):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

        if init_method:
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v
