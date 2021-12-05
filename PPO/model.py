import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal


class Actor_net(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros([1, action_dim]))

    def forward(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std

    def evaluate_action_log_probs(self, obs, actions):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        entropy = dist.entropy()
        log_probs = dist.log_prob(actions)
        return log_probs, entropy


class Value_net(nn.Module):
    def __init__(self, obs_dim):
        super(Value_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.v(x)
        return x
