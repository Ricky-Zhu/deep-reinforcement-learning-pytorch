import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete, Box
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical


class Actor_net(nn.Module):
    def __init__(self, obs_dim, action_dim, action_space):
        super(Actor_net, self).__init__()
        self.action_space = action_space

        if isinstance(self.action_space, Box):
            self.fc1 = nn.Linear(obs_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.mean = nn.Linear(64, action_dim)
            self.log_std = nn.Parameter(torch.zeros([1, action_dim]))

            # init the mean weight and bias
            self.mean.weight.data.mul_(0.1)
            self.mean.bias.data.zero_()
        else:
            self.fc1 = nn.Linear(obs_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_dim)
            self.fc3.weight.data.mul_(0.1)
            self.fc3.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        if isinstance(self.action_space, Box):
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            mean = self.mean(x)
            log_std = self.log_std.expand_as(mean)
            std = torch.exp(log_std)
            return mean, std
        else:
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = self.fc3(x)
            probs = F.softmax(x, dim=1)
            return probs

    def evaluate_action_log_probs(self, obs, actions):
        if isinstance(self.action_space, Box):
            mean, std = self.forward(obs)
            dist = Normal(mean, std)
            entropy = dist.entropy().mean()
            log_probs = dist.log_prob(actions)
        else:
            probs = self.forward(obs)
            dist = Categorical(probs)
            log_probs = torch.log(probs)
            log_probs = torch.gather(log_probs, 1, actions)
            entropy = dist.entropy().mean()
        return log_probs, entropy


class Value_net(nn.Module):
    def __init__(self, obs_dim):
        super(Value_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v = nn.Linear(64, 1)

        self.v.weight.data.mul_(0.1)
        self.v.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.v(x)
        return x
