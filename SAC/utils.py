import torch
from torch.distributions.normal import Normal
from torch.distributions import Distribution
import random
from collections import deque
import numpy as np


class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(self.normal_mean, self.normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2

        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pre_tanh_value=False):
        z = self.normal.sample().detach()
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pre_tanh_value=False):
        sample_mean = torch.zeros(self.normal_mean.size(), dtype=torch.float32)
        sample_std = torch.ones_like(sample_mean)

        z = self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample()
        z.requires_grad_()

        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class get_action_info:
    def __init__(self, pis):
        self.mean, self.std = pis
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std)

    def select_action(self, exploration=True, reparam=False):
        if exploration:
            if reparam:
                action, pretanh = self.dist.rsample(return_pre_tanh_value=True)
                return action, pretanh
            else:
                action = self.dist.sample()
        else:
            action = torch.tanh(self.mean)
        return action

    def get_log_prob(self, actions, pre_tanh_values):
        log_probs = self.dist.log_prob(actions, pre_tanh_values)
        return log_probs.sum(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def length(self):
        return len(self.memory)

    def sample(self, batch_size):
        batchs = random.sample(self.memory, batch_size)
        batch_s, batch_a, batch_r, batch_s_, batch_done = map(np.array, zip(*batchs))
        return batch_s, batch_a, batch_r, batch_s_, batch_done


def to_tensor(value):
    if value.ndim == 1:
        value = value[np.newaxis, :]
    return torch.tensor(value, dtype=torch.float32)


