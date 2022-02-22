import  torch
import torch.nn as nn
from torch.nn import functional as F
from math import sqrt


def initialization_fc(p):
    if isinstance(p,nn.Linear):
        p.weight.data.uniform_(-1/sqrt(p.weight.shape[1]),1/sqrt(p.weight.shape[1]))
        p.bias.data.uniform_(-1/sqrt(p.weight.shape[1]),1/sqrt(p.weight.shape[1]))

class Critic(nn.Module):
    def __init__(self,input_dim,hidden_size,action_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(input_dim+action_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.q_value = nn.Linear(hidden_size,1)

        # initialize weight and bias
        self.apply(initialization_fc)

    def forward(self,s,a):
        x = torch.cat([s,a],dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.q_value(x)
        return x

class tanh_gaussian_actor(nn.Module):
    def __init__(self,input_dim,action_dim,hidden_size,log_std_min,log_std_max):
        super(tanh_gaussian_actor, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.mean = nn.Linear(hidden_size,action_dim)
        self.log_std = nn.Linear(hidden_size,action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.apply(initialization_fc)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = log_std.clamp(self.log_std_min,self.log_std_max)

        return (mean,torch.exp(log_std))