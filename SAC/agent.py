from models import QNet, Actor, ValueNet
import torch
from torch.distributions.normal import Normal
import numpy as np
import copy
from utilities import Buffer


class SACAgent:
    def __init__(self, args, env):
        self.env = env
        self.args = args

        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_max = self.env.action_space.high
        self.device = self.args.device
        self.alpha = 0.2

        # set the networks
        self.q1 = QNet(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
        self.q2 = QNet(obs_dim=self.obs_dim, action_dim=self.action_dim, init_method=True).to(self.device)

        self.value_net = ValueNet(obs_dim=self.obs_dim).to(self.device)
        self.target_value_net = copy.deepcopy(self.value_net).to(self.device)

        self.actor = Actor(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)

        # set the optimizers
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=self.args.lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=self.args.lr)
        self.v_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.args.lr)
        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)

        # set the replay buffer
        self.buffer = Buffer(buffer_size=self.args.buffer_size, batch_size=self.args.batch_size)

        self.env_step = 0

    def learn(self):
        obs = self.env.reset()
        for i in range(self.args.pre_training_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                mean, std = self.actor(obs_tensor)
                action = self.select_action(mean, std)

            next_obs, reward, done, _ = self.env.step(action * self.action_max)
            self.buffer.add((obs, action, reward, next_obs, done))
            obs = next_obs
            if done:
                obs = self.env.reset()

        for i in range(self.args.iterations):
            obs = self.env.reset()
            while True:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    mean, std = self.actor(obs_tensor)
                    action = self.select_action(mean, std)

                next_obs, reward, done, _ = self.env.step(action * self.action_max)

                self.buffer.add((obs, action, reward, next_obs, float(done)))
                self.update()
                obs = next_obs
                if self.env_step % self.args.evaluation_steps == 0:
                    eval_rew = self.evaluation()
                    print('environment step:{}, eval_rew:{}'.format(self.env_step, eval_rew))
                if done:
                    break

    def update(self):
        b_s, b_a, b_r, b_s_, b_done = self.buffer.sample()
        b_s = torch.tensor(b_s, dtype=torch.float32).to(self.device)
        b_a = torch.tensor(b_a, dtype=torch.float32).to(self.device)
        b_r = torch.tensor(b_r, dtype=torch.float32).to(self.device)
        b_s_ = torch.tensor(b_s_, dtype=torch.float32).to(self.device)
        b_done = torch.tensor(b_done, dtype=torch.float32).to(self.device)

        # update the value net
        with torch.no_grad():
            _, actions, log_probs = self.actor.sample_action(b_s)
            q_values = torch.min(self.q1(b_s, actions), self.q2(b_s, actions))
            v_target = q_values - self.alpha * log_probs
        real_values = self.value_net(b_s)
        v_loss = (real_values - v_target).pow(2).mean()
        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        # update the q net
        with torch.no_grad():
            q_target = b_r + self.args.gamma * self.target_value_net(b_s_) * (1. - b_done)
        q1_loss = (self.q1(b_s, b_a) - q_target).pow(2).mean()
        q2_loss = (self.q2(b_s, b_a) - q_target).pow(2).mean()

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # update the actor
        pre_tanh_actions, repara_actions, repa_log_probs = self.actor.sample_action(b_s)
        repa_q_values = torch.min(self.q1(b_s, repara_actions), self.q2(b_s, repara_actions))
        a_loss = (self.alpha * repa_log_probs - repa_q_values).mean()
        self.a_optim.zero_grad()
        a_loss.backward()
        self.a_optim.step()

        # sof update the target v net
        self.soft_update_networks(self.target_value_net, self.value_net)

        self.env_step += 1

    def evaluation(self):
        """only run for one seed"""
        self.env.seed(1)
        self.actor.eval()
        obs = self.env.reset()
        episode_reward = 0
        while True:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                mean, _ = self.actor(obs_tensor)
                action = torch.tanh(mean).cpu().numpy().squeeze()
            next_obs, reward, done, _ = self.env.step(action * self.action_max)
            episode_reward += reward
            obs = next_obs
            if done:
                break
        self.actor.train()
        return episode_reward

    @staticmethod
    def select_action(mean, std):
        action = Normal(mean, std).sample()
        action = torch.tanh(action)
        return action.cpu().numpy().squeeze()

    def soft_update_networks(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * source_param.data + self.args.polyak * target_param.data)
