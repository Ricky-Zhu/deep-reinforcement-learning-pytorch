import numpy as np
from models import Critic, tanh_gaussian_actor
from utils import get_action_info, ReplayBuffer, to_tensor
import torch
import os
import numpy
import copy
from tensorboardX import SummaryWriter
import datetime



class SAC:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]

        self.q1 = Critic(self.obs_dim, self.args.hidden_size, self.action_dim)
        self.q2 = Critic(self.obs_dim, self.args.hidden_size, self.action_dim)

        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        self.actor_net = tanh_gaussian_actor(self.obs_dim, self.action_dim, self.args.hidden_size,
                                             self.args.log_std_min, self.args.log_std_max)

        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=self.args.q_lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=self.args.q_lr)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.actor_lr)

        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.actor_lr)

        self.buffer = ReplayBuffer(max_size=self.args.buffer_size)

        self.action_max = self.env.action_space.high[0]
        self.global_step = 0

        # set the logger
        log_dir = os.path.join(self.args.log_path,self.env.spec._env_name)

        if os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)

    def learn(self):
        self._init_exploration()
        for episode in range(self.args.n_episodes):
            obs = self.env.reset()
            for _ in range(self.args.episode_max_steps):
                with torch.no_grad():
                    obs_tensor = to_tensor(obs)
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi).select_action()
                    action = action.cpu().numpy()[0]

                next_obs, reward, done, _ = self.env.step(self.action_max * action)
                self.buffer.store(obs, action, reward, next_obs, done)

                self.global_step += 1
                self._update_networks()

                if self.global_step % self.args.eval_every_steps == 0:
                    eval_reward = self._evaluation()
                    self.writer.add_scalar('evaluation_reward',eval_reward,self.global_step)
                    print('global_steps:{}, evaluation_reward:{}'.format(self.global_step, eval_reward))

                obs = next_obs
                if done:
                    obs = self.env.reset()
        self.writer.close()

    def _update_networks(self):
        batch_s, batch_a, batch_r, batch_s_, batch_done = self.buffer.sample(self.args.batch_size)
        batch_s = to_tensor(batch_s)
        batch_a = to_tensor(batch_a)
        batch_r = to_tensor(batch_r.reshape(-1,1))
        batch_s_ = to_tensor(batch_s_)
        batch_done = to_tensor(batch_done.reshape(-1,1))

        # update alpha
        pis = self.actor_net(batch_s)
        actions_info = get_action_info(pis)
        actions, pre_tanh_values = actions_info.select_action(reparam=True)
        log_probs = actions_info.get_log_prob(actions, pre_tanh_values)

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        alpha = torch.exp(self.log_alpha)

        # update the actor
        actor_loss = (alpha * log_probs - torch.min(self.q1(batch_s, actions), self.q2(batch_s, actions))).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the q nets
        qf1_value = self.q1(batch_s, batch_a)
        qf2_value = self.q2(batch_s, batch_a)

        with torch.no_grad():
            pis_next = self.actor_net(batch_s_)
            next_action_info = get_action_info(pis_next)
            action_next, next_pre_tanh_values = next_action_info.select_action(reparam=True)
            log_probs_next = next_action_info.get_log_prob(action_next, next_pre_tanh_values)
            target_q_next = torch.min(self.target_q1(batch_s_, action_next),
                                      self.target_q2(batch_s_, action_next)) - alpha * log_probs_next

            target_q_values = batch_r + self.args.gamma * (1 - batch_done) * target_q_next

        qf1_loss = (qf1_value - target_q_values).pow(2).mean()
        qf2_loss = (qf2_value - target_q_values).pow(2).mean()

        self.q1_optim.zero_grad()
        qf1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        qf2_loss.backward()
        self.q2_optim.step()

        self._soft_update_target_network(self.target_q1, self.q1)
        self._soft_update_target_network(self.target_q2, self.q2)

        # log the these losses
        self.writer.add_scalar('loss/q1_loss',qf1_loss.item(),self.global_step)
        self.writer.add_scalar('loss/q2_loss', qf2_loss.item(), self.global_step)
        self.writer.add_scalar('loss/actor_loss', actor_loss.item(), self.global_step)
        self.writer.add_scalar('loss/alpha_loss', alpha_loss.item(), self.global_step)
        self.writer.add_scalar('alpha', alpha.item(), self.global_step)


    def _evaluation(self):
        self.actor_net.eval()
        self.env.seed(self.args.seed * 2)
        obs = self.env.reset()
        episode_reward = 0.
        while True:
            with torch.no_grad():
                obs_tensor = to_tensor(obs)
                pi = self.actor_net(obs_tensor)
                action = get_action_info(pi).select_action()
                action = action.cpu().numpy()[0]

            next_obs, reward, done, _ = self.env.step(self.action_max * action)
            episode_reward += reward
            obs = next_obs
            if done:
                break
        self.actor_net.train()
        return episode_reward

    def _init_exploration(self):
        obs = self.env.reset()
        for _ in range(self.args.init_exploration_steps):
            with torch.no_grad():
                obs_tensor = to_tensor(obs)
                pi = self.actor_net(obs_tensor)
                action = get_action_info(pi).select_action()
                action = action.cpu().numpy()[0]

            next_obs, reward, done, _ = self.env.step(self.action_max * action)
            self.buffer.store(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                obs = self.env.reset()

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
