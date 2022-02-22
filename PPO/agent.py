import torch
import numpy as np
from models import Actor_net, Value_net
from torch.distributions.normal import Normal
from utilities import ZFilter
from gym.spaces import Box, Discrete
from torch.distributions import Categorical
import os


class PPOAgent:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space

        # check the env action space type
        if isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            self.action_dim = self.action_space.n

        self.actor_net = Actor_net(obs_dim=self.obs_dim, action_dim=self.action_dim,
                                   action_space=self.action_space)
        self.value_net = Value_net(obs_dim=self.obs_dim)

        self.a_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.a_lr)
        self.c_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.args.c_lr)

        # add the running estimate for the observation
        self.zfilter = ZFilter(shape=self.obs_dim, clip=5.0)

    def learn(self):
        """only for one actor"""
        num_updates = self.args.total_frames // self.args.nsteps
        ep_all_rewards = []  # for recording the evaluation episode rewards

        for i in range(num_updates):

            b_obs, b_rewards, b_actions, b_dones, b_values = [], [], [], [], []
            obs = self.zfilter(self.env.reset())
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if isinstance(self.action_space, Box):
                        mean, std = self.actor_net(obs_tensor)
                        action = self.select_action(mean, std)
                    else:
                        probs = self.actor_net(obs_tensor)
                        dist = Categorical(probs)
                        action = dist.sample().cpu().numpy().squeeze()
                    value = self.value_net(obs_tensor)
                next_obs, reward, done, _ = self.env.step(action)

                b_obs.append(obs)
                b_actions.append(action)
                b_rewards.append([reward])
                b_dones.append([float(done)])
                b_values.append(value.cpu().numpy().squeeze())

                obs = next_obs
                if done:
                    obs = self.env.reset()
                obs = self.zfilter(obs)

            # append the last obs value
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                value = self.value_net(obs_tensor)
                value = value.detach().cpu().numpy().squeeze()
            b_values.append(value)

            b_obs = np.asarray(b_obs, dtype=np.float32)
            b_actions = np.asarray(b_actions, dtype=np.float32 if isinstance(self.action_space, Box) else np.int)
            b_rewards = np.asarray(b_rewards, dtype=np.float32)
            b_dones = np.asarray(b_dones, dtype=np.float32)
            b_values = np.asarray(b_values, dtype=np.float32).reshape(-1, 1)

            self.update(b_obs, b_actions, b_rewards, b_dones, b_values)
            eval_reward = self.evaluate()
            ep_all_rewards.append(eval_reward)
            print('updates:{}/{},eval_rew:{}'.format(i, num_updates, eval_reward))
        self.save()
        np.save('../PPO/all_rewards.npy', ep_all_rewards)

    def update(self, b_obs, b_actions, b_rewards, b_dones, b_values):

        # Generalized Advantage Estimate
        b_advs = np.zeros_like(b_rewards)

        last_value = b_values[-1]
        b_values = b_values[:-1]

        next_adv = 0

        for i in reversed(np.arange(len(b_rewards))):
            delta = b_rewards[i] + self.args.gamma * (1. - b_dones[i]) * last_value - b_values[i]
            b_advs[i] = next_adv = delta + self.args.tau * self.args.gamma * (1. - b_dones[i]) * next_adv
            last_value = b_values[i]

        b_returns = b_advs + b_values
        # get the old log probs of actions
        with torch.no_grad():
            b_obs_tensor = torch.tensor(b_obs, dtype=torch.float32)
            b_actions_tensor = torch.tensor(b_actions, dtype=torch.float32 if isinstance(self.action_space,
                                                                                         Box) else torch.int64)
            if isinstance(self.action_space, Discrete):
                b_actions_tensor = b_actions_tensor.reshape(-1, 1)

            b_log_probs, _ = self.actor_net.evaluate_action_log_probs(b_obs_tensor, b_actions_tensor)
            b_log_probs = b_log_probs.detach()

        # update the networks for self.args.epochs times
        mn_batches = len(b_rewards) // self.args.mini_batch_size
        for ep in range(self.args.epochs):
            indices = np.arange(len(b_rewards))
            np.random.shuffle(indices)
            for mb in range(mn_batches):
                ind = indices[mb * self.args.mini_batch_size:(mb + 1) * self.args.mini_batch_size]
                mb_obs = torch.tensor(b_obs[ind], dtype=torch.float32)
                mb_advs = torch.tensor(b_advs[ind], dtype=torch.float32)
                # normalize the advantages
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                mb_returns = torch.tensor(b_returns[ind], dtype=torch.float32)
                mb_log_probs = b_log_probs[ind]
                mb_actions = b_actions_tensor[ind]

                # update the value net
                real_values = self.value_net(mb_obs)
                value_loss = (real_values - mb_returns).pow(2).mean()
                self.c_optim.zero_grad()
                value_loss.backward()
                self.c_optim.step()

                # update the actor net
                real_log_probs, entropy = self.actor_net.evaluate_action_log_probs(mb_obs, mb_actions)
                ratios = torch.exp(real_log_probs - mb_log_probs)
                surr1 = ratios * mb_advs
                surr2 = torch.clamp(ratios, 1. - self.args.clip, 1. + self.args.clip) * mb_advs
                a_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                self.a_optim.zero_grad()
                a_loss.backward()
                self.a_optim.step()

    @staticmethod
    def select_action(mean, std):
        action = Normal(mean, std).sample()
        return action.cpu().numpy().squeeze(axis=0)

    def save(self):
        path = self.args.save_path
        torch.save({'model': self.actor_net.state_dict(),
                    'normalizer': self.zfilter}, path)

    def load(self):
        if os.path.exists(self.args.save_path):
            checkpoint = torch.load(self.args.save_path)
            self.actor_net.load_state_dict(checkpoint['model'])
            self.zfilter = checkpoint['normalizer']
        else:
            pass

    def evaluate(self):
        self.actor_net.eval()
        obs = self.zfilter(self.env.reset(), update=False)
        episode_rew = 0
        while True:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if isinstance(self.action_space, Box):
                    action, _ = self.actor_net(obs_tensor)
                    action = action.detach().cpu().numpy().squeeze(axis=0)
                else:
                    probs = self.actor_net(obs_tensor)
                    probs = probs.cpu().numpy()
                    action = np.argmax(probs)

            next_obs, reward, done, _ = self.env.step(action)
            episode_rew += reward
            obs = self.zfilter(next_obs, update=False)

            if done:
                break
        return episode_rew
