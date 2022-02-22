import numpy as np
import torch
import copy
from models import Actor, Critic
from utilities import ZFilter, ounoise, ReplayBuffer


class DDPGAgent:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        obs_dims = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        self.action_max = self.env.action_space.high[0]

        self.actor = Actor(obs_dims, self.action_dims)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dims, self.action_dims)
        self.target_critic = copy.deepcopy(self.critic)

        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.a_lr)
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.c_lr,
                                        weight_decay=self.args.critic_l2_reg)

        self.zfilter = ZFilter(shape=obs_dims)

        self.buffer = ReplayBuffer(max_size=self.args.buffer_size)
        self.noise_generator = ounoise(std=0.2, action_dim=self.action_dims)

    def learn(self):
        obs = self.env.reset()
        self.noise_generator.reset()

        nb_epochs = self.args.total_frames // (self.args.nb_rollout_steps * self.args.nb_cycles)
        for epoch in range(nb_epochs):
            for _ in range(self.args.nb_cycles):

                for _ in range(self.args.nb_rollout_steps):
                    with torch.no_grad():
                        inputs_tensor = torch.tensor(obs, dtype=torch.float32)
                        a = self.actor(inputs_tensor)
                        action = self._select_actions(a)
                    # feed actions into the environment
                    obs_, reward, done, _ = self.env.step(self.action_max * action)

                    self.buffer.store(obs, action, reward, obs_, float(done))
                    obs = obs_
                    # if done, reset the environment
                    if done:
                        obs = self.env.reset()
                        self.noise_generator.reset()

                # then start to update the network
                for _ in range(self.args.nb_train):
                    a_loss, c_loss = self._update_network()
                    # update the target network
                    self._soft_update_target_network(self.target_actor, self.actor)
                    self._soft_update_target_network(self.target_critic, self.critic)
            print('epoch:{},evaluation reward:{}'.format(epoch, self._evaluate()))

    def _update_network(self):
        samples = self.buffer.sample(self.args.batch_size)
        obses, actions, rewards, obses_next, dones = samples

        norm_obses_tensor = torch.tensor(obses, dtype=torch.float32)
        norm_obses_next_tensor = torch.tensor(obses_next, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            actions_next = self.target_actor(norm_obses_next_tensor)
            q_next_value = self.target_critic(norm_obses_next_tensor, actions_next)
            target_q_value = rewards_tensor + (1 - dones_tensor) * self.args.gamma * q_next_value
        # the real q value
        real_q_value = self.critic(norm_obses_tensor, actions_tensor)
        critic_loss = (real_q_value - target_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor(norm_obses_tensor)
        actor_loss = -self.critic(norm_obses_tensor, actions_real).mean()
        # start to update the network
        self.a_optim.zero_grad()
        actor_loss.backward()
        self.a_optim.step()
        # update the critic network
        self.c_optim.zero_grad()
        critic_loss.backward()
        self.c_optim.step()
        return actor_loss.item(), critic_loss.item()

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _select_actions(self, a):
        action = a.cpu().numpy().squeeze()
        action = action + self.noise_generator.noise()
        action = np.clip(action, -1., 1.)
        return action

    def _evaluate(self):
        self.actor.eval()
        obs = self.env.reset()
        episode_reward = 0
        while True:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                a = self.actor(obs_tensor)
                action = a.cpu().numpy().squeeze()
            obs_, reward, done, _ = self.env.step(action * self.action_max)
            episode_reward += reward

            obs = obs_
            if done:
                break
        self.actor.train()
        return episode_reward
