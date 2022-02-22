import torch
import gym
from argments import get_args
from agent import PPOAgent
import numpy as np
import os
import cv2

env = gym.make('Acrobot-v1')
path = '../PPO/models1.pt'
checkpoint = torch.load(path)
args = get_args()
agent = PPOAgent(env,args)


agent.load()

image_counter = 0
obs = agent.zfilter(env.reset())
frame = env.render(mode="rgb_array")
while True:
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        probs = agent.actor_net(obs_tensor)
        probs = probs.cpu().numpy()
        action = np.argmax(probs)

    next_obs, reward, done, _ = env.step(action)
    frame = env.render(mode="rgb_array")

    obs = agent.zfilter(next_obs, update=False)

    if done:
        obs = agent.zfilter(env.reset(), update=False)

