import gym
from agent import PPOAgent
from argments import get_args

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    args = get_args()
    agent = PPOAgent(env, args)
    agent.load()
    agent.learn()

    env.close()