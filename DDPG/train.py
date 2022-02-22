from agent import DDPGAgent
import gym
from argments import get_args


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    args = get_args()
    agent = DDPGAgent(env, args)
    agent.learn()

    env.close()