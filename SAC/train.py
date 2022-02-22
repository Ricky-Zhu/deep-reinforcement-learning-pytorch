import gym
from agent import SAC
from argments import get_args


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    args = get_args()
    agent = SAC(env,args)

    agent.learn()
    env.close()