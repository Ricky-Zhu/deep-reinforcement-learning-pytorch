from agent import SACAgent
from argments import get_args
import gym

if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    args = get_args()
    agent = SACAgent(env=env, args=args)
    agent.learn()
    env.close()
