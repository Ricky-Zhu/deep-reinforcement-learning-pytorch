import argparse


def get_args():
    parse = argparse.ArgumentParser(description='SAC')
    parse.add_argument('--hidden-size',type=int,default=128)
    parse.add_argument('--log-std-min', type=float, default=-20.)
    parse.add_argument('--log-std-max', type=float, default=2.)
    parse.add_argument('--q-lr', type=float, default=1e-3)
    parse.add_argument('--actor-lr', type=float, default=1e-3)
    parse.add_argument('--buffer-size', type=int, default=int(1e6))
    parse.add_argument('--n-episodes', type=int, default=int(1e4))
    parse.add_argument('--episode-max-steps', type=int, default=200)
    parse.add_argument('--eval_every_steps', type=int, default=200)
    parse.add_argument('--batch-size', type=int, default=64)
    parse.add_argument('--gamma', type=float, default=0.95)
    parse.add_argument('--seed', type=int, default=5)
    parse.add_argument('--init_exploration_steps', type=int, default=800)
    parse.add_argument('--polyak', type=float, default=0.95)
    parse.add_argument('--log-path', type=str, default='../SAC/data/')


    args = parse.parse_args()
    return args

