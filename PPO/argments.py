import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--a-lr', type=float, default=1e-3)
    parser.add_argument('--c-lr', type=float, default=3e-3)
    parser.add_argument('--total-frames', type=int, default=int(1e6))
    parser.add_argument('--nsteps', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--mini-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--clip', type=float, default=0.2)


    args = parser.parse_args()

    return args
