import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--pre-training-steps', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--evaluation_steps', type=int, default=1000)
    parser.add_argument('--reward_scale', type=float, default=5.)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.95)

    args = parser.parse_args()
    return args
