from collections import deque
import numpy as np
import random


# this is from the https://github.com/ikostrikov/pytorch-trpo/blob/master/running_state.py

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class ounoise():
    def __init__(self, std, action_dim, mean=0, theta=0.15, dt=1e-2, x0=None):
        self.std = std
        self.mean = mean
        self.action_dim = action_dim
        self.theta = theta
        self.dt = dt
        self.x0 = x0

    # reset the noise
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.action_dim)

    # generate noise
    def noise(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.x_prev = x
        return x


class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def length(self):
        return len(self.memory)

    def sample(self, batch_size):
        batchs = random.sample(self.memory, batch_size)
        batch_s, batch_a, batch_r, batch_s_, batch_done = map(np.array, zip(*batchs))
        return batch_s, batch_a, batch_r, batch_s_, batch_done
