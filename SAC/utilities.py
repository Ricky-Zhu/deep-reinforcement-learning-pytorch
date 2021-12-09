from collections import deque
import numpy as np
import random


class Buffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, transition):
        """ transition = (state,action,reward,next_state,done) """
        self.buffer.append(transition)

    def sample(self):
        assert len(self.buffer) >= self.batch_size
        batches = random.sample(self.buffer, self.batch_size)
        b_s, b_a, b_r, b_s_, b_done = zip(*batches)
        return np.array(b_s), np.array(b_a), np.array(b_r).reshape(-1, 1), np.array(b_s_), np.array(b_done).reshape(-1,
                                                                                                                    1)
