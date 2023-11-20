import numpy as np
import random
from collections import deque

class Buffer:

    def __init__(self, size = 100000): 
        # initialize buffer
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.maxDistSize = 200
        self.len = 0
        self.pos = 0
        self.cont = 0
        self.t = [0]*self.maxDistSize

    def sample(self, count):
        # sample a random batch from the replay buffer
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def add(self, s, a, r, s1):
        # add new memory buffer
        transition = (s,a,r,s1)
        self.t[self.cont%self.maxDistSize] = self.pos
        self.cont += 1
        self.pos = self.cont%self.maxSize
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)
