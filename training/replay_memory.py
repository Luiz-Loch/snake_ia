import random
import numpy as np
from collections import deque
from snake_game import Direction


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.memory: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def add(self, state: np.ndarray, action: Direction, reward: int, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def clear(self) -> None:
        self.memory.clear()

    def is_full(self) -> bool:
        return len(self.memory) == self.capacity
