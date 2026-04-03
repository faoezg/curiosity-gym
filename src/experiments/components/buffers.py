from collections import deque
from dataclasses import dataclass
from curiosity_gym.utils.enums import Action

import numpy as np
import random

@dataclass
class Transition():
    state: np.ndarray
    action: Action | int | float
    reward: float
    next_state: np.ndarray

@dataclass
class PriorizedTransition(Transition):
    priority: float



class ReplayBuffer():
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer: deque = deque(maxlen=size)
    
    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def is_fully_populated(self) -> bool:
        return len(self.buffer) == self.size

    def __len__(self) -> int:
        return len(self.buffer)

class PriorityReplayBuffer():
    def __init__(self, size: int, alpha: float = 0.5) -> None:
        self.size = size
        self.buffer: deque = deque(maxlen=size)
        self.alpha = alpha
        self.priorities: deque = deque(maxlen=size)
    
    def add(self, transition: PriorizedTransition) -> None:
        self.buffer.append(transition)
        self.priorities.append(transition.priority)

    def sample(self, batch_size: int) -> tuple[list[PriorizedTransition], np.ndarray, np.ndarray]:
        indicies, probs = self._get_sample_indicies(batch_size)
        samples = [self.buffer[i] for i in indicies]
        weights = (len(self) * probs[indicies]) ** (-1.0)
        weights = weights / weights.max()
        return samples, indicies, weights
    
    def update_prioritites(self, indicies: np.ndarray, new_priorities: np.ndarray) -> None:
        for idx, new_priority in zip(indicies, new_priorities):
            self.priorities[idx] = new_priority 
       
    def _get_sample_indicies(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if (len(self) < batch_size):
            indicies = np.arange(len(self))
            probs = np.ones(len(self))/ len(self)
        else:
            try:
                probs = np.array(self.priorities)
                probs = probs ** self.alpha
                probs /= probs.sum()
                indicies = np.random.choice(len(self), batch_size, p=probs, replace=False)
            except:
                print("UUUUU", self.priorities)
        return indicies, probs

    def is_fully_populated(self) -> bool:
        return len(self.buffer) == self.size

    def __len__(self) -> int:
        return len(self.buffer)

 