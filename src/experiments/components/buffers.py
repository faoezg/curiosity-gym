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
    
    def sample_continues_slices(self, batch_size: int, slice_length: int, causal_break_point: int) -> list[list[Transition]]:
        segments = []
        num_segment = len(self) // causal_break_point

        for segment in range(num_segment + 1):
            start = segment * causal_break_point
            end = min((segment + 1) * causal_break_point - 1, len(self) - 1)
            segments.append((start, end))
        
        slices = []

        for _ in range(batch_size):
            random_selected_slice = random.randint(0, len(segments) - 1)
            start, end = segments[random_selected_slice]

            valid_start = start
            valid_end = end - slice_length + 1

            if (valid_start > valid_end):
                continue
            
            random_selected_start = random.randint(valid_start, valid_end)
            data_slice = list(self.buffer)[random_selected_start:random_selected_start + slice_length]
            slices.append(data_slice)

        return slices   

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
            probs = np.array(self.priorities)
            probs = probs ** self.alpha
            probs /= probs.sum()
            indicies = np.random.choice(len(self), batch_size, p=probs, replace=False)
        return indicies, probs

    def is_fully_populated(self) -> bool:
        return len(self.buffer) == self.size

    def __len__(self) -> int:
        return len(self.buffer)

 