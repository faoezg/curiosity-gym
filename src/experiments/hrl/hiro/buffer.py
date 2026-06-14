from collections import deque
from dataclasses import dataclass
from curiosity_gym.utils.enums import Action

import numpy as np
import random

@dataclass
class LowHIROTransition():
    state: np.ndarray
    action: Action | int | float
    reward: float
    next_state: np.ndarray
    goal: np.ndarray
    not_done: bool

@dataclass
class HighHIROTransition():
    state: np.ndarray
    action: Action | int | float
    reward: float
    next_state: np.ndarray
    goal: np.ndarray
    not_done: bool
    option_trajectorie: list[LowHIROTransition]

class HIROReplayBuffer():
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer: deque = deque(maxlen=size)
    
    def add(self, transition: HighHIROTransition | LowHIROTransition) -> None:
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> list[HighHIROTransition | LowHIROTransition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def sample_continues_slices(self, batch_size: int, slice_length: int, causal_break_point: int) -> list[list[HighHIROTransition | LowHIROTransition]]:
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
