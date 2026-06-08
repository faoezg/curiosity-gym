from dataclasses import dataclass
from collections import deque
import random

from experiments.components import Transition

@dataclass
class OptionTransition(Transition):
    done: bool
    option_idx: int

class OptionReplayBuffer():
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer: deque = deque(maxlen=size)
       

    def add(self, transition: OptionTransition) -> None:
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> list[OptionTransition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def is_fully_populated(self) -> bool:
        return len(self.buffer) == self.size

    def __len__(self) -> int:
        return len(self.buffer)
