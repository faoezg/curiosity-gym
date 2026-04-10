import numpy as np
import random
import torch
from dataclasses import dataclass
from typing import override
from experiments.components import PriorityReplayBuffer, PriorizedTransition


@dataclass
class R2D2Transition(PriorizedTransition):
    goal_state: np.ndarray
    done: bool # used to check wheter a Transition terminates an episode so that sampled trajecories do not cross episodes
    gru_hidden_states: tuple[torch.Tensor, torch.Tensor] | None = None


class SequenceReplayBuffer(PriorityReplayBuffer):
    def __init__(self,
                 size: int,
                 alpha: float = 0.9,
                 sequence_length: int = 80,
                 sequence_overlap_size: int = 40) -> None:
        super().__init__(size, alpha)
        self.sequence_length = sequence_length
        self.sequence_overlap_size = sequence_overlap_size
    
    @override
    def sample(self, batch_size: int) -> tuple[list[list[R2D2Transition]], np.ndarray, np.ndarray]: # type: ignore
        probs = np.array(self.priorities)
        probs = probs ** self.alpha
        probs /= probs.sum()

        valid_starting_transitions_indicies = []
        for i in range(len(self) - self.sequence_length + 1):
            if any(self.buffer[i + k].done for k in range(self.sequence_length - 1)):
                continue
            valid_starting_transitions_indicies.append(i)

        if (len(valid_starting_transitions_indicies) == 0):
            raise RuntimeError("No valid starting transition in SequenceBuffer")

        starter_probs = probs[valid_starting_transitions_indicies]
        starter_probs /= starter_probs.sum() # normalize probs to be a density func
        sequence_count_to_be_choosen = min(np.count_nonzero(starter_probs), batch_size)
        chosen_starter_indicies = np.random.choice(valid_starting_transitions_indicies, size=sequence_count_to_be_choosen, replace=False, p=starter_probs)

        batch_sequences = []
        batch_indicies = []
        for starter_index in chosen_starter_indicies:
            sequence = [self.buffer[starter_index + k] for k in range(self.sequence_length)]
            batch_sequences.append(sequence)
            batch_indicies.append(starter_index)
        
        beta = 0.6 # according to the R2D2 paper
        weights = (len(self) * probs[chosen_starter_indicies]) ** (-beta)
        weights /= weights.max()
        return batch_sequences, np.array(batch_indicies, dtype=np.int32), weights
