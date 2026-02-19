from collections import defaultdict
import numpy as np

class UnifiedCountModel():
    """
    Docstring for UnifiedCountModel
    This model does approximate any functions.
    It is table based given that in a gridworld such curosity gym
    the state-space is discrete.

    On that note, this model will treat a state as a set of three floats
    """
    def __init__(self,
                 possible_state_dim: int,
                 pseudo_prior: float = 0.01 # ensure non-zero prob, 
                 ) -> None:
        self.possible_state_dim = possible_state_dim
        self.prior = pseudo_prior
        self.visitation_dict = defaultdict(float) # ensures that unseen states return 0 when read 
        self.total_visited_states = 0
    
    def update_observation(self, state: np.ndarray) -> None:
        for cell in state:
            self.visitation_dict[tuple(cell)] += 1
            self.total_visited_states += 1
    
    def calc_visitation_prob(self, state: np.ndarray) -> float:
        log_prob = 0.0
        for cell in state:
            # calc in log-space for stability on large state-spaces
            log_prob += np.log(self._calc_cell_visitation_prob(tuple(cell)))
        return np.exp(log_prob)

    def calc_visitation_prob_after_observation(self, state: np.ndarray) -> float:
        log_prob = 0.0
        for cell in state:
            # calc in log-space for stability on large state-spaces
            log_prob += np.log(self._calc_new_state_visitation_prob(tuple(cell)))
        return np.exp(log_prob)

    def _calc_current_cell_visitation_prob(self, cell: tuple[float, float, float]) -> float:
        return self._calc_cell_visitation_prob(cell)

    def _calc_new_state_visitation_prob(self, cell: tuple[float, float, float]) -> float:
        return self._calc_cell_visitation_prob(cell, 1)
    
    
    def _calc_cell_visitation_prob(self, cell: tuple[float, float, float], observation_count: int = 0) -> float:
        """
        Docstring for _calc_current_state_visitation_prob
        
        :param cell: current state of the env
        :type cell: tuple[float, float, float]
        :param observation_count: the amount of new data samples, that is the amount of observation of a given state since the latest update
        :type: int
        :return: rho(cell) - visitation probability of the cell
        :rtype: float
        """
        pseudo_count = self.visitation_dict[cell] + observation_count + self.prior # + prior if state is unvisited
        total_visited_states = self.total_visited_states + observation_count + self.prior * self.possible_state_dim

        return pseudo_count / total_visited_states
