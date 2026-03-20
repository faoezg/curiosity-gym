import numpy as np

class RademacherDistGenerator():
    def __init__(self, d_dim: int, p_replace: float = 1.0, should_return_zero_vector: bool = False) -> None:
        self.d_dim = d_dim
        self.p_replace = p_replace
        self.should_return_zero_vector = should_return_zero_vector
        self.reset()

    def _generate_sample_vector(self):
        if self.should_return_zero_vector:
            return np.zeros(self.d_dim, dtype=np.float32)
        return 2 * np.random.binomial(1, 0.5, size=self.d_dim) - 1 # taken from https://github.com/samlobel/CFN
    
    def __call__(self) -> np.ndarray:
        if self.should_return_zero_vector:
            return self._generate_sample_vector()
        new_sampling_vector = self._generate_sample_vector()
        mask = np.random.rand(self.d_dim) < self.p_replace
        self.prev_sampling_vector = np.where(mask, new_sampling_vector, self.prev_sampling_vector)
        return self.prev_sampling_vector
    
    def reset(self):
        self.prev_sampling_vector = self._generate_sample_vector()
