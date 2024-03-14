from abc import ABC, abstractmethod
import numpy as np

class AbstractMutation(ABC):

    @abstractmethod
    def mutate(self, individuals: np.ndarray, mutation_rate: float) -> np.ndarray:
        pass