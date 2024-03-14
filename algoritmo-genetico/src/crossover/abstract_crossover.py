from abc import ABC, abstractmethod
import numpy as np

class AbstractCrossover(ABC):

    @abstractmethod
    def crossover(self, individuals: np.ndarray, parents: np.ndarray) -> np.ndarray:
        pass