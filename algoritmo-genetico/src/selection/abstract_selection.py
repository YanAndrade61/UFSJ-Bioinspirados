from abc import ABC, abstractmethod
import numpy as np

class AbstractSelection(ABC):

    @abstractmethod
    def select(self, fitness: np.ndarray, n_parents: int) -> np.ndarray:
        pass