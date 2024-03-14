import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class AbstractSelection(ABC):
    """Abstract class representing a selection operator in a genetic algorithm.

    This class defines the interface for selection operators. Specific selection
    implementations (e.g., roulette wheel selection, tournament selection) should
    inherit from this class and implement the `select` method.
    """

    @abstractmethod
    def select(self, fitness: np.ndarray, otimizer: Callable[[np.ndarray], int]) -> np.ndarray:
        """Selects parents from a population based on fitness.

        This method takes a population's fitness values, and returns the
        indices of selected parent individuals in pairs without duplication.

        Args:
            fitness (np.ndarray): The fitness values of the individuals.
            otimize (Callable[[np.ndarray], int]) : The function to select best individuals
                                                    np.argmin() or np.argmax.

        Returns:
            np.ndarray: The indices of the selected parent individuals.

        """
        pass