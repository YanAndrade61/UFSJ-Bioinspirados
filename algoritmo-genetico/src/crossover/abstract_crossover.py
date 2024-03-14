from abc import ABC, abstractmethod
import numpy as np

class AbstractCrossover(ABC):
    """Abstract class representing a crossover operator in a genetic algorithm.

    This class defines the interface for crossover operators. Specific crossover
    implementations (e.g., one-point crossover, two-point crossover) should
    inherit from this class and implement the `crossover` method.

    """

    @abstractmethod
    def crossover(self, individuals: np.ndarray, parents: np.ndarray) -> np.ndarray:
        """Performs crossover on a population of individuals.

        This method takes a population of individuals and a set of parent indices
        and returns a new population of offspring created through crossover.

        Args:
            individuals (np.ndarray): The population of individuals.
            parents (np.ndarray): The indices of parent individuals for 
                                  crossover.

        Returns:
            np.ndarray: The new offspring individuals.
        """
        pass
