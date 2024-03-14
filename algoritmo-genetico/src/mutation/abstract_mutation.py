from abc import ABC, abstractmethod
import numpy as np


class AbstractMutation(ABC):
    """Abstract class representing a mutation operator in a genetic algorithm.

    This class defines the interface for mutation operators. Specific mutation
    implementations (e.g., swap mutation, position mutation) should inherit
    from this class and implement the `mutate` method.

    """

    @abstractmethod
    def mutate(self, individuals: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Applies mutation to a population of individuals.

        This method takes a population of individuals and a mutation rate, and
        returns a new population with potential mutations applied.

        Args:
            individuals (np.ndarray): The population of individuals to be mutated.
            mutation_rate (float): The probability of mutation for each gene.

        Returns:
            np.ndarray: The mutated population of individuals.

        """
        pass
