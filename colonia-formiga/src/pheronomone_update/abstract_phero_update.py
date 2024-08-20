from abc import ABC, abstractmethod
import numpy as np

class AbstractPheroUpdate(ABC):
    """Abstract class representing a pheromone update operator in an ant colony optimization algorithm.

    This class defines the interface for pheromone update operators. Specific pheromone update
    implementations should inherit from this class and implement the `update` method.

    """
    def __init__(self, Q: int, evaporation_rate: float) -> None:
        self.Q = Q
        self.evaporation_rate = evaporation_rate

    @abstractmethod
    def update(self, pheromone_matrix: np.ndarray, ants: np.ndarray, fitness: np.ndarray, **kwargs) -> np.ndarray:
        """Updates the pheromone matrix based on the ants' solutions and the best solution found.

        This method takes the current pheromone matrix, the ants' solutions, the best solution found,
        and the evaporation rate, and returns the updated pheromone matrix.

        Args:
            pheromone_matrix (np.ndarray): The current pheromone matrix.
            ants (np.ndarray): The solutions found by the ants.
            fitness (np.ndarray): The fitness found by the ants.

        Returns:
            np.ndarray: The updated pheromone matrix.

        """
        pass
