from src.pheronomone_update.abstract_phero_update import AbstractPheroUpdate
import numpy as np

class StandartPheroUpdate(AbstractPheroUpdate):
    """Updates the pheromone matrix based on the ants' solutions.

    Increase the pheromones in the path of all the ants.
    
    Args:
        Q (int): The pheromone constant.
        evaporation_rate (float): The rate at which the pheromone evaporates.

    """
    def __init__(self, Q: int = 100, evaporation_rate: float = 0.5) -> None:
        super().__init__(Q, evaporation_rate)

    def update(self, pheromones: np.ndarray, ants: np.ndarray, fitness: np.ndarray, **kwargs) -> np.ndarray:

        pheromones = (1 - self.evaporation_rate) * pheromones
        for k, ant in enumerate(ants):
            for idx, j in enumerate(ant):
                i = ant[idx-1]
                pheromones[i][j] += self.Q / fitness[j]

        return pheromones
