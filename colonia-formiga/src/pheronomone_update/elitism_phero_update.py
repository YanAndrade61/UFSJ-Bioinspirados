from src.pheronomone_update.abstract_phero_update import AbstractPheroUpdate
import numpy as np

class ElitismPheroUpdate(AbstractPheroUpdate):
    """Updates the pheromone matrix based on the ants' solutions and the best solution found.
    
    Increases the weight of the best path found so far.
    
    Args:
        Q (int): The pheromone constant.
        evaporation_rate (float): The rate at which the pheromone evaporates.
        best_rate (float): The rate at which the pheromone of the best path found so far is updated.
        
    """
    def __init__(self, Q: int, evaporation_rate: float, best_rate: float) -> None:
        self.best_rate = best_rate
        super().__init__(Q, evaporation_rate)

    def update(self, pheromones: np.ndarray, ants: np.ndarray, fitness: np.ndarray, **kwargs) -> np.ndarray:

        pheromones = (1 - self.evaporation_rate) * pheromones
        best = np.argmin(fitness)
        for k, ant in enumerate(ants):
            for idx, j in enumerate(ant):
                i = ant[idx-1]
                pheromones[i][j] += self.Q / fitness[j]
                if k == best: pheromones[i][j] += self.best_rate*(self.Q / fitness[j])

        return pheromones
