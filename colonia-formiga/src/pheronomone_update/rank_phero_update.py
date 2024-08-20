from src.pheronomone_update.abstract_phero_update import AbstractPheroUpdate
import numpy as np

class RankPheroUpdate(AbstractPheroUpdate):
    """Updates the pheromone matrix based on the ants' solutions and the rank of best solution found.
    
    Increase the pheromones in the path of the best K ants.
    
    Args:
        Q (int): The pheromone constant.
        evaporation_rate (float): The rate at which the pheromone evaporates.
        rank (float): The rate at which the pheromone of the best path found so far is updated.
        
    """
    def __init__(self, Q: int, evaporation_rate: float = 0.1, rank: float = 5) -> None:
        self.rank = rank
        super().__init__(Q, evaporation_rate)

    def update(self, pheromones: np.ndarray, ants: np.ndarray, fitness: np.ndarray, **kwargs) -> np.ndarray:

        pheromones = (1 - self.evaporation_rate) * pheromones
        best_ranks = np.argsort(fitness)[:self.rank]

        for r,k in enumerate(best_ranks):
            ant = ants[k]
            for idx, j in enumerate(ant):
                i = ant[idx-1]
                pheromones[i][j] += (self.rank-r)*(self.Q / fitness[j])
                if k == best_ranks[0]: pheromones[i][j] += self.rank*(self.Q / fitness[j])

        return pheromones
