import numpy as np
from src.ACO_base import ACOBase

class TravelingSalesmanACO(ACOBase):
    """
    Solves the Traveling Salesman Problem (TSP) using the Ant Colony Optimization Algorithm.

    This class extends ACOBase to find the shortest route visiting each city 
    exactly once and returning to the origin. Each individual represents a 
    different order to pass trougth the cities.
    """
    
    def __init__(self, n_ants: int, n_paths: int, 
                n_generations: int = 500, 
                alpha: float = 1.0, beta: float = 1.0,
                distance_matrix: np.ndarray = None):

        self.distance_matrix = distance_matrix
        self.distance_matrix[self.distance_matrix == 0] = 1

        self.alpha = alpha
        self.beta = beta
        super().__init__(n_ants, n_paths, n_generations)

    def create_ants(self, pheronomes: np.ndarray) -> np.ndarray:
        """
        Creates the initial population of ants.

        Each ant is represented by a sequence of cities.

        Returns:
            np.ndarray: The initial population of ants.
        """

        ants = np.zeros((self.n_ants, self.n_paths), dtype=int)
        start = -1

        for i in range(self.n_ants):
            ant = ants[i]
            ant[0] = (start:=(start+1)%self.n_paths)
            neighbors = np.ones((self.n_paths), dtype=int)
            neighbors[ant[0]] = 0
            for j in range(1, self.n_paths):
                weights = neighbors * pow(pheronomes[ant[j-1]],self.alpha) * pow(1/self.distance_matrix[ant[j-1]],self.beta) 
                weights = weights/sum(weights)
                ant[j] = np.random.choice(range(self.n_paths), p = weights)
                neighbors[ant[j]] = 0

        return ants
    
    def fitness(self, ants: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of each ant in the colony.

        Args:
            ants (np.ndarray): The colony of ants to evaluate.

        Returns:
            np.ndarray: The fitness values for each ant.
        """

        ants_fitness = np.empty(self.n_ants, dtype=float)

        for i,ant in enumerate(ants):
            path_distance = [self.distance_matrix[ant[j-1]][ant[j]]
                             for j in range(self.n_paths)]
            ants_fitness[i] = np.sum(path_distance)

        return ants_fitness