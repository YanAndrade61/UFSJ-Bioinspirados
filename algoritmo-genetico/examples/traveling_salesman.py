import numpy as np
from typing import Callable
from src.GA_base import GABase

class TravelingSalesmanGA(GABase):
    """
    Solves the Traveling Salesman Problem (TSP) using a Genetic Algorithm.

    This class extends GABase to find the shortest route visiting each city 
    exactly once and returning to the origin. Each individual represents a 
    different order to pass trougth the cities.
    """
    
    def __init__(self, n_individuals: int = 500, n_genes: int = 10, 
                otimizer: Callable[[np.ndarray], int] = np.argmin, n_generations: int = 500,
                mutation_rate: float = 0.1, distance_matrix: np.ndarray = None):
        """
        Initializes the binary function optimizer.

        Args:
            distance_matrix (np.ndarray): The distance matrix of all cities. 
        """

        self.distance_matrix = distance_matrix
        super().__init__(n_individuals, n_genes, otimizer, n_generations, mutation_rate)

    def create_individuals(self) -> np.ndarray:
        """
        Creates the initial population of individuals.

        Each individual is represented by a sequence of cities.

        Returns:
            np.ndarray: The initial population of individuals.
        """

        return np.array([np.random.choice(self.n_genes, self.n_genes, replace=False)
                        for _ in range(self.n_individuals)])
    
    def fitness(self, individuals: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of each individual in the population.

        Args:
            individuals (np.ndarray): The population of individuals to evaluate.

        Returns:
            np.ndarray: The fitness values for each individual.
        """

        individuals_fitness = np.empty(self.n_individuals, dtype=float)

        for i,ind in enumerate(individuals):
            path_distance = [self.distance_matrix[ind[j-1]][ind[j]]
                             for j in range(self.n_genes)]
            individuals_fitness[i] = np.sum(path_distance)

        return individuals_fitness