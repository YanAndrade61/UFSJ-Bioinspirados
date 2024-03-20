import numpy as np
from typing import Callable
from src.GA_base import GABase

class BinaryKnapSack(GABase):
    """
    Solves the Binary KnapSack Problem (TSP) using a Genetic Algorithm.

    This class extends GABase to find the best combination of items that
    maximize the profit of the knapsack. Each individual represented by a
    binary array indicating wheter a item is inside or not.
    """
    
    def __init__(self, n_individuals: int, n_genes: int, 
                otimizer: Callable[[np.ndarray], int], n_generations: int = 500,
                mutation_rate: float = 0.1, weights: np.ndarray = None, 
                profits: np.ndarray = None, capacity: int = None):
        """
        Initializes the binary function optimizer.

        Args:
            weights (np.ndarray): The weight of each item. 
            profits (np.ndarray): The profit of each item. 
            capacity (int): The capacity of knapsack. 
        """

        self.weights = weights
        self.profits = profits 
        self.capacity = capacity 
        super().__init__(n_individuals, n_genes, otimizer, n_generations, mutation_rate)

    def create_individuals(self) -> np.ndarray:
        """
        Creates the initial population of individuals.

        Each individual is represented by a sequence of cities.

        Returns:
            np.ndarray: The initial population of individuals.
        """

        return np.random.randint(0, 2, (self.n_individuals, self.n_genes))
    
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
            ind_weight = np.sum(self.weights * ind)
            profits_ind = np.sum(self.profits * ind)

            if ind_weight < self.capacity:
                individuals_fitness[i] = profits_ind
            else:
                penalty = profits_ind*(ind_weight-self.capacity)
                individuals_fitness[i] = profits_ind - penalty

        return individuals_fitness