import numpy as np
from typing import Callable
from src.GA_base import GABase

class NumericFunctionOtimizer(GABase):
    """
    Optimizes functions using numeric values.

    This class extends the GABase genetic algorithm framework to work with functions
    that can be represented by real numbers individuals.

    """

    def __init__(self, n_individuals: int = 100, n_genes: int = 3, 
                 otimizer: Callable[[np.ndarray], int] = None, n_generations: int = 500,
                 mutation_rate: float = 0.1, lmin: float = -1,
                 lmax: float = 1, function: Callable[[np.ndarray], int] = None):
       """
       Initializes the binary function optimizer.

       Args:
           n_bits (int, optional): The number of bits used to represent each value.
           lmin (float, optional): The minimum value in the search space.
           lmax (float, optional): The maximum value in the search space
           function (Callable[[np.ndarray], int], optional): The function to be optimized.
       """

       self.lmin = lmin
       self.lmax = lmax
       self.function = function
       super().__init__(n_individuals, n_genes, otimizer, n_generations, mutation_rate)

    def create_individuals(self) -> np.ndarray:
        """
        Creates the initial population of individuals.

        Each individual is represented by a binary array.

        Returns:
            np.ndarray: The initial population of individuals.
        """

        return np.random.uniform(self.lmin, self.lmax, (self.n_individuals, self.n_genes))


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
            individuals_fitness[i] = self.function(ind)

        return individuals_fitness
