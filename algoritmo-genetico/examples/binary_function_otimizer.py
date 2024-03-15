import numpy as np
from typing import Callable
from src.GA_base import GABase

class BinaryFunctionOtimizer(GABase):
    """
    Optimizes functions using binary encoding of numbers, discretizing the search space.

    This class extends the GABase genetic algorithm framework to work with functions
    that can be represented by binary-encoded individuals. It handles the encoding and
    decoding of individuals to and from their binary representations, allowing the
    algorithm to explore the search space.

    """

    def __init__(self, n_individuals: int, n_genes: int, 
                 otimizer: Callable[[np.ndarray], int], n_generations: int = 500,
                 mutation_rate: float = 0.1, n_bits: int = 6, min_value: float = -1,
                 max_value: float = 10, function: Callable[[np.ndarray], int] = None):
       """
       Initializes the binary function optimizer.

       Args:
           n_bits (int, optional): The number of bits used to represent each value.
           min_value (float, optional): The minimum value in the search space.
           max_value (float, optional): The maximum value in the search space
           function (Callable[[np.ndarray], int], optional): The function to be optimized.
       """

       self.n_bits = n_bits
       self.min_value = min_value
       self.max_value = max_value
       self.function = function
       super().__init__(n_individuals, n_genes, otimizer, n_generations, mutation_rate)

    def create_individuals(self) -> np.ndarray:
        """
        Creates the initial population of individuals.

        Each individual is represented by a binary array.

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

        individuals_values = self.binary_to_value(individuals)
        individuals_fitness = np.empty(self.n_individuals, dtype=float)

        for i,ind in enumerate(individuals_values):
            individuals_fitness[i] = self.function(ind)

        return individuals_fitness


    def binary_to_value(self, individuals: np.ndarray) -> np.ndarray:
        """
        Converts a population of binary individuals to their numerical values.

        This function efficiently converts each binary segment of an individual
        (represented by self.n_bits bits) into its corresponding numerical value
        within the defined search space ([self.min_value, self.max_value]).

        Args:
            individuals (np.ndarray): The population of individuals.

        Returns:
            np.ndarray: The population of individuals represented as numerical values.
        """

        ind_values = []
        fn = lambda x: self.min_value + x * \
                      (self.max_value - self.min_value) / (np.power(2, self.n_bits) - 1)

        for ind in individuals:
            values = []
            for i in range(0, self.n_genes, self.n_bits):
                values.append(
                    np.array(fn(int("".join(str(bit) for bit in ind[i:i + self.n_bits]), 2)))
                )
            ind_values.append(values)
        
        return np.array(ind_values)
