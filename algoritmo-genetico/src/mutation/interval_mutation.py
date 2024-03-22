from src.mutation.abstract_mutation import AbstractMutation
import numpy as np

class IntervalMutation(AbstractMutation):
    """
    Implements Interval Mutation for genetic algorithms.

    Interval mutation replaces genes in individuals with random values
    within a specified interval, introducing controlled variation.

    Parameters:
        lmin (float): The lower bound of the mutation interval.
        lmax (float): The upper bound of the mutation interval.
    """

    def __init__(self, lmin: float = -2, lmax: float = 2) -> None:
        self.lmin = lmin
        self.lmax = lmax

    def mutate(self, individuals: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Performs interval mutation on a population of individuals.

        Randomly selects genes based on the mutation rate and replaces them
        with values within the specified interval.

        Args:
            individuals (np.ndarray): The population of individuals to mutate.
            mutation_rate (float): The probability of mutation for each gene.

        Returns:
            np.ndarray: The mutated population.
        """
        n_individuals, n_genes = individuals.shape
        for i in range(n_individuals):
            for j in range(n_genes):
                if np.random.rand() <= mutation_rate:
                    new_value = np.random.uniform(self.lmin, self.lmax)
                    individuals[i][j] = new_value

        return individuals