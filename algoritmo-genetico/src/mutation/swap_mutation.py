import numpy as np
from src.mutation.abstract_mutation import AbstractMutation

class SwapMutation(AbstractMutation):
    """
    Performs swap mutation on a population of individuals.

    It works by swaping elements at randomly chosen positions within each
    individual.

    Args:
        individuals (np.ndarray): The population of individuals.
        mutation_rate (float): The probability of mutation for each element.

    Returns:
        np.ndarray: The mutated population.
    """

    def mutate(self, individuals: np.ndarray, mutation_rate: float) -> np.ndarray:

        n_individuals, n_genes = individuals.shape
        for i in range(n_individuals):
            if np.random.rand() <= mutation_rate:
                pos1, pos2 = np.random.choice(n_genes, 2, replace=False)
                individuals[i][pos1], individuals[i][pos2] =\
                    individuals[i][pos2], individuals[i][pos1]

        return individuals