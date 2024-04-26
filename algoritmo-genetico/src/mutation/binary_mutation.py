from src.mutation.abstract_mutation import AbstractMutation
import numpy as np

class BinaryMutation(AbstractMutation):
    """
    Performs position mutation on a population of individuals.

    It works by swaping elements between a random position and another
    randomly chosen position within each individual.

    Args:
        individuals (np.ndarray): The population of individuals.
        mutation_rate (float): The probability of mutation for each element.

    Returns:
        np.ndarray: The mutated population.
    """
    def mutate(self, individuals: np.ndarray, mutation_rate: float) -> np.ndarray:
 
        n_individuals, n_genes = individuals.shape
        for i in range(n_individuals):
            for j in range(n_genes):
                if np.random.rand() <= mutation_rate:
                    individuals[i][j] = (individuals[i][j]+1)%2

        return individuals