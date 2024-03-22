from src.mutation.abstract_mutation import AbstractMutation
import numpy as np

class AmplificationMutation(AbstractMutation):
    """
    Implements Amplification Mutation for genetic algorithms.

    Amplificarion mutation multiplies genes in individuals with a definied
    value, distorting it to a greater or lesser degree.

    Parameters:
        alpha (float): The distorcion intensity of amplification mutation.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def mutate(self, individuals: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Performs amplification mutation on a population of individuals.

        Randomly selects genes based on the mutation rate and multiplies them
        by distorcion tax.

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
                    individuals[i][j] += (individuals[i][j]*self.alpha)

        return individuals