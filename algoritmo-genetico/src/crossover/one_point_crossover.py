from src.crossover.abstract_crossover import AbstractCrossover
import numpy as np


class OnePointCrossover(AbstractCrossover):
    """Performs one-point crossover on a population of individuals.

    It works by selecting a random crossover point within each pair of
    parent individuals. Offspring are created by swapping genetic material
    between parents from the crossover point onwards.

    Args:
        individuals (np.ndarray): The population of individuals.
        parents (np.ndarray): The indices of the parent individuals selected for
                              crossover.

    Returns:
        np.ndarray: The new offspring individuals.
    """

    def crossover(self, individuals: np.ndarray, parents: np.ndarray) -> np.ndarray:
        new_individuals = []
        n_individuals, n_genes = individuals.shape

        for i in range(0, n_individuals, 2):
            parent1, parent2 = individuals[parents[i]], individuals[parents[i + 1]]

            point = np.random.randint(1, n_genes - 2)

            offspring1 = np.copy(parent1)
            offspring2 = np.copy(parent2)
            offspring1[:point], offspring2[:point] = parent2[:point], parent1[:point]

            new_individuals.append(offspring1)
            new_individuals.append(offspring2)

        return np.array(new_individuals)
