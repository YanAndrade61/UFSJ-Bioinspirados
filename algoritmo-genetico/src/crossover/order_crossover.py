from src.crossover.abstract_crossover import AbstractCrossover
import numpy as np


class OrderCrossover(AbstractCrossover):
    """
    Performs order crossover on a population of individuals.

    It works by selecting a random segment from one parent and inserting it
    into the same position in the other parent. The remaining elements are filled
    sequentially from both parents to ensure all unique elements are included
    in the offspring.

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

            point1 = np.random.randint(1, n_genes - 2)
            point2 = np.random.randint(point1, n_genes - 1)

            parent1_reorder = np.concatenate([parent1[point2:], parent1[:point2]])
            parent2_reorder = np.concatenate([parent2[point2:], parent2[:point2]])

            offspring1, offspring2 = np.full(n_genes, -1), np.full(n_genes, -1)

            offspring1[point1:point2] = parent2[point1:point2]
            offspring2[point1:point2] = parent1[point1:point2]

            fill_index = point2
            for value in parent1_reorder:
                if value not in offspring1:
                    offspring1[fill_index] = value
                    fill_index = (fill_index + 1) % n_genes
            fill_index = point2
            for value in parent2_reorder:
                if value not in offspring2:
                    offspring2[fill_index] = value
                    fill_index = (fill_index + 1) % n_genes

            new_individuals.append(offspring1)
            new_individuals.append(offspring2)

        return np.array(new_individuals)
