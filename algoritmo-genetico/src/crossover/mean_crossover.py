from src.crossover.abstract_crossover import AbstractCrossover
import numpy as np

#TODO como cria dois individuos assim
class MeanCrossover(AbstractCrossover):
    """
    Implements Mean Crossover for genetic algorithms.

    Mean crossover creates new offspring by the mean between 
    the parents' genes

    """

    def crossover(self, individuals: np.ndarray, parents: np.ndarray) -> np.ndarray:
        """
        Performs blend crossover on a population of individuals.

        This method selects pairs of parents from the population and creates
        two new offspring individuals for each pair using mean crossover.

        Args:
            individuals (np.ndarray): The population of individuals.
            parents (np.ndarray): The indices of the parent individuals selected
                                  for crossover.

        Returns:
            np.ndarray: The new offspring individuals.
        """
        new_individuals = []
        n_individuals, n_genes = individuals.shape

        for i in range(0, n_individuals, 2):
            parent1, parent2 = individuals[parents[i]], individuals[parents[i + 1]]
            
            offspring1 = np.random.uniform(linf, lsup)
            offspring2 = np.random.uniform(linf, lsup)

            new_individuals.append(offspring1)
            new_individuals.append(offspring2)

        return np.array(new_individuals)
