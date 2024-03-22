from src.crossover.abstract_crossover import AbstractCrossover
import numpy as np


class BlendCrossover(AbstractCrossover):
    """
    Implements Blend Crossover for genetic algorithms.

    Blend crossover creates new offspring by randomly sampling values
    between the parents' genes, weighted by their fitness. Parents with
    higher fitness contribute more to the offspring's values.

    Parameters:
        alpha (float): Controls the influence of the less fit parent on the
                       sampling range.
        beta (float): Controls the influence of the fitter parent on the
                      sampling range.
    """
    def __init__(self, alpha: float = 0.01, beta: float = 0.01) -> None:
        self.alpha = alpha
        self.beta = beta
        super().__init__()


    def crossover(self, individuals: np.ndarray, parents: np.ndarray) -> np.ndarray:
        """
        Performs blend crossover on a population of individuals.

        This method selects pairs of parents from the population and creates
        two new offspring individuals for each pair using blend crossover.

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

            diff = np.abs(parent1-parent2)
            linf = np.empty(n_genes)
            lsup = np.empty(n_genes)
            for i, x in enumerate(np.array([parent1,parent2]).T):
                linf[i] = x[0] - self.beta*diff[i] if x[0] < x[[1]] else x[[1]] - self.alpha*diff[i] 
                lsup[i] = x[[1]] + self.alpha*diff[i] if x[0] < x[[1]] else x[0] + self.beta*diff[i] 
            
            offspring1 = np.random.uniform(linf, lsup)
            offspring2 = np.random.uniform(linf, lsup)

            new_individuals.append(offspring1)
            new_individuals.append(offspring2)

        return np.array(new_individuals)
