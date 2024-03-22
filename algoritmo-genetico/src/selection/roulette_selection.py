import numpy as np
from typing import Callable
from src.selection.abstract_selection import AbstractSelection

class RouletteSelection(AbstractSelection):

    def select(self, fitness: np.ndarray, otimizer: Callable[[np.ndarray], int]) -> np.ndarray:
        """Performs roulette selection on a population.

        This method implements tournament selection, where a random subset of
        individuals compete, and the fittest one becomes a parent. This process
        is repeated to select the number of parents.
        #TODO Refazer explicacao

        Args:
            fitness (np.ndarray): The fitness values of the individuals.
            otimize (Callable[[np.ndarray], int]) : The function to select best individuals
                                                    np.argmin() or np.argmax.

        Returns:
            np.ndarray: The indices of the selected parent individuals.

        """
        n_individuals = fitness.shape[0]
        parents = np.empty(n_individuals, dtype=int)
        
        prob = 1/fitness if otimizer == np.argmin else fitness
        prob = fitness / np.sum(fitness)
        
        for i in range(0, n_individuals, 2):
            candidates = np.random.choice(n_individuals, 2, replace=False, p=prob)

            parents[i] = candidates[0] if fitness[candidates[0]] > fitness[candidates[1]] else candidates[1]
            parents[i+1] = candidates[0] if fitness[candidates[0]] < fitness[candidates[1]] else candidates[1]

        return parents
