import numpy as np
from typing import Callable
from src.selection.abstract_selection import AbstractSelection

class RouletteSelection(AbstractSelection):

    def select(self, fitness: np.ndarray, otimizer: Callable[[np.ndarray], int]) -> np.ndarray:
        """Performs roulette selection on a population.

        This method implements roulette selection, where individuals are selected
        proportionally to their fitness values. The probability of selection for each
        individual is determined by its fitness relative to the total fitness of the
        population. Higher fitness values increase the likelihood of selection.

        Args:
            fitness (np.ndarray): The fitness values of the individuals.
            otimizer (Callable[[np.ndarray], int]): The function to select the best individuals,
                either np.argmin() or np.argmax().

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
