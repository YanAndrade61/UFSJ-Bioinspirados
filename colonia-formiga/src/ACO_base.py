import itertools
import numpy as np
from tqdm import tqdm
from typing import Callable
from src.pheronomone_update.abstract_phero_update import AbstractPheroUpdate

class ACOBase:
    """
    Base class for ant colony implementations.

    This class provides a framework for implementing ant colony algorithms,
    including ant initialization, fitness evaluation, pheronomone management
    and simulation.

    Concrete subclasses should implement the `create_ants` and `fitness`
    methods to define problem-specific representation and fitness evaluation.
    """

    def __init__(self, n_ants: int, n_paths: int,
                 n_generations: int = 500):
        """
        Initializes the genetic algorithm base class.

        Args:
            n_ants (int): The number of ants in the colony.
            n_paths (int): The number of paths in the problem.
            n_generations (int, optional): The number of generations to run.
        """
        self.n_ants = n_ants
        self.n_paths = n_paths
        self.n_generations = n_generations

    def simulate(self, verbose: bool = False) -> np.ndarray:
        """
        Simulates the ant colony system for the specified number of generations.

        Args:
            verbose (bool): If True, prints progress information for each generation.

        Returns:
            np.ndarray: The best path found after the simulation.
        """
        self.history_ants = []
        ants = None
        best_result = float('inf')
        best_ant = None
        pheronomes = np.ones((self.n_paths, self.n_paths)) * 1e-6

        for i in range(self.n_generations):

            ants = self.create_ants(pheronomes)

            self.history_ants.append(ants)  

            fitness = self.fitness(ants)

            pheronomes = self.phero_update.update(pheronomes.copy(), ants, fitness)

            if verbose:
                print(f'Geracao {i}: {fitness[np.argmin(fitness)]}')

            if np.min(fitness) < best_result:
                best_result = np.min(fitness)
                best_ant = ants[np.argmin(fitness)]

        # print('Melhor resultado:', best_result)
        # print('Melhor caminho:', best_ant)

        return best_result, best_ant

    def create_ants(self) -> np.ndarray:
        """
        Creates the initial population of individuals.

        Should be implemented by subclasses to define problem-specific representation.

        Returns:
            np.ndarray: The initial population of individuals.
        """
        raise NotImplementedError(
            "Subclasses must implement create_individuals")

    def fitness(self, individuals: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of each individual in the population.

        Should be implemented by subclasses to define problem-specific fitness function.

        Args:
            individuals (np.ndarray): The population of individuals to evaluate.

        Returns:
            np.ndarray: The fitness values for each individual.
        """
        raise NotImplementedError("Subclasses must implement fitness")

    def set_phero_update(self, phero_update: AbstractPheroUpdate) -> None:
        self.phero_update = phero_update