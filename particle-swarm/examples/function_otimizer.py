import numpy as np
from typing import Callable
from src.PSO_base import PSOBase

class FunctionOtimizerPSO(PSOBase):
    """
    Optimizes functions using particle swarm optimization.

    This class extends the PSOBase to work with functions that can be 
    represented by positions of each particle.

    """

    def __init__(self, n_particles: int, n_dim: int, 
                 velocity_range: tuple, position_range: tuple, 
                 inertia: float, cognitive: float, social: float,
                 otimizer: Callable[[np.ndarray], int], n_generations: int = 500,
                 function: Callable[[np.ndarray], int] = None,
            ):

       self.function = function
       super().__init__(n_particles, n_dim, velocity_range, position_range, 
                        inertia, cognitive, social, otimizer, n_generations)

    def create_particles(self) -> np.ndarray:
        """
        Creates the initial swam of particles.

        Each particle is represented by a position array.

        Returns:
            np.ndarray: The initial swarm of particles.
        """

        return np.random.uniform(self.position_range[0], self.position_range[1], (self.n_particles, self.n_dim))


    def fitness(self, particles: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of each particle in the swarm.

        Args:
            particles (np.ndarray): The swarm of particles to evaluate.

        Returns:
            np.ndarray: The fitness values for each particle.
        """

        fitness = np.empty(self.n_particles, dtype=float)

        for i, p in enumerate(particles):
            fitness[i] = self.function(p)

        return fitness
