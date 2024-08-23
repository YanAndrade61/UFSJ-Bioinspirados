from typing import Callable
import numpy as np

class PSOBase:
    """
    Base class for particle swarm optimization.

    This class provides a framework for implementing particle swarm algorithms,
    including particles initialization, fitness evaluation, position management
    and simulation.

    Concrete subclasses should implement the `create_particles` and `fitness`
    methods to define problem-specific representation and fitness evaluation.
    """

    def __init__(self, n_particles: int, n_dim: int, 
                 velocity_range: tuple, position_range: tuple, 
                 inertia: float, cognitive: float, social: float,
                 otimizer: Callable[[np.ndarray], int], n_generations: int = 500):
        """
        Initializes the Particle Swarm Optimization (PSO) base class.

        Args:
            n_particles (int): The number of particles in the swarm.
            n_dim (int): The number of dimensions in the problem.
            velocity_range (tuple): The range of velocities for the particles.
            position_range (tuple): The range of positions for the particles.
            inertia (float): The inertia weight for the PSO algorithm.
            cognitive (float): The cognitive weight for the PSO algorithm.
            social (float): The social weight for the PSO algorithm.
            otimizer (Callable[[np.ndarray], int]): The optimization function to be used.
            n_generations (int, optional): The number of generations to run the PSO algorithm. Default is 500.
        """
        self.n_particles = n_particles
        self.n_generations = n_generations
        self.n_dim = n_dim
        self.velocity_range = velocity_range
        self.position_range = position_range
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.otimizer = otimizer

    def simulate(self, verbose: bool = False) -> np.ndarray:
        """
        Simulates the particle swarm system for the specified number of generations.

        Args:
            verbose (bool): If True, prints progress information for each generation.

        Returns:
            np.ndarray: The best path found after the simulation.
        """
        self.history_particles = []
        best_result = float('inf')
        best_particle = None

        particles = self.create_particles()
        velocity = np.zeros_like(particles)
        particles_best = particles.copy()

        for i in range(self.n_generations):

            self.history_particles.append(particles.copy())  

            fitness = self.fitness(particles) 

            self.position_update(particles, particles_best, velocity, fitness)

            if verbose:
                print(f'Geracao {i}: {fitness[self.otimizer(fitness)]}')

            if np.min(fitness) < best_result:
                best_result = np.min(fitness)
                best_particle = particles[self.otimizer(fitness)]

        print('Melhor resultado:', best_result)
        print('Melhor caminho:', best_particle)

        return best_result, best_particle

    def position_update(self, particles: np.ndarray, particles_best: np.ndarray, velocity: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Updates the velocity of a single particle. The best particle global considered is the one with the best fitness.

        Args:
            particle (np.ndarray): The current particle position.
            particle_best (np.ndarray): The best position of the current particle.
            best_particle (np.ndarray): The best position of the swarm.
            velocity (np.ndarray): The current velocity of the particle.

        Returns:
            np.ndarray: The updated velocity of the particle.
        """
        best_particle = particles[self.otimizer(fitness)]

        r1, r2 = np.random.rand(2, self.n_particles, self.n_dim)

        velocity = self.inertia * velocity \
                   + self.cognitive * r1 * (particles_best - particles) \
                   + self.social * r2 * (best_particle - particles)
        velocity = np.clip(velocity, self.velocity_range[0], self.velocity_range[1])

        particles += velocity
        particles = np.clip(particles, self.position_range[0], self.position_range[1])

        idx = np.argwhere(fitness < self.fitness(particles_best))
        particles_best[idx] = particles[idx]

    def create_particles(self) -> np.ndarray:
        """
        Creates the initial swarm of particles.

        Should be implemented by subclasses to define problem-specific representation.

        Returns:
            np.ndarray: The initial population of particles.
        """
        raise NotImplementedError(
            "Subclasses must implement create_particles")

    def fitness(self, particles: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of each particle in the swarm.

        Should be implemented by subclasses to define problem-specific fitness function.

        Args:
            particles (np.ndarray): The population of particles to evaluate.

        Returns:
            np.ndarray: The fitness values for each particle.
        """
        raise NotImplementedError("Subclasses must implement fitness")