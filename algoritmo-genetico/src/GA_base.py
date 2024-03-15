import numpy as np
from typing import Callable
from src.selection.abstract_selection import AbstractSelection
from src.mutation.abstract_mutation import AbstractMutation
from src.crossover.abstract_crossover import AbstractCrossover


class GABase:
    """
    Base class for genetic algorithm implementations.

    This class provides a framework for implementing genetic algorithms, including
    population initialization, fitness evaluation, selection, crossover, mutation,
    and simulation. It relies on abstract classes for selection, mutation,
    and crossover operators, allowing for flexibility in algorithm design.

    Concrete subclasses should implement the `create_individuals` and `fitness`
    methods to define problem-specific representation and fitness evaluation.
    """

    def __init__(self, n_individuals: int, n_genes: int, otimizer: Callable[[np.ndarray], int],
                 n_generations: int = 500, mutation_rate: float = 0.1):
       """
       Initializes the genetic algorithm base class.

       Args:
           n_individuals (int): The number of individuals in the population.
           n_genes (int): The number of genes in each individual.
           n_generations (int, optional): The number of generations to run.
           mutation_rate (float, optional): The probability of mutation.
           otimizer (Callable[[np.ndarray], int]): The function to select best individuals
                                                 np.argmin() or np.argmax.
       """
       self.n_individuals = n_individuals
       self.n_genes = n_genes
       self.n_generations = n_generations
       self.mutation_rate = mutation_rate
       self.otimizer = otimizer

       self.selection = None
       self.mutation = None
       self.crossover = None

    def simulate(self, verbose: bool = False) -> np.ndarray:
       """
       Simulates the genetic algorithm's evolution for the specified number
       of generations.

       Args:
           verbose (bool): If True, prints progress information for each generation.

       Returns:
           np.ndarray: The best individual found after the simulation.
       """

       individuals = self.create_individuals()

       for i in range(self.n_generations):

           fitness = self.fitness(individuals)
           parents = self.selection.select(fitness, self.otimizer)
           new_ind = self.crossover.crossover(individuals, parents)
           new_ind = self.mutation.mutate(new_ind, self.mutation_rate)
           new_ind[0] = individuals[self.otimizer(fitness)]  # Elitism
           individuals = np.array(new_ind)
           if verbose:
               print(f'Geracao {i}: {fitness[self.otimizer(fitness)]}')

       return individuals[0]  # Return the best individual

    def create_individuals(self) -> np.ndarray:
       """
       Creates the initial population of individuals.

       Should be implemented by subclasses to define problem-specific representation.

       Returns:
           np.ndarray: The initial population of individuals.
       """
       raise NotImplementedError("Subclasses must implement create_individuals")

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


    def set_selection(self, selection: AbstractSelection):
        self.selection = selection

    def set_mutation(self, mutation: AbstractMutation):
        self.mutation = mutation

    def set_crossover(self, crossover: AbstractCrossover):
        self.crossover = crossover
