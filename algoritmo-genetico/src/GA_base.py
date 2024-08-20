import itertools
import numpy as np
from tqdm import tqdm
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

    def __init__(self, n_individuals: int, n_genes: int, 
                 otimizer: Callable[[np.ndarray], int],
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
       self.history_individuals = []
       individuals = self.create_individuals()

       for i in range(self.n_generations):

           self.history_individuals.append(individuals)
           
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

    def grid_search(self, n_individuals: list[int], n_genes: list[int],
                    otimizer: Callable[[np.ndarray], int], n_generations: list[int],
                    mutation_rate: list[float], selection: list[AbstractSelection],
                    mutation: list[AbstractMutation], crossover: list[AbstractMutation],
                    **kwargs: dict[str, any]) -> None:
        """
        Performs a grid search over hyperparameters of the genetic algorithm.

        This method iterates through all combinations of provided hyperparameter values
        and runs the simulation for each combination. It stores the best individual and
        parameters found for each run.

        Args:
            n_individuals (list(int)): List of values for the number of individuals.
            n_genes (list(int)): List of values for the number of genes.
            otimizer (Callable[[np.ndarray], int]): The function to select best individuals
                                                    (np.argmin() or np.argmax()).
            n_generations (list(int)): List of values for the number of generations.
            mutation_rate (list(float)): List of values for the mutation rate.
            mutation (list(AbstractMutation)): List of types for the mutation operator.
            selection (list(AbstractSelection)): List of types for the selection operator.
            crossover (list(AbstractCrossover)): List of types for the crossover operator.
            **kwargs: Additional keyword arguments to pass to the subclass constructor.
        """
        steps = 10
        best_fitness = float('-inf') if otimizer == np.argmax else float('inf')
        combinations = itertools.product(n_individuals, n_genes, n_generations, mutation_rate,
                                         selection, mutation, crossover) 
        for params in tqdm(combinations):
            num_ind, num_gene, num_gen, rate, sel, mut, cross = params

            model = self.__class__(num_ind, num_gene, otimizer, num_gen, rate, **kwargs)
            model.set_selection(sel)
            model.set_crossover(cross)
            model.set_mutation(mut)

            fitness = 0
            for _ in range(steps):
                best_individual = model.simulate(verbose=False)
                fitness += model.fitness([best_individual])[0]
            fitness /= steps 

            if(((otimizer == np.argmax) & (fitness > best_fitness)) |
               ((otimizer == np.argmin) & (fitness < best_fitness))):
                best_params = {
                    'n_individuals': num_ind,
                    'n_genes': num_gene,
                    'n_generations': num_gen,
                    'otimizer': otimizer,
                    'mutation_rate': rate,
                    'selection': sel,
                    'crossover': cross,
                    'mutation': mut,
                }
                best_fitness = fitness

        return best_fitness, best_params 
    
    def keep_parents(self, n_parents: int) -> None:
        """
        For each individual verify if we keep the last individual or change to new one.

        Args:
            n_parents (int): The number of parents to keep.
        """
        self.history_individuals[-1] = self.history_individuals[-1][:n_parents]