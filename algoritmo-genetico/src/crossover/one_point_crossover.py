from abstract_crossover import AbstractCrossover
import numpy as np

class OnePointCrossover(AbstractCrossover):

    def crossover(self, individuals: np.ndarray, parents: np.ndarray) -> np.ndarray:
        new_individuals = []
        for i in range(0, individuals.shape[0], 2):  # Iterate over pairs of individuals
            parent1, parent2 = individuals[parents[i]], individuals[parents[i+1]]

            n_genes = parent1.shape[0]
            # Randomly choose a crossover point
            point = np.random.randint(1, n_genes - 2)

            # Create offspring by swapping from the crossover point
            offspring1 = np.copy(parent1)
            offspring2 = np.copy(parent2)
            offspring1[:point], offspring2[:point] = parent2[:point], parent1[:point]

            new_individuals.append(offspring1)
            new_individuals.append(offspring2)

        return np.array(new_individuals)
