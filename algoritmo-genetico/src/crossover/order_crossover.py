from abstract_crossover import AbstractCrossover
import numpy as np

class OrderCrossover(AbstractCrossover):

    def crossover(self, individuals: np.ndarray, parents: np.ndarray) -> np.ndarray:
        new_individuals = []
        for i in range(0, individuals.shape[0], 2):  # Iterate over pairs of individuals
            parent1, parent2 = individuals[parents[i]], individuals[parents[i+1]]

            n_genes = parent1.shape[0]
            # Choose random points for the segment to be copied
            point1 = np.random.randint(1, n_genes - 2)
            point2 = np.random.randint(point1, n_genes - 1)

            # Create reordered versions of parents
            parent1_reorder = np.concatenate([parent1[point2:], parent1[:point2]])
            parent2_reorder = np.concatenate([parent2[point2:], parent2[:point2]])

            # Initialize offspring with -1 to mark unfilled slots
            offspring1, offspring2 = np.full(n_genes, -1), np.full(n_genes, -1)

            # Copy the chosen segment from one parent to both offspring
            offspring1[point1:point2] = parent2[point1:point2]
            offspring2[point1:point2] = parent1[point1:point2]

            # Fill remaining slots in offspring with elements not yet present
            fill_index = point2
            for value in parent1_reorder:
                if value not in offspring1:
                    offspring1[fill_index ] = value
                    fill_index = (fill_index+1) % n_genes
            fill_index = point2
            for value in parent2_reorder:
                if value not in offspring2:
                    offspring2[fill_index] = value
                    fill_index = (fill_index+1) % n_genes

            new_individuals.append(offspring1)
            new_individuals.append(offspring2)

        return np.array(new_individuals)