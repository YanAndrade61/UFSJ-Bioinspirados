import numpy as np
import random


class AGBase:

    def __init__(self, n_ind: int, sz_ind: int, n_gen: int = 500, mutate_rate: float = 0.1, greedy_rate: float = 0):
        self.n_ind = n_ind
        self.sz_ind = sz_ind
        self.n_gen = n_gen
        self.mutate_rate = mutate_rate
        self.greedy_rate = greedy_rate

    def simulate(self, verbose: bool = False) -> None:

        individuals = self.create_individuals()

        for i in range(self.n_gen):

            fitness = self.fitness(individuals)

            parents = self.tournament_selection(fitness)

            new_ind = self.point_cross(individuals, parents)

            new_ind = self.mutate_pos(new_ind)

            new_ind[0] = individuals[np.argmin(fitness)]

            individuals = np.array(new_ind)

            if verbose:
                print(f'Geracao: {i}: {np.min(fitness)}')

        return individuals[0]

    def create_individuals(self) -> list:
        pass

    def fitness(self, ind: list) -> list:
        pass

    def tournament_selection(self, fitness: list) -> list:
        parents = []
        for i in range(self.n_ind):
            while True:
                choosen = np.random.choice(self.n_ind, 2, replace=False)
                best = choosen[np.argmin([fitness[j] for j in choosen])]
                if (i % 2 != 0) or (i == 0) or best != parents[-1]:
                    parents.append(best)
                    break
        return parents

    def ox_cross(self, individuals: list, parents: list):
        new_individuals = []
        for i in range(0, self.n_ind, 2):
            a = individuals[parents[i]]
            b = individuals[parents[i+1]]

            point1 = random.randint(1, self.sz_ind-2)
            point2 = random.randint(point1, self.sz_ind-1)

            a_reorder = np.concatenate([a[point2:], a[:point2]])
            b_reorder = np.concatenate([b[point2:], b[:point2]])

            new_ind1 = [-1] * self.sz_ind
            new_ind2 = [-1] * self.sz_ind
            new_ind1[point1:point2] = b[point1:point2]
            new_ind2[point1:point2] = a[point1:point2]

            cross_list = [(a_reorder, new_ind1), (b_reorder, new_ind2)]
            for parent, new_ind in cross_list:
                cont = point2
                for n in parent:
                    if cont == len(a):
                        cont = 0
                    if n not in new_ind:
                        new_ind[cont] = n
                        cont += 1

            new_individuals.append(new_ind1)
            new_individuals.append(new_ind2)

        return new_individuals
    
    def point_cross(self, individuals: list, parents: list):
        new_individuals = []
        for i in range(0, self.n_ind, 2):
            a = individuals[parents[i]]
            b = individuals[parents[i+1]]

            point = random.randint(1, self.sz_ind-2)

            new_ind1 = np.copy(a)
            new_ind2 = np.copy(b)
            new_ind1[:point],new_ind2[:point] = b[:point],a[:point]

            new_individuals.append(new_ind1)
            new_individuals.append(new_ind2)

        return new_individuals
    
    def mutate_ind(self, individuals: list) -> list:

        for i in range(self.n_ind):
            rate = random.random()
            if rate <= self.mutate_rate:
                pos1, pos2 = np.random.choice(self.sz_ind, 2, replace=False)
                
                individuals[i][pos1], individuals[i][pos2] = \
                individuals[i][pos2], individuals[i][pos1]

        return individuals

    def mutate_pos(self, individuals: list) -> list:

        for i in range(self.n_ind):
            for j in range(self.sz_ind):
                rate = random.random()
                if rate <= self.mutate_rate:
                    pos1 = np.random.choice(self.sz_ind, 1, replace=False)
                    individuals[i][j], individuals[i][pos1[0]] = \
                    individuals[i][pos1[0]], individuals[i][j]

        return individuals
