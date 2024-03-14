import numpy as np
from typing import Callable
from src.GA_base import GABase

class BinaryFunctionOtimizer(GABase):

    def __init__(self, n_individuals: int, n_genes: int, otimizer: Callable[[np.ndarray], int],
                 n_generations: int = 500, mutation_rate: float = 0.1,  n_bits: int = 6,
                 min_value: float = -1, max_value: float = 10, function: Callable[[np.ndarray], int] = None):
        self.n_bits = n_bits
        self.min_value = min_value        
        self.max_value = max_value
        self.function = function        
        super().__init__(n_individuals, n_genes, otimizer, n_generations, mutation_rate)

    def create_individuals(self) -> np.ndarray:
        individuals = np.random.randint(
            low = 0, 
            high = 2,
            size = (self.n_individuals, self.n_genes)
        )
        
        return individuals

    def fitness(self, individuals: np.ndarray) -> np.ndarray:

        ind_values = self.binary2value(individuals)
        ind_fitness = np.empty(self.n_individuals, dtype=float)

        for i,ind in enumerate(ind_values):
            ind_fitness[i] = self.function(ind)

        return ind_fitness

    def binary2value(self, individuals: np.ndarray) -> np.ndarray:

        ind_values = []
                    
        fn = lambda x:  self.min_value \
                        + x * (self.max_value-self.min_value)/(np.power(2,self.n_bits)-1)
        
        for ind in individuals:
            values = []
            for i in range(0, self.n_genes, self.n_bits): 
                values.append(
                    np.array(fn(int("".join(str(bit) for bit in ind[i:i+self.n_bits]),2)))
                )

            ind_values.append(values)
        
        return np.array(ind_values)