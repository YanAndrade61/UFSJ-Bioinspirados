# Genetic Algorithms

## What Are They?

Genetic algorithms are a class of optimization algorithms inspired by the natural selection process that occurs in nature. They are widely used to solve optimization, search, and machine learning problems.

## Project

This project provides a flexible template for genetic algorithms, adaptable to various situations. It allows for data modeling, individual creation, and objective function definition as needed for the specific problem at hand.

## Implemented Modules

### Mutation

The mutation function introduces variations in the individuals of the population, allowing for the exploration of new solutions.

- **Position Mutation:** Randomly alters the value of a gene at a specific position.
- **Swap Mutation:** Randomly swaps two genes in an individual.

### Selection

Selection is the process by which the fittest individuals are chosen for reproduction.

- **Tournament Selection:** Individuals are grouped into tournaments, and the best from each tournament is selected for reproduction.

### Crossover

Crossover combines genetic information from two individuals to generate offspring.

- **Single-Point Crossover:** A crossover point is chosen, and the parts of the parents before and after this point are exchanged to generate offspring.
- **Order Crossover:** Maintains the order of genes from one parent and fills in the missing genes from the other parent.

## Examples

- **Binary Function Optimization Problem:** Finding the minimum or maximum of a function by discretizing the search space with a binary representation of individuals.

## Usage

First, install the requirements:

```bash
pip install -r requirements.txt
```

In the test folder, there are usage examples for each implemented module.