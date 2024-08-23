# Ant Colony Optimization

## What Are They?

Ant colony optimization is a population-based metaheuristic inspired by the foraging behavior of ants. Many ant species are almost blind, so communication between ants is done through a chemical substance called pheromone. In some species, pheromone is used to create ant trails.

## Implemented Modules

### Pheromone Update

The pheromone update function, update the pheromone matrix based on the path of each ant, it also has lead with the evaporation of pheromone.

- **Standart:** Increase the pheromone in the path of all the ants.
- **Elitism:** Increase the pheromone in the path of all the ants and give a higher weight for the best ant.
- **Rank-based:** Increase the pheromone in the path of just the best K ants.

## Examples

- **Traveling Salesman Problem:** Finding the minimum path that pass all the cities and return to origin with a order representation of ants.

## Usage

First, install the requirements:

```bash
pip install -r requirements.txt
```

In the test folder, there are usage examples for each implemented module.