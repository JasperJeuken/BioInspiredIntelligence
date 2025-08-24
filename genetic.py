import numpy as np
import random

from aircraft import Aircraft2D
from controller import Controller
from evaluate import evaluate_aircraft
from terrain import Terrain


class GeneticAlgorithm:

    def __init__(self,
                 population_size: int = 50,
                 elite_fraction: float = 0.2,
                 mutation_rate: float = 0.1) -> None:
        self.population_size: int = population_size
        self.elite_fraction: int = elite_fraction
        self.mutation_rate: float = mutation_rate
        self.generation: int = 0

    def evaluate(self,
                 aircraft: list[Aircraft2D], terrain: Terrain) -> np.ndarray[np.float64]:
        fitness_scores = []
        for ac in aircraft:
            score = evaluate_aircraft(ac, terrain)
            fitness_scores.append(score)
        return np.array(fitness_scores)
    
    def next_generation(self,
                        controllers: list[Controller],
                        fitness_scores: np.ndarray[np.float64]) -> list[Controller]:
        elite_count = int(self.elite_fraction * self.population_size)
        sorted_idcs = np.argsort(fitness_scores)[::-1]

        new_controllers = []
        for i in range(elite_count):
            new_controllers.append(controllers[sorted_idcs[i]])
        while len(new_controllers) < self.population_size:
            parent1, parent2 = random.sample(new_controllers[:elite_count], 2)
            child = self.crossover(parent1, parent2)
            child.mutate(self.mutation_rate)
            new_controllers.append(child)

        self.generation += 1
        return new_controllers
    
    def crossover(self, parent1: Controller, parent2: Controller) -> Controller:
        child = Controller()
        child.w1 = (parent1.w1 + parent2.w1) / 2
        child.b1 = (parent1.b1 + parent2.b1) / 2
        child.w2 = (parent1.w2 + parent2.w2) / 2
        child.b2 = (parent1.b2 + parent2.b2) / 2
        return child
