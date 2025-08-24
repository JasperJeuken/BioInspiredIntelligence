import numpy as np
import random

from aircraft import Aircraft2D
from controller import Controller
from evaluate import evaluate_aircraft
from terrain import Terrain


class GeneticAlgorithm:
    """Genetic Algorithm for evolving aircraft controllers"""

    def __init__(self,
                 population_size: int = 50,
                 elite_fraction: float = 0.2,
                 mutation_rate: float = 0.1) -> None:
        """Create a new algorithm instance

        Args:
            population_size (int, optional): size of population per generation. Defaults to 50.
            elite_fraction (float, optional): percentage of best population. Defaults to 0.2.
            mutation_rate (float, optional): mutation rate. Defaults to 0.1.
        """
        self.population_size: int = population_size
        self.elite_fraction: int = elite_fraction
        self.mutation_rate: float = mutation_rate
        self.generation: int = 0

    def evaluate(self,
                 aircraft: list[Aircraft2D], terrain: Terrain) -> np.ndarray[np.float64]:
        """Evaluate the fitness of each aircraft in the population

        Args:
            aircraft (list[Aircraft2D]): population of aircraft
            terrain (Terrain): terrain for collision checks

        Returns:
            np.ndarray[np.float64]: scores for each aircraft
        """
        fitness_scores = []
        for ac in aircraft:
            score = evaluate_aircraft(ac, terrain)
            fitness_scores.append(score)
        return np.array(fitness_scores)
    
    def next_generation(self,
                        controllers: list[Controller],
                        fitness_scores: np.ndarray[np.float64]) -> list[Controller]:
        """Create the next generation of controllers

        Args:
            controllers (list[Controller]): current population of controllers
            fitness_scores (np.ndarray[np.float64]): scores for each controller

        Returns:
            list[Controller]: new population of controllers
        """
        elite_count = int(self.elite_fraction * self.population_size)
        sorted_idcs = np.argsort(fitness_scores)[::-1]
        new_controllers = []

        # Select elites
        for i in range(elite_count):
            new_controllers.append(controllers[sorted_idcs[i]])

        # Add mutations of best controller (for landing)
        best_controller = controllers[sorted_idcs[0]]
        for _ in range(elite_count):
            mutant = Controller()
            mutant.w1 = np.copy(best_controller.w1)
            mutant.b1 = np.copy(best_controller.b1)
            mutant.w2 = np.copy(best_controller.w2)
            mutant.b2 = np.copy(best_controller.b2)
            mutant.mutate(self.mutation_rate * 2)
            new_controllers.append(mutant)

        # Crossover and mutate elites
        while len(new_controllers) < self.population_size:
            parent1, parent2 = random.sample(new_controllers[:elite_count], 2)
            child = self.crossover(parent1, parent2)
            child.mutate(self.mutation_rate)
            new_controllers.append(child)

        self.generation += 1
        return new_controllers
    
    def crossover(self, parent1: Controller, parent2: Controller) -> Controller:
        """Crossover two controllers to create a child controller

        Args:
            parent1 (Controller): first parent
            parent2 (Controller): second parent

        Returns:
            Controller: child controller
        """
        child = Controller()
        child.w1 = (parent1.w1 + parent2.w1) / 2
        child.b1 = (parent1.b1 + parent2.b1) / 2
        child.w2 = (parent1.w2 + parent2.w2) / 2
        child.b2 = (parent1.b2 + parent2.b2) / 2
        return child
