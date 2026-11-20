"""ÉTAPE 6 — Faire rouler toute une population et entraîner génération après génération."""
import numpy as np

from src import config
from src.car import Car
from src.genetic import compute_fitness, next_generation
from src.neural_network import NeuralNetwork


def run_generation(circuit, networks: list, max_steps: int = config.MAX_STEPS_PER_GENERATION) -> list:
    """Fait rouler une voiture par réseau jusqu'à ce qu'elles soient toutes mortes
    (ou que max_steps soit atteint). Renvoie la liste des voitures.

    À chaque pas, pour chaque voiture vivante :
        inputs = car.get_inputs(circuit)
        steering, throttle = network.forward(inputs)
        car.update(steering, throttle, circuit)
    """
    raise NotImplementedError


def train(circuit, n_generations: int = config.N_GENERATIONS,
          population_size: int = config.POPULATION_SIZE, verbose: bool = True):
    """Boucle d'entraînement complète.

    1. Créer une population de réseaux aléatoires.
    2. Pour chaque génération : run_generation -> fitness -> next_generation.
    3. Garder une trace des scores pour tracer la courbe d'apprentissage.

    Returns:
        best_network : le meilleur réseau rencontré
        history      : dict {"best": [...], "mean": [...]} une valeur par génération
    """
    raise NotImplementedError
