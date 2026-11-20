"""ÉTAPE 5 — L'algorithme génétique : sélection naturelle des meilleurs cerveaux.

Principe à chaque génération :
  1. On évalue chaque voiture (fitness = score).
  2. On garde les meilleures (élitisme).
  3. On crée des enfants en mélangeant les génomes de bons parents (croisement).
  4. On ajoute un peu de hasard (mutation) pour explorer de nouvelles solutions.
"""
import numpy as np

from src.neural_network import NeuralNetwork


def compute_fitness(car) -> float:
    """Score d'une voiture à la fin de la génération.

    Commence simple (ex : la distance parcourue), puis améliore-le :
    bonus pour les tours complétés ? pénalité si elle tourne en rond ?
    C'est l'un des choix les plus importants du projet !
    """
    raise NotImplementedError


def select_parent(genomes: list, fitnesses: np.ndarray) -> np.ndarray:
    """Choisit UN parent ; les meilleurs doivent avoir plus de chances.

    Méthodes classiques : tournoi (tirer k individus, garder le meilleur)
    ou roulette (probabilité proportionnelle à la fitness).
    """
    raise NotImplementedError


def crossover(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """Crée un enfant en mélangeant deux génomes (même taille).

    Idée simple : pour chaque gène, prendre celui de A ou de B au hasard.
    """
    raise NotImplementedError


def mutate(genome: np.ndarray, rate: float, strength: float) -> np.ndarray:
    """Renvoie une COPIE du génome où une proportion `rate` des gènes
    reçoit un bruit gaussien d'écart-type `strength`."""
    raise NotImplementedError


def next_generation(networks: list, fitnesses: np.ndarray, layer_sizes,
                    elite_count: int, mutation_rate: float,
                    mutation_strength: float) -> list:
    """Produit la nouvelle population (même taille que l'ancienne).

    1. Trier par fitness décroissante.
    2. Copier les `elite_count` meilleurs sans modification (en tête de liste).
    3. Compléter avec des enfants : select_parent x2 -> crossover -> mutate.
    4. Reconvertir les génomes en réseaux (NeuralNetwork.from_genome).
    """
    raise NotImplementedError
