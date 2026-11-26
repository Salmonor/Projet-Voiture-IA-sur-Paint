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
    # Vitesse MOYENNE (et pas car.speed = vitesse au moment du crash).
    # max(1, ...) évite une division par zéro si la voiture n'a fait aucun pas.
    avg_speed = car.distance / max(1, car.steps)
    score = car.distance + car.laps * 1000.0 + avg_speed * 10.0
    return score


def select_parent(genomes: list, fitnesses: np.ndarray) -> np.ndarray:
    """Choisit UN parent ; les meilleurs doivent avoir plus de chances.

    Méthodes classiques : tournoi (tirer k individus, garder le meilleur)
    ou roulette (probabilité proportionnelle à la fitness).
    """
    # Tournoi : on tire k individus au hasard et on garde le meilleur des k.
    # Les bons gagnent plus souvent, mais les moins bons ont encore une chance
    # -> la population garde de la diversité.
    k = min(3, len(genomes))           # k grand = on favorise beaucoup les meilleurs
    candidates = np.random.choice(len(genomes), size=k, replace=False)
    # argmax donne la position PARMI LES CANDIDATS -> on la retraduit en indice de population
    winner = candidates[np.argmax(fitnesses[candidates])]
    return genomes[winner]


def crossover(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """Crée un enfant en mélangeant deux génomes (même taille).

    Idée simple : pour chaque gène, prendre celui de A ou de B au hasard.
    """
    # Masque : True (~1 fois sur 2) -> gène de A, False -> gène de B
    mask = np.random.rand(genome_a.size) < 0.5
    return np.where(mask, genome_a, genome_b)

def mutate(genome: np.ndarray, rate: float, strength: float) -> np.ndarray:
    """Renvoie une COPIE du génome où une proportion `rate` des gènes
    reçoit un bruit gaussien d'écart-type `strength`."""
    genome_mutated = np.copy(genome)    # COPIE : ne jamais toucher au génome du parent
    for i in range(len(genome_mutated)):
        if np.random.rand() < rate:
            genome_mutated[i] += np.random.randn() * strength
    return genome_mutated


def next_generation(networks: list, fitnesses: np.ndarray, layer_sizes,
                    elite_count: int, mutation_rate: float,
                    mutation_strength: float) -> list:
    """Produit la nouvelle population (même taille que l'ancienne).

    1. Trier par fitness décroissante.
    2. Copier les `elite_count` meilleurs sans modification (en tête de liste).
    3. Compléter avec des enfants : select_parent x2 -> crossover -> mutate.
    4. Reconvertir les génomes en réseaux (NeuralNetwork.from_genome).
    """
    fitnesses = np.asarray(fitnesses, dtype=float)

    # 1. Trier (nouvelles listes, les originales ne bougent pas) par fitness décroissante
    order = np.argsort(fitnesses)[::-1]
    genomes_sorted = [networks[i].get_genome() for i in order]
    fitnesses_sorted = fitnesses[order]

    # 2. Élites : copiées telles quelles, en tête de liste
    new_genomes = genomes_sorted[:elite_count]

    # 3. Enfants jusqu'à retrouver la taille de la population
    while len(new_genomes) < len(networks):
        parent_a = select_parent(genomes_sorted, fitnesses_sorted)
        parent_b = select_parent(genomes_sorted, fitnesses_sorted)
        child = crossover(parent_a, parent_b)
        child = mutate(child, mutation_rate, mutation_strength)
        new_genomes.append(child)

    # 4. Génomes -> réseaux
    return [NeuralNetwork.from_genome(layer_sizes, g) for g in new_genomes]
