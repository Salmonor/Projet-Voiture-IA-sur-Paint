"""Tous les réglages du projet au même endroit.

Ce fichier est déjà rempli : tu peux jouer avec les valeurs pour voir leur effet.
"""
from pathlib import Path

# ---------------------------------------------------------------- Chemins
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "Images"
CIRCUIT_IMAGE = IMAGES_DIR / "ClaudeCircuit.png"       # image jolie (affichage)
CIRCUIT_MASK = IMAGES_DIR / "ClaudeCircuit_mask.png"   # blanc = route, noir = dehors
CAR_IMAGE = IMAGES_DIR / "Voiture.png"

# ---------------------------------------------------------------- Circuit
# Repère image : x vers la droite, y vers le BAS, angle 0° = vers la droite.
START_POSITION = (560.0, 115.0)   # juste avant la ligne d'arrivée
START_ANGLE = 0.0                 # en degrés
FINISH_LINE_X = 605               # la ligne d'arrivée est verticale à x = 605 ...
FINISH_LINE_Y_RANGE = (75, 155)   # ... entre ces deux y

# ---------------------------------------------------------------- Voiture
CAR_LENGTH = 30                   # px (pour l'affichage)
CAR_WIDTH = 15
MAX_SPEED = 8.0                   # px par pas de simulation
ACCELERATION = 0.3                # px / pas²
FRICTION = 0.05                   # ralentissement naturel par pas
TURN_SPEED = 4.0                  # degrés par pas (braquage max)

# ---------------------------------------------------------------- Capteurs
SENSOR_ANGLES = [-90, -45, 0, 45, 90]   # degrés, relatifs à la direction de la voiture
SENSOR_MAX_DISTANCE = 200               # px

# ---------------------------------------------------------------- Réseau de neurones
N_INPUTS = len(SENSOR_ANGLES) + 1       # capteurs + vitesse
N_OUTPUTS = 2                           # [direction, accélération]
LAYER_SIZES = [N_INPUTS, 8, N_OUTPUTS]

# ---------------------------------------------------------------- Algorithme génétique
POPULATION_SIZE = 50
N_GENERATIONS = 100
ELITE_COUNT = 5                   # meilleurs individus copiés tels quels
MUTATION_RATE = 0.1               # proportion de poids modifiés
MUTATION_STRENGTH = 0.5           # écart-type du bruit ajouté
MAX_STEPS_PER_GENERATION = 2000

# ---------------------------------------------------------------- Affichage
FPS = 60
