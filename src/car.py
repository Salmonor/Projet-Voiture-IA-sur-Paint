"""ÉTAPES 2 et 3 — La voiture : sa physique, puis ses capteurs (ses "yeux")."""
import numpy as np

from src import config


class Car:
    """Une voiture vue de dessus.

    État à stocker dans __init__ :
        x, y        : position (px)
        angle       : direction en degrés (0 = vers la droite, repère image : y vers le bas)
        speed       : vitesse actuelle (px / pas)
        alive       : False dès que la voiture sort de la route
        distance    : distance totale parcourue (servira à calculer le score)
        laps        : nombre de tours complétés
        steps       : nombre de pas de simulation vécus
    """

    def __init__(self, x=config.START_POSITION[0], y=config.START_POSITION[1],
                 angle=config.START_ANGLE):
        raise NotImplementedError

    # ------------------------------------------------------------ ÉTAPE 2 : physique
    def update(self, steering: float, throttle: float, circuit) -> None:
        """Avance la voiture d'un pas de simulation.

        Args:
            steering : entre -1 (gauche) et +1 (droite)
            throttle : entre -1 (frein) et +1 (accélérateur)
            circuit  : pour vérifier si on est toujours sur la route

        Étapes suggérées :
          1. Ne rien faire si la voiture est morte.
          2. Mettre à jour l'angle avec steering * config.TURN_SPEED.
          3. Mettre à jour la vitesse (accélération, frottement), la limiter entre 0 et MAX_SPEED.
          4. Déplacer : x += cos(angle) * speed, y += sin(angle) * speed  (angle en RADIANS !)
          5. Mettre à jour distance, steps, laps (circuit.crossed_finish_line).
          6. Si on n'est plus sur la route -> alive = False.
        """
        raise NotImplementedError

    # ------------------------------------------------------------ ÉTAPE 3 : capteurs
    def cast_ray(self, circuit, relative_angle: float) -> float:
        """Distance jusqu'au bord de la route dans une direction (lancer de rayon).

        relative_angle est relatif à la direction de la voiture (0 = droit devant).
        Méthode simple : avancer pas à pas le long du rayon jusqu'à sortir de la
        route ou atteindre config.SENSOR_MAX_DISTANCE.
        """
        raise NotImplementedError

    def read_sensors(self, circuit) -> np.ndarray:
        """Distances mesurées par tous les capteurs (config.SENSOR_ANGLES)."""
        raise NotImplementedError

    def get_inputs(self, circuit) -> np.ndarray:
        """Vecteur d'entrée du réseau de neurones, de taille config.N_INPUTS.

        Important en ML : NORMALISER les entrées (valeurs entre 0 et 1).
        -> distances / SENSOR_MAX_DISTANCE, vitesse / MAX_SPEED.
        """
        raise NotImplementedError
