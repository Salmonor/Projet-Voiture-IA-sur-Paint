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
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)
        self.speed = 0.0
        self.alive = True
        self.distance = 0.0
        self.laps = 0
        self.steps = 0
        # Distance au moment du dernier tour validé (voir update, point 5)
        self._distance_at_last_lap = 0.0

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
        # 1. Une voiture crashée ne bouge plus.
        if not self.alive:
            return

        # Le réseau sort déjà des valeurs dans [-1, 1] (tanh), mais on protège quand même
        # la physique contre des commandes absurdes (ex : bug, clavier...).
        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))

        # 2. Direction. On ne tourne que si on roule : une voiture à l'arrêt qui pivote
        #    sur place, c'est irréaliste... et l'algo génétique adorerait tricher comme ça !
        if self.speed > 0:
            self.angle = (self.angle + steering * config.TURN_SPEED) % 360

        # 3. Vitesse : accélérateur/frein, puis frottement qui ralentit toujours un peu.
        #    np.clip borne la vitesse : pas de marche arrière, pas d'excès de vitesse.
        self.speed += throttle * config.ACCELERATION
        self.speed -= config.FRICTION
        self.speed = float(np.clip(self.speed, 0.0, config.MAX_SPEED))

        # 4. Déplacement. cos/sin attendent des RADIANS.
        #    Repère image : y vers le BAS, donc un angle positif fait tourner vers la droite
        #    (sens horaire à l'écran) -> cohérent avec steering = +1 = droite.
        old_pos = (self.x, self.y)
        rad = np.radians(self.angle)
        self.x += np.cos(rad) * self.speed
        self.y += np.sin(rad) * self.speed

        # 5. Statistiques.
        self.distance += self.speed
        self.steps += 1
        # Piège : la voiture démarre JUSTE AVANT la ligne d'arrivée, donc elle la franchit
        # au bout de quelques pas. Sans précaution, ça compterait comme un tour !
        # On n'accepte un tour que si on a roulé assez depuis le dernier.
        if circuit.crossed_finish_line(old_pos, (self.x, self.y)):
            if self.distance - self._distance_at_last_lap >= config.MIN_LAP_DISTANCE:
                self.laps += 1
                self._distance_at_last_lap = self.distance

        # 6. Sortie de route = crash. (On teste seulement le centre de la voiture :
        #    simple, et suffisant pour ce projet.)
        if not circuit.is_on_track(self.x, self.y):
            self.alive = False

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
