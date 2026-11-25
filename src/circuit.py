"""ÉTAPE 1 — Le circuit : charger l'image et savoir si un point est sur la route."""
import numpy as np
from PIL import Image

from . import config


class Circuit:
    """Représente le circuit à partir de deux images de même taille.

    Attributs à créer dans __init__ :
        self.image : np.ndarray (H, W, 3) — l'image couleur, pour l'affichage
        self.mask  : np.ndarray (H, W) de booléens — True là où c'est la route
    """

    def __init__(self, image_path, mask_path):
        # TODO : charger les deux images avec PIL (Image.open), les convertir en np.array.
        # Indice : le masque est en niveaux de gris -> route si pixel > 127.
        self.image = np.array(Image.open(image_path).convert("RGB"))  # RGB : garantit 3 canaux (un PNG peut être RGBA)
        self.mask = np.array(Image.open(mask_path).convert("L")) > 127

    @property
    def width(self) -> int:
        # shape = (H, W) -> la largeur est la 2e dimension
        return self.mask.shape[1]

    @property
    def height(self) -> int:
        return self.mask.shape[0]

    def is_on_track(self, x: float, y: float) -> bool:
        """True si le point (x, y) est sur la route.

        Pièges :
          - x, y sont des float -> il faut des entiers pour indexer.
          - un point hors de l'image doit renvoyer False (pas planter !).
        """
        xi, yi = int(x), int(y)
        # Vérifier les bornes AVANT d'indexer : un indice négatif ne planterait pas
        # en numpy, il lirait silencieusement l'autre bord de l'image !
        if not (0 <= xi < self.width and 0 <= yi < self.height):
            return False
        return bool(self.mask[yi, xi])  # numpy : [ligne, colonne] = [y, x]

    def crossed_finish_line(self, old_pos, new_pos) -> bool:
        """True si la voiture vient de franchir la ligne d'arrivée DANS LE BON SENS.

        La ligne est verticale (config.FINISH_LINE_X, entre FINISH_LINE_Y_RANGE) et
        les voitures roulent vers la droite à cet endroit.
        Indice : old_x < ligne <= new_x, et y dans l'intervalle.
        """
        old_x, _ = old_pos
        new_x, new_y = new_pos
        y_min, y_max = config.FINISH_LINE_Y_RANGE
        return old_x < config.FINISH_LINE_X <= new_x and y_min <= new_y <= y_max
