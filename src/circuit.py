"""ÉTAPE 1 — Le circuit : charger l'image et savoir si un point est sur la route."""
import numpy as np
from PIL import Image


class Circuit:
    """Représente le circuit à partir de deux images de même taille.

    Attributs à créer dans __init__ :
        self.image : np.ndarray (H, W, 3) — l'image couleur, pour l'affichage
        self.mask  : np.ndarray (H, W) de booléens — True là où c'est la route
    """

    def __init__(self, image_path, mask_path):
        # TODO : charger les deux images avec PIL (Image.open), les convertir en np.array.
        # Indice : le masque est en niveaux de gris -> route si pixel > 127.
        
        raise NotImplementedError

    @property
    def width(self) -> int:
        # TODO : largeur de l'image (attention : un tableau numpy est indexé [ligne, colonne] = [y, x])
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError

    def is_on_track(self, x: float, y: float) -> bool:
        """True si le point (x, y) est sur la route.

        Pièges :
          - x, y sont des float -> il faut des entiers pour indexer.
          - un point hors de l'image doit renvoyer False (pas planter !).
        """
        raise NotImplementedError

    def crossed_finish_line(self, old_pos, new_pos) -> bool:
        """True si la voiture vient de franchir la ligne d'arrivée DANS LE BON SENS.

        La ligne est verticale (config.FINISH_LINE_X, entre FINISH_LINE_Y_RANGE) et
        les voitures roulent vers la droite à cet endroit.
        Indice : old_x < ligne <= new_x, et y dans l'intervalle.
        """
        raise NotImplementedError
