"""Affichage : matplotlib dans le notebook, pygame pour l'animation.

Ce n'est pas du ML, donc si tu préfères te concentrer sur l'IA, demande-moi
de remplir ce fichier.
"""
import matplotlib.pyplot as plt
import numpy as np

from src import config


def show_circuit(circuit, cars=None, show_sensors: bool = False, ax=None):
    """Affiche le circuit (plt.imshow) et, optionnellement, des voitures
    (un point + un trait pour la direction) et leurs rayons capteurs."""
    raise NotImplementedError


def plot_trajectory(circuit, positions: list, ax=None):
    """Trace la trajectoire [(x, y), ...] d'une voiture sur le circuit."""
    raise NotImplementedError


def plot_history(history: dict):
    """Courbe d'apprentissage : fitness max et moyenne par génération."""
    raise NotImplementedError


def replay(circuit, network, max_steps: int = 3000):
    """Ouvre une fenêtre pygame et montre une voiture pilotée par `network`.

    Squelette pygame : init -> fenêtre -> boucle (événements, update, dessin,
    clock.tick(config.FPS)) -> pygame.quit() (sinon le notebook peut bloquer).
    """
    raise NotImplementedError
