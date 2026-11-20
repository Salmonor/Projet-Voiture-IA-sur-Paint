"""ÉTAPE 4 — Un réseau de neurones codé à la main avec numpy (pas de PyTorch !).

Ici le réseau n'apprend PAS par descente de gradient : ses poids seront choisis
par l'algorithme génétique (étape 5). On a donc seulement besoin du "forward".
"""
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """max(0, x) élément par élément."""
    raise NotImplementedError


def tanh(x: np.ndarray) -> np.ndarray:
    """Sortie entre -1 et 1 : parfait pour direction et accélération."""
    raise NotImplementedError


class NeuralNetwork:
    """Perceptron multicouche.

    Exemple : layer_sizes = [6, 8, 2]
      -> une matrice de poids (6, 8) + un biais (8,) pour la couche cachée,
      -> une matrice de poids (8, 2) + un biais (2,) pour la sortie.

    Attributs :
        self.layer_sizes : list[int]
        self.weights     : list[np.ndarray]  — une matrice par couche
        self.biases      : list[np.ndarray]  — un vecteur par couche
    """

    def __init__(self, layer_sizes):
        # TODO : stocker layer_sizes et initialiser poids/biais au hasard
        # (np.random.randn suffit pour commencer).
        raise NotImplementedError

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propage les entrées couche par couche.

        Pour chaque couche : x = activation(x @ W + b)
        ReLU sur les couches cachées, tanh sur la dernière.
        """
        raise NotImplementedError

    # ------------------------------------------------------------ Pour l'algo génétique
    def get_genome(self) -> np.ndarray:
        """Tous les poids et biais mis bout à bout dans UN SEUL vecteur 1D.

        Indice : np.concatenate et .ravel() / .flatten()
        """
        raise NotImplementedError

    @classmethod
    def from_genome(cls, layer_sizes, genome: np.ndarray) -> "NeuralNetwork":
        """Opération inverse de get_genome : reconstruit un réseau depuis un vecteur.

        Indice : découper le vecteur morceau par morceau puis .reshape().
        Test : NeuralNetwork.from_genome(sizes, net.get_genome()) doit donner les mêmes sorties.
        """
        raise NotImplementedError
