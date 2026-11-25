"""ÉTAPE 4 — Un réseau de neurones codé à la main avec numpy (pas de PyTorch !).

Ici le réseau n'apprend PAS par descente de gradient : ses poids seront choisis
par l'algorithme génétique (étape 5). On a donc seulement besoin du "forward".
"""
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """max(0, x) élément par élément."""
    # np.maximum compare ÉLÉMENT PAR ÉLÉMENT (≠ np.max qui renvoie UN seul nombre).
    # Le 0 est "diffusé" (broadcasting) contre chaque case de x.
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    """Sortie entre -1 et 1 : parfait pour direction et accélération."""
    return np.tanh(x)


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
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        n_layers = len(layer_sizes) - 1           # [6, 8, 2] -> 2 couches de poids
        for i in range(n_layers):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            self.weights.append(np.random.randn(n_in, n_out))   # (6, 8) puis (8, 2)
            self.biases.append(np.random.randn(n_out))          # (8,)   puis (2,)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propage les entrées couche par couche.

        Pour chaque couche : x = activation(x @ W + b)
        ReLU sur les couches cachées, tanh sur la dernière.
        """
        x = inputs
        n_layers = len(self.weights)              # 2 couches, pas 3 !
        for i in range(n_layers):
            x = x @ self.weights[i] + self.biases[i]
            if i < n_layers - 1:                  # couche cachée
                x = relu(x)
            else:                                 # dernière couche (i == n_layers - 1)
                x = tanh(x)
        return x

    # ------------------------------------------------------------ Pour l'algo génétique
    def get_genome(self) -> np.ndarray:
        """Tous les poids et biais mis bout à bout dans UN SEUL vecteur 1D.

        Ordre : couche par couche -> [W1, b1, W2, b2, ...]
        (from_genome doit relire EXACTEMENT dans cet ordre)
        """
        pieces = []
        for w, b in zip(self.weights, self.biases):
            pieces.append(w.ravel())
            pieces.append(b.ravel())
        return np.concatenate(pieces)

    def _set_weights_and_biases(self, weights: list, biases: list) -> "NeuralNetwork":
        """Remplace les poids et biais du réseau, puis renvoie le réseau lui-même.

        Le `return self` permet d'enchaîner : cls(sizes)._set_weights_and_biases(w, b)
        """
        self.weights = weights
        self.biases = biases
        return self

    @classmethod
    def from_genome(cls, layer_sizes, genome: np.ndarray) -> "NeuralNetwork":
        """Opération inverse de get_genome : reconstruit un réseau depuis un vecteur.

        Indice : découper le vecteur morceau par morceau puis .reshape().
        Test : NeuralNetwork.from_genome(sizes, net.get_genome()) doit donner les mêmes sorties.
        """
        # Copie : le réseau ne doit pas partager sa mémoire avec le génome reçu
        # (sinon modifier ce génome plus tard modifierait aussi le réseau !)
        genome = np.array(genome, dtype=float)
        n = len(layer_sizes) - 1
        weights = []
        biases = []
        for k in range(n):
            n_in, n_out = layer_sizes[k], layer_sizes[k + 1]
            w_size = n_in * n_out
            b_size = n_out
            w = genome[:w_size].reshape((n_in, n_out))
            genome = genome[w_size:]
            b = genome[:b_size]
            genome = genome[b_size:]
            weights.append(w)
            biases.append(b)
        return cls(layer_sizes)._set_weights_and_biases(weights, biases)