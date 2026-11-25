# Voitures IA

Des voitures apprennent à faire le tour d'un circuit grâce à un **réseau de neurones**
dont les poids sont trouvés par un **algorithme génétique**. Tout est codé à la main avec numpy.

## Lancer
```
pip install -r requirements.txt
```
Puis ouvrir `main.ipynb` et suivre les étapes dans l'ordre.

## Architecture
```
Voitures IA/
├── main.ipynb            # lance tout, étape par étape, avec des tests
├── requirements.txt
├── Images/
│   ├── Circuit.png             # circuit (affichage)
│   ├── Circuit_mask.png        # blanc = route, noir = dehors (collisions)
│   ├── Circuit Paint.png       # circuit d'origine
│   └── Voiture.png
└── src/
    ├── config.py          # tous les réglages (déjà rempli)
    ├── circuit.py         # Étape 1 : charger le circuit, détecter la route
    ├── car.py             # Étapes 2-3 : physique + capteurs
    ├── neural_network.py  # Étape 4 : réseau de neurones (forward)
    ├── genetic.py         # Étape 5 : sélection, croisement, mutation
    ├── simulation.py      # Étape 6 : boucle d'entraînement
    └── render.py          # Étape 7 : affichage matplotlib / pygame
```

## Comment ça marche
```
capteurs (distances aux bords) + vitesse
            │
            ▼
   réseau de neurones  ──►  [direction, accélération]  ──►  la voiture bouge
            ▲
            │  poids choisis par
   algorithme génétique  ◄──  fitness (distance parcourue, tours…)
```
