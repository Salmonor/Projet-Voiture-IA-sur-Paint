"""Affichage : matplotlib dans le notebook, pygame pour l'animation.

Ce fichier est un outil (pas du ML) : il est déjà rempli pour que tu puisses
VOIR ce que tu codes à chaque étape.

  show_circuit(circuit, car, show_sensors=True)  -> image fixe (étapes 1-3)
  plot_trajectory(circuit, positions)            -> chemin suivi par une voiture
  plot_history(history)                          -> courbe d'apprentissage (étape 6)
  replay(circuit)                                -> pilote toi-même au clavier (teste l'étape 2 !)
  replay(circuit, network)                       -> regarde un réseau conduire
  replay_population(circuit, networks)           -> regarde toute une génération

Il suppose seulement que les voitures ont x, y, angle, speed, alive, distance,
laps, steps, et les méthodes update / read_sensors / get_inputs (src/car.py).
"""
import matplotlib.pyplot as plt
import numpy as np

from src import config

ALIVE_COLOR = "#2ecc71"
DEAD_COLOR = "#e74c3c"
SENSOR_COLOR = "#f1c40f"


# ====================================================================== Outils communs
def _car_corners(x, y, angle_deg, length=config.CAR_LENGTH, width=config.CAR_WIDTH):
    """Les 4 coins du rectangle de la voiture, tourné selon son angle. Forme (4, 2)."""
    a = np.radians(angle_deg)
    forward = np.array([np.cos(a), np.sin(a)]) * length / 2
    side = np.array([-np.sin(a), np.cos(a)]) * width / 2
    center = np.array([x, y])
    return np.array([center + forward + side, center + forward - side,
                     center - forward - side, center - forward + side])


def _sensor_rays(car, circuit):
    """Liste de (x_fin, y_fin) pour chaque capteur, ou None si pas encore codé."""
    try:
        distances = car.read_sensors(circuit)
    except NotImplementedError:
        return None
    ends = []
    for rel_angle, dist in zip(config.SENSOR_ANGLES, distances):
        a = np.radians(car.angle + rel_angle)
        ends.append((car.x + np.cos(a) * dist, car.y + np.sin(a) * dist))
    return ends


def _as_list(cars):
    if cars is None:
        return []
    return list(cars) if isinstance(cars, (list, tuple)) else [cars]


# ====================================================================== Matplotlib
def show_circuit(circuit, cars=None, show_sensors: bool = False, ax=None,
                 show_mask: bool = False):
    """Affiche le circuit et, optionnellement, des voitures et leurs rayons capteurs.

    Args:
        cars         : une voiture ou une liste de voitures (ou None)
        show_sensors : dessine les rayons des capteurs (nécessite read_sensors)
        show_mask    : superpose le masque (route en bleu) pour vérifier l'étape 1
    Returns:
        ax (pour pouvoir dessiner d'autres choses par-dessus)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(circuit.image)
    if show_mask:
        overlay = np.zeros((*circuit.mask.shape, 4))
        overlay[circuit.mask] = (0.2, 0.5, 1.0, 0.35)
        ax.imshow(overlay)

    # Ligne d'arrivée
    y_min, y_max = config.FINISH_LINE_Y_RANGE
    ax.plot([config.FINISH_LINE_X] * 2, [y_min, y_max], "w--", lw=1)

    sensors_missing = False
    for car in _as_list(cars):
        color = ALIVE_COLOR if car.alive else DEAD_COLOR
        if show_sensors:
            ends = _sensor_rays(car, circuit)
            if ends is None:
                sensors_missing = True
            else:
                for ex, ey in ends:
                    ax.plot([car.x, ex], [car.y, ey], "-", color=SENSOR_COLOR, lw=1)
                    ax.plot(ex, ey, "o", color=SENSOR_COLOR, ms=4)
        ax.add_patch(plt.Polygon(_car_corners(car.x, car.y, car.angle),
                                 closed=True, fc=color, ec="black", lw=1))
        a = np.radians(car.angle)
        ax.plot([car.x, car.x + np.cos(a) * config.CAR_LENGTH],
                [car.y, car.y + np.sin(a) * config.CAR_LENGTH], "k-", lw=2)

    if sensors_missing:
        print("ℹ️ read_sensors n'est pas encore codé : rayons non affichés.")
    ax.set_xlim(0, circuit.width)
    ax.set_ylim(circuit.height, 0)
    ax.axis("off")
    return ax


def plot_trajectory(circuit, positions: list, ax=None, label=None):
    """Trace la trajectoire [(x, y), ...] d'une voiture sur le circuit.

    La couleur va du violet (début) au jaune (fin) pour voir le sens de parcours.
    Appelle-la plusieurs fois avec le même `ax` pour comparer des voitures.
    """
    if ax is None:
        ax = show_circuit(circuit)
    pts = np.asarray(positions, dtype=float)
    if len(pts) == 0:
        return ax
    ax.plot(pts[:, 0], pts[:, 1], "-", color="white", lw=1, alpha=0.5, label=label)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap="plasma", s=4)
    ax.plot(*pts[0], "o", color=ALIVE_COLOR, ms=8, mec="black")
    ax.plot(*pts[-1], "X", color=DEAD_COLOR, ms=10, mec="black")
    plt.colorbar(sc, ax=ax, fraction=0.025, label="pas de simulation")
    if label:
        ax.legend()
    return ax


def plot_history(history: dict, ax=None):
    """Courbe d'apprentissage : fitness max et moyenne par génération."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    gens = np.arange(1, len(history["best"]) + 1)
    ax.plot(gens, history["best"], "-o", ms=3, label="meilleure", color="#2980b9")
    ax.plot(gens, history["mean"], "-", label="moyenne", color="#95a5a6")
    ax.fill_between(gens, history["mean"], history["best"], color="#2980b9", alpha=0.1)
    ax.set_xlabel("génération")
    ax.set_ylabel("fitness")
    ax.set_title("Courbe d'apprentissage")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()
    return ax


# ====================================================================== Pygame
def replay(circuit, network=None, max_steps: int = 3000):
    """Ouvre une fenêtre pygame avec UNE voiture.

    - network=None : tu conduis au clavier (flèches) -> parfait pour tester l'étape 2.
    - sinon        : la voiture est pilotée par network.forward(car.get_inputs(circuit)).

    Touches : flèches = conduire, S = capteurs, ESPACE = pause, R = recommencer,
              ÉCHAP = quitter.
    Returns:
        la liste des positions [(x, y), ...] -> à passer à plot_trajectory.
    """
    return _pygame_loop(circuit, [network], max_steps)[0]


def replay_population(circuit, networks: list, max_steps: int = config.MAX_STEPS_PER_GENERATION):
    """Comme replay, mais toutes les voitures d'une génération roulent en même temps.
    Le meilleur réseau (le 1er de la liste après next_generation) est en surbrillance.

    Returns:
        la liste des trajectoires (une par réseau).
    """
    return _pygame_loop(circuit, list(networks), max_steps)


def _pygame_loop(circuit, networks, max_steps):
    import pygame  # importé ici : le notebook marche même si pygame bugue
    from src.car import Car

    pygame.init()
    try:
        screen = pygame.display.set_mode((circuit.width, circuit.height))
        pygame.display.set_caption("Voitures IA")
        background = pygame.surfarray.make_surface(circuit.image.swapaxes(0, 1))
        font = pygame.font.SysFont("consolas", 18)
        big_font = pygame.font.SysFont("consolas", 32, bold=True)
        clock = pygame.time.Clock()
        keyboard = networks == [None]
        show_sensors = len(networks) == 1

        def reset():
            cars = [Car() for _ in networks]
            return cars, [[(c.x, c.y)] for c in cars], 0

        cars, trails, step = reset()
        paused, running = False, True

        while running:
            # ---------------------------------------------------- événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_s:
                        show_sensors = not show_sensors
                    elif event.key == pygame.K_r:
                        cars, trails, step = reset()

            any_alive = any(c.alive for c in cars)
            finished = step >= max_steps or not any_alive

            # ---------------------------------------------------- simulation
            if not paused and not finished:
                keys = pygame.key.get_pressed()
                for car, net, trail in zip(cars, networks, trails):
                    if not car.alive:
                        continue
                    if keyboard:
                        steering = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
                        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
                    else:
                        steering, throttle = net.forward(car.get_inputs(circuit))
                    car.update(float(steering), float(throttle), circuit)
                    trail.append((car.x, car.y))
                step += 1

            # ---------------------------------------------------- dessin
            screen.blit(background, (0, 0))
            # On dessine le meilleur (index 0) en dernier pour qu'il soit au-dessus
            order = list(range(len(cars)))[::-1]
            for i in order:
                car, trail = cars[i], trails[i]
                highlight = i == 0
                if len(trail) > 1 and (highlight or len(cars) <= 10):
                    pygame.draw.lines(screen, (255, 255, 255), False, trail, 2 if highlight else 1)
                if show_sensors and car.alive and (highlight or len(cars) == 1):
                    ends = _sensor_rays(car, circuit)
                    for end in ends or []:
                        pygame.draw.line(screen, (241, 196, 15), (car.x, car.y), end, 1)
                        pygame.draw.circle(screen, (241, 196, 15), (int(end[0]), int(end[1])), 3)
                color = (46, 204, 113) if car.alive else (231, 76, 60)
                if highlight and len(cars) > 1:
                    color = (52, 152, 219) if car.alive else (155, 89, 182)
                corners = _car_corners(car.x, car.y, car.angle)
                pygame.draw.polygon(screen, color, corners)
                pygame.draw.polygon(screen, (0, 0, 0), corners, 1)
                # pare-brise : petit trait à l'avant pour voir la direction
                pygame.draw.line(screen, (0, 0, 0), corners[0], corners[1], 3)

            # ---------------------------------------------------- texte (HUD)
            if len(cars) == 1:
                c = cars[0]
                lines = [
                    "Mode : " + ("clavier (flèches)" if keyboard else "réseau"),
                    f"Pas : {step}/{max_steps}",
                    f"Vitesse : {c.speed:5.2f}",
                    f"Distance : {c.distance:7.1f}",
                    f"Tours : {c.laps}",
                ]
            else:
                lines = [
                    f"Pas : {step}/{max_steps}",
                    f"En vie : {sum(c.alive for c in cars)}/{len(cars)}",
                    f"Meilleure distance : {max(c.distance for c in cars):7.1f}",
                    f"Tours max : {max(c.laps for c in cars)}",
                ]
            lines.append("S capteurs  ESPACE pause  R reset  ÉCHAP quitter")
            panel = pygame.Surface((440, 24 * len(lines) + 10), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 150))
            screen.blit(panel, (10, circuit.height - panel.get_height() - 10))
            for k, text in enumerate(lines):
                surf = font.render(text, True, (255, 255, 255))
                screen.blit(surf, (20, circuit.height - panel.get_height() + 24 * k))

            message = None
            if paused:
                message = "PAUSE"
            elif not any_alive:
                message = "CRASH !  (R pour recommencer)"
            elif step >= max_steps:
                message = "Temps écoulé  (R pour recommencer)"
            if message:
                surf = big_font.render(message, True, (255, 255, 255))
                rect = surf.get_rect(center=(circuit.width // 2, circuit.height // 2))
                pygame.draw.rect(screen, (0, 0, 0), rect.inflate(30, 20))
                screen.blit(surf, rect)

            pygame.display.flip()
            clock.tick(config.FPS)
    finally:
        pygame.display.quit()
        pygame.quit()  # indispensable, sinon le kernel du notebook peut rester bloqué

    return trails
