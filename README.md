# Swarm Intelligence Exploration

A collection of interactive Python simulations exploring emergent behaviour through two classic swarm-intelligence paradigms: Ant Colony Optimisation (ACO) and Reynolds Boids. Built with Pygame and NumPy.

---

## Project Structure

| File | Description |
|------|-------------|
| `boids-sim.py` | Classic open-space Boids flock with predator, rotating goals, and real-time slider controls |
| `boids-maze.py` | Pure Boids flock navigating a procedurally generated maze -- no pheromone, navigation entirely emergent |
| `aco-maze.py` | Pure Ant Colony Optimisation maze solver -- pheromone trails reinforce the shortest route over time |
| `swarm-maze-fast.py` | Hybrid ACO + Boids in a dynamic maze (walls toggle at runtime); optimised with NumPy vectorisation and a spatial bucket grid |

---

## Concepts Demonstrated

- Reynolds Boids -- separation, alignment, and cohesion rules producing lifelike flocking behaviour
- Ant Colony Optimisation -- stigmergic pheromone deposit and evaporation guiding ants to shortest paths
- Hybrid swarms -- combining pheromone memory (ACO) with local social forces (Boids) in a single agent
- Spatial hashing -- O(1) neighbour lookup via a bucket grid, enabling large swarms at 60 FPS
- Emergent navigation -- agents solve mazes through collective behaviour, not explicit pathfinding

---

## Quick Start

### Prerequisites

- Python 3.9 or later
- pip

### 1 - Set up the environment

Windows:
```
setup.bat
```

macOS / Linux:
```
chmod +x setup.sh && ./setup.sh
```

This creates a virtual environment in `.venv/`, installs all dependencies, and prints next steps.

### 2 - Activate the environment

Windows:
```
.venv\Scripts\activate
```

macOS / Linux:
```
source .venv/bin/activate
```

### 3 - Run a simulation

```
# Open-space Boids with sliders
python boids-sim.py

# Boids navigating a maze
python boids-maze.py

# ACO maze solver
python aco-maze.py

# Hybrid ACO + Boids in a dynamic maze (fastest, most complex)
python swarm-maze-fast.py
```

---

## Controls

| Simulation | Controls |
|-----------|----------|
| `boids-sim.py` | Drag the Separation / Alignment / Cohesion slider knobs at the bottom to tune the flock in real time. Press ESC to quit. |
| All others | Close the window or press ESC to exit. |

---

## Tuning Parameters

Every script exposes a clearly labeled configuration block near the top. Key parameters:

| Parameter | Effect |
|-----------|--------|
| `NUM_BOIDS` / `NUM_AGENTS` / `NUM_ANTS` | Population size |
| `MAX_SPEED` | Maximum agent velocity (px/frame) |
| `BOID_RADIUS` | Neighbour perception radius |
| `SEPARATION_W` / `ALIGNMENT_W` / `COHESION_W` | Boids rule weights |
| `ALPHA` / `BETA` | ACO pheromone vs heuristic weight |
| `EVAPORATION` | Pheromone decay rate per frame |
| `WALL_TOGGLE_RATE` | Maze mutation speed (swarm-maze-fast.py only) |

---

## Dependencies

See `requirements.txt`.

| Package | Purpose |
|---------|---------|
| `pygame` | Window, rendering, event loop |
| `numpy` | Vectorised agent physics and pheromone arrays |

---

## License

This project is released under the Swarm Intelligence Exploration Source License -- see `LICENSE` for full terms.

Summary: Free to use, modify, and distribute for any purpose. If you use this code in academic work (papers, theses, course projects submitted for credit, or published research), you must:

1. Credit this repository in your acknowledgements or references.
2. Fork from this repository as the starting point for your work.

---

## Acknowledgements

- Craig Reynolds -- original Boids model (1987)
- Marco Dorigo -- Ant Colony Optimisation (1992)

