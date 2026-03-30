"""
aco-maze.py
===========
Pure Ant Colony Optimisation maze solver  –  no Boids forces.

Each ant:
  1. Starts at cell (0, 0).
  2. At every cell, picks the next reachable cell with a roulette-wheel
     probability proportional to  tau^alpha * eta^beta  where
       tau  = pheromone on the candidate cell
       eta  = 1 / (distance_to_goal + 1)   (distance heuristic)
  3. Upon reaching goal deposits pheromone backwards along its whole path.
  4. Pheromone evaporates every frame.

Over time the colony reinforces the shortest route, which glows bright
green on screen.  Path quality improves visibly over the first ~30 s.
"""

import pygame
import random
import math
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 800
GRID_SIZE     = 40                  # pixels per maze cell
COLS, ROWS    = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

NUM_ANTS  = 80
FPS       = 60

# ACO
ALPHA              = 2.0    # pheromone weight
BETA               = 3.0    # heuristic weight (higher = greedier toward goal)
EVAPORATION        = 0.97   # pheromone decay per frame
DEPOSIT_AMOUNT     = 15.0   # pheromone added per cell of a successful path
PHEROMONE_MIN      = 0.1
PHEROMONE_BOOST    = 1.0    # bonus deposit on the goal cell itself

# Visual: how fast ants glide between cell centres (pixels / frame)
GLIDE_SPEED = 5.0

# COLORS
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
RED    = (200,  50,  50)
BLUE   = ( 50,  50, 200)
ORANGE = (255, 165,   0)    # ant colour
YELLOW = (255, 220,   0)    # ant that just found goal

# ---------------------------------------------------------------------------
# Wall bitmask constants (shared by Maze, Ant, and BFS)
# ---------------------------------------------------------------------------
W_TOP    = 1
W_RIGHT  = 2
W_BOTTOM = 4
W_LEFT   = 8

OPPOSITE = {W_TOP: W_BOTTOM, W_RIGHT: W_LEFT, W_BOTTOM: W_TOP, W_LEFT: W_RIGHT}
INVERT   = {wb: np.uint8((~wb) & 0xFF) for wb in (W_TOP, W_RIGHT, W_BOTTOM, W_LEFT)}

DIR_INFO = [        # (dc, dr, wall_bit_of_source, wall_bit_of_dest)
    ( 0, -1, W_TOP,    W_BOTTOM),
    ( 1,  0, W_RIGHT,  W_LEFT  ),
    ( 0,  1, W_BOTTOM, W_TOP   ),
    (-1,  0, W_LEFT,   W_RIGHT ),
]

# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

class Maze:
    def __init__(self):
        self.walls     = np.full((COLS, ROWS),
                                 W_TOP | W_RIGHT | W_BOTTOM | W_LEFT, dtype=np.uint8)
        self.pheromone = np.full((COLS, ROWS), PHEROMONE_MIN, dtype=np.float32)
        self._generate_prims()
        self._open_nbrs = [[[] for _ in range(ROWS)] for _ in range(COLS)]
        self._rebuild_nbr_cache()
        self.wall_surf  = None   # built after pygame.init()

    # ---- generation --------------------------------------------------------

    def _generate_prims(self):
        visited = np.zeros((COLS, ROWS), dtype=bool)
        visited[0, 0] = True
        wlist = []

        def add(c, r):
            for dc, dr, wb, _ in DIR_INFO:
                nc, nr = c + dc, r + dr
                if 0 <= nc < COLS and 0 <= nr < ROWS and not visited[nc, nr]:
                    wlist.append((c, r, nc, nr, wb))

        add(0, 0)
        while wlist:
            i = random.randrange(len(wlist))
            wlist[i], wlist[-1] = wlist[-1], wlist[i]
            c, r, nc, nr, wb = wlist.pop()
            if visited[nc, nr]:
                continue
            self.walls[c,  r ] &= INVERT[wb]
            self.walls[nc, nr] &= INVERT[OPPOSITE[wb]]
            visited[nc, nr] = True
            add(nc, nr)

    # ---- caches ------------------------------------------------------------

    def _rebuild_nbr_cache(self):
        for c in range(COLS):
            for r in range(ROWS):
                nbrs = []
                for dc, dr, wb, _ in DIR_INFO:
                    nc, nr = c + dc, r + dr
                    if 0 <= nc < COLS and 0 <= nr < ROWS:
                        if not (self.walls[c, r] & wb):
                            nbrs.append((nc, nr))
                self._open_nbrs[c][r] = nbrs

    def build_wall_surface(self):
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        for c in range(COLS):
            for r in range(ROWS):
                cx, cy = c * GRID_SIZE, r * GRID_SIZE
                w = self.walls[c, r]
                if w & W_TOP:
                    pygame.draw.line(surf, WHITE, (cx, cy),            (cx + GRID_SIZE, cy),            2)
                if w & W_RIGHT:
                    pygame.draw.line(surf, WHITE, (cx + GRID_SIZE, cy),(cx + GRID_SIZE, cy + GRID_SIZE), 2)
                if w & W_BOTTOM:
                    pygame.draw.line(surf, WHITE, (cx, cy + GRID_SIZE),(cx + GRID_SIZE, cy + GRID_SIZE), 2)
                if w & W_LEFT:
                    pygame.draw.line(surf, WHITE, (cx, cy),            (cx, cy + GRID_SIZE),             2)
        self.wall_surf = surf

    # ---- pheromone ---------------------------------------------------------

    def evaporate(self):
        np.multiply(self.pheromone, EVAPORATION, out=self.pheromone)
        np.maximum(self.pheromone, PHEROMONE_MIN, out=self.pheromone)

    def deposit(self, path):
        """Deposit pheromone on every cell in path (list of (c,r))."""
        n = len(path)
        if n == 0:
            return
        # Shorter paths get a bigger per-cell reward
        reward = DEPOSIT_AMOUNT * (COLS + ROWS) / max(n, 1)
        for (c, r) in path:
            self.pheromone[c, r] += reward
        # Extra bonus on goal itself
        gc, gr = COLS - 1, ROWS - 1
        self.pheromone[gc, gr] += PHEROMONE_BOOST

# ---------------------------------------------------------------------------
# Ant  –  pure ACO agent
# ---------------------------------------------------------------------------

GOAL_C, GOAL_R = COLS - 1, ROWS - 1
GOAL_PX = np.array([GOAL_C * GRID_SIZE + GRID_SIZE / 2,
                     GOAL_R * GRID_SIZE + GRID_SIZE / 2], dtype=np.float32)

class Ant:
    def __init__(self):
        self._reset()
        self.flash = 0          # frames to show "found" colour

    def _reset(self):
        self.cell  = (0, 0)     # current logical cell
        self.pos   = np.array([GRID_SIZE / 2, GRID_SIZE / 2], dtype=np.float32)
        self.target_pos = self.pos.copy()
        self.path  = [(0, 0)]
        self.visited = set()
        self.visited.add((0, 0))
        self.moving = False

    def update(self, maze: Maze):
        c, r = self.cell

        # --- Glide toward target cell centre ---
        delta = self.target_pos - self.pos
        dist  = float(np.linalg.norm(delta))

        if dist > GLIDE_SPEED:
            self.pos += (delta / dist) * GLIDE_SPEED
            return  # still travelling

        # Arrived at target cell
        self.pos = self.target_pos.copy()

        # --- Check goal ---
        if c == GOAL_C and r == GOAL_R:
            maze.deposit(self.path)
            self.flash = 12
            self._reset()
            return

        # --- ACO cell selection ---
        nbrs = maze._open_nbrs[c][r]
        if not nbrs:
            self._reset()
            return

        # Filter unvisited neighbours (optional: allow revisit if no unvisited)
        unvisited = [(nc, nr) for nc, nr in nbrs if (nc, nr) not in self.visited]
        candidates = unvisited if unvisited else nbrs

        # Compute selection probabilities
        probs = []
        for nc, nr in candidates:
            tau = max(maze.pheromone[nc, nr], 1e-9) ** ALPHA
            # Euclidean heuristic inverse
            cx  = nc * GRID_SIZE + GRID_SIZE / 2
            cy  = nr * GRID_SIZE + GRID_SIZE / 2
            ddx = GOAL_PX[0] - cx
            ddy = GOAL_PX[1] - cy
            eta = (1.0 / (math.sqrt(ddx*ddx + ddy*ddy) + 1.0)) ** BETA
            probs.append(tau * eta)

        denom = sum(probs)
        if denom == 0:
            next_cell = random.choice(candidates)
        else:
            probs = [p / denom for p in probs]
            rnd, cum = random.random(), 0.0
            next_cell = candidates[-1]
            for j, (nc, nr) in enumerate(candidates):
                cum += probs[j]
                if rnd <= cum:
                    next_cell = (nc, nr)
                    break

        # Move to chosen cell
        self.cell = next_cell
        self.path.append(next_cell)
        self.visited.add(next_cell)
        nc, nr = next_cell
        self.target_pos = np.array([nc * GRID_SIZE + GRID_SIZE / 2,
                                     nr * GRID_SIZE + GRID_SIZE / 2], dtype=np.float32)

        if self.flash > 0:
            self.flash -= 1

# ---------------------------------------------------------------------------
# Pheromone overlay surface
# ---------------------------------------------------------------------------

class PheroSurf:
    def __init__(self):
        self.surf = pygame.Surface((WIDTH, HEIGHT))

    def update(self, phero: np.ndarray):
        green    = np.clip(phero * 25.0, 0, 255).astype(np.uint8)
        expanded = np.repeat(np.repeat(green, GRID_SIZE, axis=0), GRID_SIZE, axis=1)
        gw, gh   = expanded.shape[0], expanded.shape[1]
        px       = pygame.surfarray.pixels3d(self.surf)
        px[:, :, :] = 0
        px[:gw, :gh, 1] = expanded
        del px

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ACO Maze Solver – Ant Colony Only")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 22)

    maze = Maze()
    maze.build_wall_surface()

    ants        = [Ant() for _ in range(NUM_ANTS)]
    phero_surf  = PheroSurf()
    frame       = 0
    total_found = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame += 1

        # Evaporate
        maze.evaporate()

        # Count finds this frame (before update)
        prev = sum(1 for a in ants if a.flash > 0)

        # Update ants
        for ant in ants:
            was_flashing = ant.flash > 0
            ant.update(maze)
            if was_flashing and ant.flash == 0:
                total_found += 1

        # Render
        screen.fill(BLACK)
        phero_surf.update(maze.pheromone)
        screen.blit(phero_surf.surf, (0, 0))
        screen.blit(maze.wall_surf,  (0, 0))

        for ant in ants:
            color = YELLOW if ant.flash > 0 else ORANGE
            px, py = int(ant.pos[0]), int(ant.pos[1])
            pygame.draw.circle(screen, color, (px, py), 4)

        # Start / Goal markers
        pygame.draw.rect(screen, BLUE, (0, 0, GRID_SIZE, GRID_SIZE), 3)
        pygame.draw.rect(screen, RED,  (WIDTH - GRID_SIZE, HEIGHT - GRID_SIZE,
                                         GRID_SIZE, GRID_SIZE))

        # HUD
        fps_t  = font.render(f"FPS:    {clock.get_fps():.0f}", True, WHITE)
        sol_t  = font.render(f"Found:  {total_found}", True, WHITE)
        ants_t = font.render(f"Ants:   {NUM_ANTS}", True, WHITE)
        algo_t = font.render("Mode: ACO only", True, (100, 255, 100))
        screen.blit(fps_t,  (8,  8))
        screen.blit(sol_t,  (8, 30))
        screen.blit(ants_t, (8, 52))
        screen.blit(algo_t, (8, 74))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
