"""
boids-maze.py
=============
Pure Boids flock maze solver  –  no pheromone, no ACO.

Every agent obeys three classic Boids rules applied to its local
neighbourhood (BOID_RADIUS):
  Separation  – steer away from close neighbours
  Alignment   – match the average heading of neighbours
  Cohesion    – steer toward the centre-of-mass of neighbours

A single global GOAL ATTRACTION force pulls every agent toward the goal
cell (bottom-right).  Walls are enforced with:
  • Soft repulsion   – gentle push away from closed walls when near them
  • Hard collision   – velocity component perpendicular to a crossed wall
                       is zeroed and position is snapped back

No pheromone, no memory, no discrete hops.  Navigation is entirely
emergent: the flock stretches into corridors, the leading edge discovers
routes and the whole group is dragged along by cohesion.
"""

import pygame
import random
import math
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 800
GRID_SIZE     = 80                  # large cells = wide corridors = minimal walls
COLS, ROWS    = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

NUM_AGENTS = 60
FPS        = 60

# BOIDS
BOID_RADIUS      = 55.0   # pixels – neighbour search radius
SEPARATION_W     = 2.2    # separation force weight
ALIGNMENT_W      = 0.6    # alignment force weight
COHESION_W       = 0.5    # cohesion force weight

# GOAL
GOAL_ATTRACT_W   = 2.5    # how hard every agent is pulled toward the goal
GOAL_FALLOFF     = 200.0  # distance (px) at which attraction is at half-strength

# WALLS
WALL_REPULSE_DIST = GRID_SIZE * 0.25   # start repelling when this close to wall
WALL_REPULSE_STR  = 0.8

# PHYSICS
MAX_SPEED  = 3.0
VEL_DAMP   = 0.90    # velocity damping per frame (smooths jitter)

# SPAWNING – trickle in batches so first agents have room to spread
SPAWN_BATCH    = 5
SPAWN_INTERVAL = 20   # frames between batches

# COLORS
WHITE   = (255, 255, 255)
BLACK   = (  0,   0,   0)
RED     = (200,  50,  50)
BLUE    = ( 50,  50, 200)
MAGENTA = (220,  80, 220)   # agent colour
LIME    = (180, 255, 100)   # agent that just reached goal

# ---------------------------------------------------------------------------
# Wall bitmasks
# ---------------------------------------------------------------------------
W_TOP    = 1
W_RIGHT  = 2
W_BOTTOM = 4
W_LEFT   = 8

OPPOSITE = {W_TOP: W_BOTTOM, W_RIGHT: W_LEFT, W_BOTTOM: W_TOP, W_LEFT: W_RIGHT}
INVERT   = {wb: np.uint8((~wb) & 0xFF) for wb in (W_TOP, W_RIGHT, W_BOTTOM, W_LEFT)}

DIR_INFO = [
    ( 0, -1, W_TOP,    W_BOTTOM),
    ( 1,  0, W_RIGHT,  W_LEFT  ),
    ( 0,  1, W_BOTTOM, W_TOP   ),
    (-1,  0, W_LEFT,   W_RIGHT ),
]

# ---------------------------------------------------------------------------
# Maze (same Prim's generator as the other files)
# ---------------------------------------------------------------------------

class Maze:
    def __init__(self):
        self.walls = np.full((COLS, ROWS),
                             W_TOP | W_RIGHT | W_BOTTOM | W_LEFT, dtype=np.uint8)
        self._generate_prims()
        self.wall_surf = None  # built after pygame.init()

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

    def build_wall_surface(self):
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        for c in range(COLS):
            for r in range(ROWS):
                cx, cy = c * GRID_SIZE, r * GRID_SIZE
                w = self.walls[c, r]
                if w & W_TOP:
                    pygame.draw.line(surf, WHITE, (cx, cy),             (cx+GRID_SIZE, cy),             2)
                if w & W_RIGHT:
                    pygame.draw.line(surf, WHITE, (cx+GRID_SIZE, cy),   (cx+GRID_SIZE, cy+GRID_SIZE),   2)
                if w & W_BOTTOM:
                    pygame.draw.line(surf, WHITE, (cx, cy+GRID_SIZE),   (cx+GRID_SIZE, cy+GRID_SIZE),   2)
                if w & W_LEFT:
                    pygame.draw.line(surf, WHITE, (cx, cy),             (cx, cy+GRID_SIZE),             2)
        self.wall_surf = surf

# ---------------------------------------------------------------------------
# Spatial bucket grid for O(1) neighbour lookup
# ---------------------------------------------------------------------------

class SpatialGrid:
    def __init__(self, bucket_size: float):
        self.bs      = bucket_size
        self.buckets: dict = {}

    def clear(self):
        self.buckets.clear()

    def insert(self, idx: int, x: float, y: float):
        key = (int(x // self.bs), int(y // self.bs))
        if key not in self.buckets:
            self.buckets[key] = []
        self.buckets[key].append(idx)

    def query(self, x: float, y: float):
        bx, by = int(x // self.bs), int(y // self.bs)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (bx + dx, by + dy)
                if k in self.buckets:
                    out.extend(self.buckets[k])
        return out

# ---------------------------------------------------------------------------
# Boids swarm
# ---------------------------------------------------------------------------

GOAL_PX = np.array([(COLS - 1) * GRID_SIZE + GRID_SIZE / 2,
                     (ROWS - 1) * GRID_SIZE + GRID_SIZE / 2], dtype=np.float32)

class BoidsSwarm:
    def __init__(self, n: int):
        self.n            = n
        self.active_count = 0

        m = 4.0
        self.pos = np.random.uniform([m, m], [GRID_SIZE-m, GRID_SIZE-m],
                                     size=(n, 2)).astype(np.float32)
        angles = np.random.uniform(0, 2 * math.pi, n)
        sp     = np.random.uniform(1.0, MAX_SPEED, n)
        self.vel       = np.column_stack([np.cos(angles)*sp,
                                          np.sin(angles)*sp]).astype(np.float32)
        self.acc       = np.zeros((n, 2), dtype=np.float32)
        self.found     = np.zeros(n, dtype=bool)
        self.flash     = np.zeros(n, dtype=int)   # frames of "found" flash

    # ---- spawning ----------------------------------------------------------

    def spawn_batch(self, size: int = SPAWN_BATCH):
        prev = self.active_count
        self.active_count = min(self.active_count + size, self.n)
        m = 4.0
        for i in range(prev, self.active_count):
            self.pos[i]   = np.random.uniform(m, GRID_SIZE - m, 2)
            angle         = random.uniform(0, 2 * math.pi)
            sp            = random.uniform(1.0, MAX_SPEED)
            self.vel[i]   = [math.cos(angle) * sp, math.sin(angle) * sp]
            self.found[i] = False
            self.flash[i] = 0

    # ---- update ------------------------------------------------------------

    def update(self, maze: Maze, sg: SpatialGrid):
        n   = self.active_count
        act = np.where(~self.found[:n])[0]
        if act.size == 0:
            return

        # Rebuild spatial grid
        sg.clear()
        for i in act:
            sg.insert(int(i), float(self.pos[i, 0]), float(self.pos[i, 1]))

        r2 = BOID_RADIUS * BOID_RADIUS

        for i in act:
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
            vx, vy = float(self.vel[i, 0]), float(self.vel[i, 1])

            gc = min(max(int(px // GRID_SIZE), 0), COLS - 1)
            gr = min(max(int(py // GRID_SIZE), 0), ROWS - 1)

            # ---- Boids neighbour scan ----
            sep = np.zeros(2, np.float32)
            ali = np.zeros(2, np.float32)
            coh = np.zeros(2, np.float32)
            cnt = 0

            for j in sg.query(px, py):
                if j == i:
                    continue
                dx = px - float(self.pos[j, 0])
                dy = py - float(self.pos[j, 1])
                d2 = dx*dx + dy*dy
                if d2 < r2 and d2 > 0:
                    d = math.sqrt(d2)
                    sep[0] += dx / d
                    sep[1] += dy / d
                    ali[0] += float(self.vel[j, 0])
                    ali[1] += float(self.vel[j, 1])
                    coh[0] += float(self.pos[j, 0])
                    coh[1] += float(self.pos[j, 1])
                    cnt += 1

            f_sep = np.zeros(2, np.float32)
            f_ali = np.zeros(2, np.float32)
            f_coh = np.zeros(2, np.float32)

            if cnt > 0:
                # Separation
                sn = math.sqrt(sep[0]**2 + sep[1]**2)
                if sn > 0:
                    f_sep = (sep / sn) * MAX_SPEED - self.vel[i]

                # Alignment
                ali /= cnt
                an = math.sqrt(ali[0]**2 + ali[1]**2)
                if an > 0:
                    f_ali = (ali / an) * MAX_SPEED - self.vel[i]

                # Cohesion  – steer toward average position
                coh /= cnt
                desired = coh - self.pos[i]
                dn = math.sqrt(desired[0]**2 + desired[1]**2)
                if dn > 0:
                    f_coh = (desired / dn) * MAX_SPEED - self.vel[i]

            # ---- Goal attraction ----
            to_goal = GOAL_PX - self.pos[i]
            dist_to_goal = math.sqrt(float(to_goal[0])**2 + float(to_goal[1])**2)
            if dist_to_goal > 0:
                # Strength scales with distance (stronger pull when far away)
                strength = GOAL_ATTRACT_W * (dist_to_goal / (dist_to_goal + GOAL_FALLOFF))
                f_goal   = (to_goal / dist_to_goal) * MAX_SPEED * strength
            else:
                f_goal = np.zeros(2, np.float32)

            # ---- Soft wall repulsion ----
            w_bits = int(maze.walls[gc, gr])
            f_wall = np.zeros(2, np.float32)
            cx0, cy0 = gc * GRID_SIZE, gr * GRID_SIZE
            if w_bits & W_LEFT:
                d = px - cx0
                if 0 < d < WALL_REPULSE_DIST:
                    f_wall[0] += WALL_REPULSE_STR * (WALL_REPULSE_DIST - d) / WALL_REPULSE_DIST
            if w_bits & W_RIGHT:
                d = (gc+1) * GRID_SIZE - px
                if 0 < d < WALL_REPULSE_DIST:
                    f_wall[0] -= WALL_REPULSE_STR * (WALL_REPULSE_DIST - d) / WALL_REPULSE_DIST
            if w_bits & W_TOP:
                d = py - cy0
                if 0 < d < WALL_REPULSE_DIST:
                    f_wall[1] += WALL_REPULSE_STR * (WALL_REPULSE_DIST - d) / WALL_REPULSE_DIST
            if w_bits & W_BOTTOM:
                d = (gr+1) * GRID_SIZE - py
                if 0 < d < WALL_REPULSE_DIST:
                    f_wall[1] -= WALL_REPULSE_STR * (WALL_REPULSE_DIST - d) / WALL_REPULSE_DIST

            # Accumulate
            self.acc[i] += f_sep * SEPARATION_W
            self.acc[i] += f_ali * ALIGNMENT_W
            self.acc[i] += f_coh * COHESION_W
            self.acc[i] += f_goal
            self.acc[i] += f_wall

            # Goal check
            if gc == COLS - 1 and gr == ROWS - 1:
                self.found[i] = True
                self.flash[i] = 20

        # ---- Vectorised physics ----
        old_pos = self.pos[act].copy()

        self.vel[act] *= VEL_DAMP
        self.vel[act] += self.acc[act]

        speeds = np.linalg.norm(self.vel[act], axis=1, keepdims=True)
        too_fast = speeds[:, 0] > MAX_SPEED
        if np.any(too_fast):
            fi = act[too_fast]
            self.vel[fi] = (self.vel[fi] / speeds[too_fast]) * MAX_SPEED

        self.pos[act] += self.vel[act]
        self.acc[act]  = 0.0

        np.clip(self.pos[:, 0], 0, WIDTH  - 1, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], 0, HEIGHT - 1, out=self.pos[:, 1])

        # Hard wall collision
        self._wall_resolve(maze, act, old_pos)

    def _wall_resolve(self, maze: Maze, act: np.ndarray, old_pos: np.ndarray):
        for idx, i in enumerate(act):
            ox, oy = float(old_pos[idx, 0]), float(old_pos[idx, 1])
            nx, ny = float(self.pos[i, 0]),  float(self.pos[i, 1])

            ogc = min(max(int(ox // GRID_SIZE), 0), COLS - 1)
            ogr = min(max(int(oy // GRID_SIZE), 0), ROWS - 1)
            ngc = min(max(int(nx // GRID_SIZE), 0), COLS - 1)
            ngr = min(max(int(ny // GRID_SIZE), 0), ROWS - 1)

            if ngc != ogc:
                wb = W_RIGHT if ngc > ogc else W_LEFT
                if maze.walls[ogc, ogr] & wb:
                    self.pos[i, 0] = ox
                    self.vel[i, 0] = 0.0
                    ngc = ogc

            if ngr != ogr:
                wb = W_BOTTOM if ngr > ogr else W_TOP
                if maze.walls[ogc, ogr] & wb:
                    self.pos[i, 1] = oy
                    self.vel[i, 1] = 0.0

    # ---- respawn -----------------------------------------------------------

    def respawn_found(self):
        done = np.where(self.found[:self.active_count])[0]
        m = 4.0
        for i in done:
            if self.flash[i] > 0:
                self.flash[i] -= 1
                if self.flash[i] > 0:
                    continue
            self.pos[i]   = np.random.uniform(m, GRID_SIZE - m, 2)
            angle         = random.uniform(0, 2 * math.pi)
            sp            = random.uniform(1.0, MAX_SPEED)
            self.vel[i]   = [math.cos(angle) * sp, math.sin(angle) * sp]
            self.found[i] = False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Maze – Flock Only")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 22)

    maze  = Maze()
    maze.build_wall_surface()

    sg    = SpatialGrid(BOID_RADIUS)
    swarm = BoidsSwarm(NUM_AGENTS)

    frame       = 0
    total_found = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame += 1

        # Batch spawn
        if frame % SPAWN_INTERVAL == 1 and swarm.active_count < swarm.n:
            swarm.spawn_batch()

        # Agent update
        prev_found = int(np.sum(swarm.found[:swarm.active_count]))
        swarm.update(maze, sg)
        total_found += int(np.sum(swarm.found[:swarm.active_count])) - prev_found

        # Render
        screen.fill(BLACK)
        screen.blit(maze.wall_surf, (0, 0))

        n   = swarm.active_count
        for i in range(n):
            if swarm.found[i] and swarm.flash[i] > 0:
                color = LIME
            elif swarm.found[i]:
                continue
            else:
                color = MAGENTA
            px, py = int(swarm.pos[i, 0]), int(swarm.pos[i, 1])
            pygame.draw.circle(screen, color, (px, py), 4)

        # Draw velocity arrows for a subset (shows flock heading)
        for i in range(0, min(n, swarm.active_count), 3):
            if swarm.found[i]:
                continue
            px, py = float(swarm.pos[i, 0]), float(swarm.pos[i, 1])
            vx, vy = float(swarm.vel[i, 0]), float(swarm.vel[i, 1])
            spd = math.sqrt(vx*vx + vy*vy)
            if spd > 0.1:
                ex = px + vx / spd * 8
                ey = py + vy / spd * 8
                pygame.draw.line(screen, (150, 80, 200),
                                 (int(px), int(py)), (int(ex), int(ey)), 1)

        # Start / Goal
        pygame.draw.rect(screen, BLUE, (0, 0, GRID_SIZE, GRID_SIZE), 3)
        pygame.draw.rect(screen, RED,  (WIDTH - GRID_SIZE, HEIGHT - GRID_SIZE,
                                         GRID_SIZE, GRID_SIZE))

        # HUD
        fps_t  = font.render(f"FPS:    {clock.get_fps():.0f}", True, WHITE)
        sol_t  = font.render(f"Found:  {total_found}", True, WHITE)
        ag_t   = font.render(f"Agents: {swarm.active_count}/{swarm.n}", True, WHITE)
        mode_t = font.render("Mode: Boids only", True, (220, 130, 255))
        screen.blit(fps_t,  (8,  8))
        screen.blit(sol_t,  (8, 30))
        screen.blit(ag_t,   (8, 52))
        screen.blit(mode_t, (8, 74))

        pygame.display.flip()

        swarm.respawn_found()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
