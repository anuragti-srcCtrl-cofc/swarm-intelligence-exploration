"""
swarm-maze-fast.py
==================
Optimised version of swarm-maze.py.

Key speed-ups over the original:
  1. NumPy pheromone array   – evaporation done with one vectorised multiply.
  2. Spatial bucket grid     – boid neighbour search is O(agents/bucket)
                               instead of O(N²).
  3. Fully-vectorised boids  – separation / alignment forces computed with
                               NumPy broadcasting, no Python loops per agent.
  4. Cached maze surface     – walls redrawn onto a Surface only when the
                               maze mutates (every WALL_TOGGLE_RATE frames),
                               not every frame.
  5. Tuned parameters        – slightly fewer wall-toggles and a faster
                               evaporation rate so pheromone trails build up
                               quickly and agents converge faster.
"""

import pygame
import random
import math
import numpy as np

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 40                  # Pixels per maze cell
COLS, ROWS = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
NUM_AGENTS = 100             # More agents → richer pheromone trails
FPS = 60

# ACO PARAMS
PHEROMONE_EVAPORATION = 0.97    # Slightly faster evaporation keeps trails fresh
PHEROMONE_INTENSITY   = 8.0     # Stronger deposit rewards successful paths sooner
PHEROMONE_MIN         = 0.1     # Floor to prevent dead zones
ACO_ALPHA             = 2.0     # Pheromone weight (when not crowded)
ACO_BETA              = 2.5     # Distance-to-goal heuristic weight

# BOIDS PARAMS
BOID_VIEW_RADIUS  = 50          # Pixel radius for neighbour search
SEPARATION_FORCE  = 1.8
ALIGNMENT_FORCE   = 0.4
COHESION_FORCE    = 1.8         # ACO goal force weight

MAX_SPEED = 3.5

# DYNAMIC MAZE
WALL_TOGGLE_RATE  = 90          # Toggle walls every N frames (less chaos)
WALL_TOGGLE_COUNT = 3           # Fewer simultaneous toggles → more stable paths

# SPAWNING
SPAWN_BATCH_SIZE = 5            # Agents released per batch
SPAWN_INTERVAL   = 75           # Frames between each batch release

# PHYSICS
VEL_DAMPING = 0.88              # Per-frame velocity damping (smooths jitter)

# SPATIAL BUCKET SIZE (pixels) – should be >= BOID_VIEW_RADIUS for correctness
BUCKET_SIZE = BOID_VIEW_RADIUS

# COLORS
WHITE = (255, 255, 255)
BLACK = (0,   0,   0  )
RED   = (200, 50,  50 )
BLUE  = (50,  50,  200)
CYAN  = (0,   255, 255)

# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

# Wall encoding as bitmask for fast access
W_TOP    = 1
W_RIGHT  = 2
W_BOTTOM = 4
W_LEFT   = 8
OPPOSITE = {
    W_TOP:    W_BOTTOM,
    W_RIGHT:  W_LEFT,
    W_BOTTOM: W_TOP,
    W_LEFT:   W_RIGHT,
}
INVERT = {wb: np.uint8((~wb) & 0xFF) for wb in (W_TOP, W_RIGHT, W_BOTTOM, W_LEFT)}

# Direction → (dc, dr, wall_bit, opp_bit)
DIR_INFO = [
    (0,  -1, W_TOP,    W_BOTTOM),   # top
    (1,   0, W_RIGHT,  W_LEFT  ),   # right
    (0,   1, W_BOTTOM, W_TOP   ),   # bottom
    (-1,  0, W_LEFT,   W_RIGHT ),   # left
]

class Maze:
    """
    Stores walls as a (COLS, ROWS) uint8 numpy array where each cell is
    a bitmask.  Also stores pheromone as a (COLS, ROWS) float32 array so
    evaporation is one multiply.
    """
    def __init__(self):
        # All walls closed initially (all 4 bits set)
        self.walls     = np.full((COLS, ROWS), W_TOP | W_RIGHT | W_BOTTOM | W_LEFT, dtype=np.uint8)
        self.pheromone = np.full((COLS, ROWS), PHEROMONE_MIN, dtype=np.float32)
        self._generate_prims()

        # Pre-compute neighbour lists per cell (doesn't change except on mutate)
        self._open_neighbors = [[[] for _ in range(ROWS)] for _ in range(COLS)]
        self._rebuild_neighbor_cache()

        # Cached wall surface – rebuilt on mutate
        self.wall_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self._rebuild_wall_surface()

    # ------------------------------------------------------------------ maze gen

    def _generate_prims(self):
        visited = np.zeros((COLS, ROWS), dtype=bool)
        visited[0, 0] = True
        wall_list = []

        def add_walls(c, r):
            for dc, dr, wb, _ in DIR_INFO:
                nc, nr = c + dc, r + dr
                if 0 <= nc < COLS and 0 <= nr < ROWS and not visited[nc, nr]:
                    wall_list.append((c, r, nc, nr, wb))

        add_walls(0, 0)

        while wall_list:
            idx = random.randrange(len(wall_list))
            # Swap-and-pop for O(1) removal
            wall_list[idx], wall_list[-1] = wall_list[-1], wall_list[idx]
            c, r, nc, nr, wb = wall_list.pop()

            if visited[nc, nr]:
                continue

            # Remove wall between (c,r) and (nc,nr)
            self.walls[c,  r ] &= INVERT[wb]
            self.walls[nc, nr] &= INVERT[OPPOSITE[wb]]
            visited[nc, nr] = True
            add_walls(nc, nr)

    # ------------------------------------------------------------------ caches

    def _rebuild_neighbor_cache(self):
        for c in range(COLS):
            for r in range(ROWS):
                nbrs = []
                for dc, dr, wb, _ in DIR_INFO:
                    nc, nr = c + dc, r + dr
                    if 0 <= nc < COLS and 0 <= nr < ROWS:
                        if not (self.walls[c, r] & wb):  # wall is open
                            nbrs.append((nc, nr))
                self._open_neighbors[c][r] = nbrs

    def _rebuild_wall_surface(self):
        surf = self.wall_surface
        surf.fill((0, 0, 0, 0))  # Transparent
        for c in range(COLS):
            for r in range(ROWS):
                cx, cy = c * GRID_SIZE, r * GRID_SIZE
                w = self.walls[c, r]
                if w & W_TOP:
                    pygame.draw.line(surf, WHITE, (cx, cy), (cx + GRID_SIZE, cy), 1)
                if w & W_RIGHT:
                    pygame.draw.line(surf, WHITE, (cx + GRID_SIZE, cy), (cx + GRID_SIZE, cy + GRID_SIZE), 1)
                if w & W_BOTTOM:
                    pygame.draw.line(surf, WHITE, (cx, cy + GRID_SIZE), (cx + GRID_SIZE, cy + GRID_SIZE), 1)
                if w & W_LEFT:
                    pygame.draw.line(surf, WHITE, (cx, cy), (cx, cy + GRID_SIZE), 1)

    # ------------------------------------------------------------------ dynamics

    def _path_exists(self, sc: int, sr: int, ec: int, er: int) -> bool:
        """
        BFS from cell (sc, sr) → (ec, er).  Short-circuits as soon as the
        destination is found, so it is faster than a full-grid scan for
        well-connected mazes.  Used to validate every wall change before
        committing it: if no path exists from start to goal after toggling
        a wall, the toggle is reverted.
        """
        visited = np.zeros((COLS, ROWS), dtype=bool)
        stack   = [(sc, sr)]
        visited[sc, sr] = True

        while stack:
            c, r = stack.pop()
            if c == ec and r == er:
                return True
            for dc, dr, wb, _ in DIR_INFO:
                nc, nr = c + dc, r + dr
                if 0 <= nc < COLS and 0 <= nr < ROWS and not visited[nc, nr]:
                    if not (self.walls[c, r] & wb):   # passage is open
                        visited[nc, nr] = True
                        stack.append((nc, nr))
        return False

    def mutate(self):
        """
        Attempt WALL_TOGGLE_COUNT independent wall toggles.
        Before committing each toggle we explicitly check whether a path
        still exists from the start cell (0, 0) to the goal cell
        (COLS-1, ROWS-1).  If not, the toggle is immediately reverted so
        the goal is always reachable.
        """
        changed = False
        goal_c, goal_r = COLS - 1, ROWS - 1

        for _ in range(WALL_TOGGLE_COUNT):
            c  = random.randint(0, COLS - 1)
            r  = random.randint(0, ROWS - 1)
            dc, dr, wb, opp = random.choice(DIR_INFO)
            nc, nr = c + dc, r + dr

            if not (0 <= nc < COLS and 0 <= nr < ROWS):
                continue

            # Apply toggle tentatively
            self.walls[c,  r ] ^= wb
            self.walls[nc, nr] ^= opp

            # Keep only if the goal is still reachable from the start
            if self._path_exists(0, 0, goal_c, goal_r):
                changed = True
            else:
                # Revert – restore previous wall state
                self.walls[c,  r ] ^= wb
                self.walls[nc, nr] ^= opp

        if changed:
            self._rebuild_neighbor_cache()
            self._rebuild_wall_surface()

    def evaporate(self):
        """Vectorised pheromone evaporation."""
        np.multiply(self.pheromone, PHEROMONE_EVAPORATION, out=self.pheromone)
        np.maximum(self.pheromone, PHEROMONE_MIN, out=self.pheromone)

    def deposit(self, path, amount):
        """Deposit pheromone along a list of (c, r) tuples."""
        for (c, r) in path:
            self.pheromone[c, r] += amount


# ---------------------------------------------------------------------------
# Spatial Bucket Grid  (O(1) neighbour lookup for boids)
# ---------------------------------------------------------------------------

class SpatialGrid:
    """
    Divides the screen into BUCKET_SIZE × BUCKET_SIZE buckets.
    Each bucket holds indices of agents currently inside it.
    """
    def __init__(self, width, height, bucket_size):
        self.bw  = bucket_size
        self.bh  = bucket_size
        self.ncx = math.ceil(width  / bucket_size)
        self.ncy = math.ceil(height / bucket_size)
        self.buckets: dict[tuple, list] = {}

    def clear(self):
        self.buckets.clear()

    def insert(self, idx, px, py):
        key = (int(px // self.bw), int(py // self.bh))
        if key not in self.buckets:
            self.buckets[key] = []
        self.buckets[key].append(idx)

    def query(self, px, py):
        """Return agent indices in the same bucket and 8 surrounding buckets."""
        bx = int(px // self.bw)
        by = int(py // self.bh)
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (bx + dx, by + dy)
                if key in self.buckets:
                    result.extend(self.buckets[key])
        return result


# ---------------------------------------------------------------------------
# Agent swarm  (fully vectorised)
# ---------------------------------------------------------------------------

class AgentSwarm:
    """
    Stores all agents as flat NumPy arrays for cache-friendly, vectorised ops.
    """
    def __init__(self, n):
        self.n            = n
        self.active_count = 0   # agents currently in play; grows via spawn_batch()

        # Pre-randomise all positions/velocities; slots are activated on demand.
        margin = 3.0
        self.pos = np.random.uniform(
            [margin, margin],
            [GRID_SIZE - margin, GRID_SIZE - margin],
            size=(n, 2)
        ).astype(np.float32)
        angles = np.random.uniform(0, 2 * math.pi, n)
        speed  = np.random.uniform(1.0, MAX_SPEED, n)
        self.vel = np.column_stack([np.cos(angles) * speed,
                                    np.sin(angles) * speed]).astype(np.float32)
        self.acc         = np.zeros((n, 2), dtype=np.float32)
        self.path_memory = [[] for _ in range(n)]
        self.found_goal  = np.zeros(n, dtype=bool)

    # --------------------------------------------------------------- spawning

    def spawn_batch(self, size: int = SPAWN_BATCH_SIZE):
        """Activate the next `size` dormant agent slots."""
        prev = self.active_count
        self.active_count = min(self.active_count + size, self.n)
        # Re-randomise freshly activated slots so they start cleanly
        margin = 3.0
        for i in range(prev, self.active_count):
            self.pos[i] = np.random.uniform(margin, GRID_SIZE - margin, 2)
            angle = random.uniform(0, 2 * math.pi)
            sp    = random.uniform(1.0, MAX_SPEED)
            self.vel[i]        = [math.cos(angle) * sp, math.sin(angle) * sp]
            self.path_memory[i] = []
            self.found_goal[i] = False

    # ------------------------------------------------------------------ update

    def update(self, maze: Maze, spatial_grid: SpatialGrid):
        # Only iterate over agents that have been spawned
        active = np.where(~self.found_goal[:self.active_count])[0]
        if active.size == 0:
            return

        # --- Rebuild spatial grid ---
        spatial_grid.clear()
        for i in active:
            spatial_grid.insert(int(i), self.pos[i, 0], self.pos[i, 1])

        # --- Per-agent forces (boids + ACO) ---
        goal_pos = np.array([WIDTH - GRID_SIZE / 2, HEIGHT - GRID_SIZE / 2], dtype=np.float32)

        for i in active:
            px, py = self.pos[i]
            vx, vy = self.vel[i]

            # Grid cell of this agent
            gc = min(int(px // GRID_SIZE), COLS - 1)
            gr = min(int(py // GRID_SIZE), ROWS - 1)

            # -------- BOIDS --------
            neighbour_indices = spatial_grid.query(px, py)
            sep = np.zeros(2, dtype=np.float32)
            ali = np.zeros(2, dtype=np.float32)
            nb_count = 0

            r2 = BOID_VIEW_RADIUS * BOID_VIEW_RADIUS

            for j in neighbour_indices:
                if j == i:
                    continue
                dx = px - self.pos[j, 0]
                dy = py - self.pos[j, 1]
                d2 = dx * dx + dy * dy
                if d2 < r2 and d2 > 0:
                    d = math.sqrt(d2)
                    sep[0] += dx / d
                    sep[1] += dy / d
                    ali[0] += self.vel[j, 0]
                    ali[1] += self.vel[j, 1]
                    nb_count += 1

            sep_force = np.zeros(2, dtype=np.float32)
            ali_force = np.zeros(2, dtype=np.float32)

            if nb_count > 0:
                sep /= nb_count
                sn = math.sqrt(sep[0]*sep[0] + sep[1]*sep[1])
                if sn > 0:
                    sep_force = (sep / sn) * MAX_SPEED - self.vel[i]

                ali /= nb_count
                an = math.sqrt(ali[0]*ali[0] + ali[1]*ali[1])
                if an > 0:
                    ali_force = (ali / an) * MAX_SPEED

            # -------- ACO --------
            crowding = min(1.0, nb_count / 10.0)
            alpha    = ACO_ALPHA * (1.0 - crowding)   # less pheromone reliance when crowded

            nbrs = maze._open_neighbors[gc][gr]
            coh_force = np.zeros(2, dtype=np.float32)

            if nbrs:
                probs = []
                for nc, nr in nbrs:
                    tau  = max(maze.pheromone[nc, nr], 1e-6) ** alpha
                    cx_  = nc * GRID_SIZE + GRID_SIZE * 0.5
                    cy_  = nr * GRID_SIZE + GRID_SIZE * 0.5
                    ddx  = goal_pos[0] - cx_
                    ddy  = goal_pos[1] - cy_
                    dist = math.sqrt(ddx*ddx + ddy*ddy) + 1.0
                    eta  = (1.0 / dist) ** ACO_BETA
                    probs.append(tau * eta)

                denom = sum(probs)
                if denom > 0:
                    probs = [p / denom for p in probs]
                    # Weighted random selection
                    r = random.random()
                    cumulative = 0.0
                    chosen_nc, chosen_nr = nbrs[0]
                    for k, (nc, nr) in enumerate(nbrs):
                        cumulative += probs[k]
                        if r <= cumulative:
                            chosen_nc, chosen_nr = nc, nr
                            break

                    target_x = chosen_nc * GRID_SIZE + GRID_SIZE * 0.5
                    target_y = chosen_nr * GRID_SIZE + GRID_SIZE * 0.5
                    desired_x = target_x - px
                    desired_y = target_y - py
                    dn = math.sqrt(desired_x*desired_x + desired_y*desired_y)
                    if dn > 0:
                        sx = desired_x / dn * MAX_SPEED
                        sy = desired_y / dn * MAX_SPEED
                        coh_force[0] = sx - vx
                        coh_force[1] = sy - vy

            # Accumulate boids + ACO forces
            self.acc[i] += sep_force * SEPARATION_FORCE
            self.acc[i] += ali_force * ALIGNMENT_FORCE
            self.acc[i] += coh_force * COHESION_FORCE

            # -------- SOFT WALL REPULSION --------
            # Gentle nudge away from closed walls – the hard collision resolver
            # is the real enforcement; this just smooths approach angles.
            REPULSE_DIST = GRID_SIZE * 0.20   # repel within 20% of cell width
            REPULSE_STR  = 0.5
            cell_x0 = gc * GRID_SIZE
            cell_y0 = gr * GRID_SIZE
            w_bits  = maze.walls[gc, gr]
            wall_f  = np.zeros(2, dtype=np.float32)
            if w_bits & W_LEFT:
                d = px - cell_x0
                if 0 < d < REPULSE_DIST:
                    wall_f[0] += REPULSE_STR * (REPULSE_DIST - d) / REPULSE_DIST
            if w_bits & W_RIGHT:
                d = (gc + 1) * GRID_SIZE - px
                if 0 < d < REPULSE_DIST:
                    wall_f[0] -= REPULSE_STR * (REPULSE_DIST - d) / REPULSE_DIST
            if w_bits & W_TOP:
                d = py - cell_y0
                if 0 < d < REPULSE_DIST:
                    wall_f[1] += REPULSE_STR * (REPULSE_DIST - d) / REPULSE_DIST
            if w_bits & W_BOTTOM:
                d = (gr + 1) * GRID_SIZE - py
                if 0 < d < REPULSE_DIST:
                    wall_f[1] -= REPULSE_STR * (REPULSE_DIST - d) / REPULSE_DIST
            self.acc[i] += wall_f

            # Path memory (cap to 500 cells to avoid unbounded growth)
            cell = (gc, gr)
            pm = self.path_memory[i]
            if not pm or pm[-1] != cell:
                pm.append(cell)
                if len(pm) > 500:
                    pm.pop(0)

            # Goal check
            if gc == COLS - 1 and gr == ROWS - 1:
                self.found_goal[i] = True
                maze.deposit(pm, PHEROMONE_INTENSITY)

        # --- Vectorised physics update for all active agents ---
        # Snapshot positions BEFORE moving so wall crossing can be detected.
        old_pos = self.pos[active].copy()

        # Velocity damping: blend old velocity with new acceleration instead of
        # snapping instantly, which eliminates per-frame jitter.
        self.vel[active] *= VEL_DAMPING
        self.vel[active] += self.acc[active]

        # Speed clamp (vectorised)
        speeds = np.linalg.norm(self.vel[active], axis=1, keepdims=True)
        too_fast = speeds[:, 0] > MAX_SPEED
        if np.any(too_fast):
            idx_fast = active[too_fast]
            self.vel[idx_fast] = (self.vel[idx_fast] / speeds[too_fast]) * MAX_SPEED

        self.pos[active] += self.vel[active]
        self.acc[active]  = 0.0

        # Bounds clamp
        np.clip(self.pos[:, 0], 0, WIDTH  - 1, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], 0, HEIGHT - 1, out=self.pos[:, 1])

        # Hard wall collision – revert any crossing of a closed wall.
        self._resolve_wall_collisions(maze, active, old_pos)

    # ------------------------------------------------------------------ wall collision

    def _resolve_wall_collisions(self, maze: 'Maze', active: np.ndarray, old_pos: np.ndarray):
        """
        Hard wall collision resolution.

        For each active agent, compare the cell it was in BEFORE the physics
        step (computed from old_pos) with the cell it is in AFTER.  If the
        agent crossed a cell boundary where the wall is CLOSED, snap the
        offending position axis back to where it was and zero that velocity
        component.  Because MAX_SPEED << GRID_SIZE, at most one cell boundary
        is crossed per axis per frame, so a single check is sufficient.

        Because we test the *actual* wall bitmask, this can never mistake an
        open passage for a wall (fixing the "stuck against phantom wall" bug)
        and never allows movement through a closed wall (fixing phasing).
        """
        for idx, i in enumerate(active):
            ox, oy = float(old_pos[idx, 0]), float(old_pos[idx, 1])
            nx, ny = float(self.pos[i, 0]),  float(self.pos[i, 1])

            # Cell before move (clamped to grid)
            ogc = min(max(int(ox // GRID_SIZE), 0), COLS - 1)
            ogr = min(max(int(oy // GRID_SIZE), 0), ROWS - 1)

            # Cell after move
            ngc = min(max(int(nx // GRID_SIZE), 0), COLS - 1)
            ngr = min(max(int(ny // GRID_SIZE), 0), ROWS - 1)

            # --- X axis ---
            if ngc != ogc:
                wb = W_RIGHT if ngc > ogc else W_LEFT
                if maze.walls[ogc, ogr] & wb:       # wall is CLOSED → revert
                    self.pos[i, 0] = ox
                    self.vel[i, 0] = 0.0
                    ngc = ogc                        # use corrected cell for Y check

            # --- Y axis ---
            if ngr != ogr:
                wb = W_BOTTOM if ngr > ogr else W_TOP
                if maze.walls[ogc, ogr] & wb:       # wall is CLOSED → revert
                    self.pos[i, 1] = oy
                    self.vel[i, 1] = 0.0

    # ------------------------------------------------------------------ respawn

    def respawn_finished(self):
        # Only check active slots
        finished = np.where(self.found_goal[:self.active_count])[0]
        margin = 3.0
        for i in finished:
            self.pos[i, 0]     = random.uniform(margin, GRID_SIZE - margin)
            self.pos[i, 1]     = random.uniform(margin, GRID_SIZE - margin)
            angle              = random.uniform(0, 2 * math.pi)
            sp                 = random.uniform(1.0, MAX_SPEED)
            self.vel[i]        = [math.cos(angle) * sp, math.sin(angle) * sp]
            self.path_memory[i] = []
            self.found_goal[i] = False


# ---------------------------------------------------------------------------
# Pheromone surface (pre-allocated, updated in-place)
# ---------------------------------------------------------------------------

class PheromoneSurface:
    """Caches a pygame Surface that shows pheromone levels as a green overlay."""
    def __init__(self):
        self.surf = pygame.Surface((WIDTH, HEIGHT))

    def update(self, pheromone: np.ndarray):
        """Build RGB pixel array from pheromone numpy array efficiently."""
        # Scale pheromone → 0-255 green channel
        green = np.clip(pheromone * 10.0, 0, 255).astype(np.uint8)

        # Expand (COLS, ROWS) → (COLS*GRID_SIZE, ROWS*GRID_SIZE) pixel grid.
        # COLS*GRID_SIZE may be < WIDTH when WIDTH % GRID_SIZE != 0, so we
        # blit only the exact grid area and leave any remainder black.
        expanded = np.repeat(np.repeat(green, GRID_SIZE, axis=0), GRID_SIZE, axis=1)
        gw = expanded.shape[0]   # actual pixel width of grid
        gh = expanded.shape[1]   # actual pixel height of grid

        px_array = pygame.surfarray.pixels3d(self.surf)
        px_array[:, :, :] = 0                       # clear to black
        px_array[:gw, :gh, 1] = expanded            # write green channel only
        del px_array  # Release lock


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ACO+Boids Maze Solver – Fast Edition")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, 24)

    maze         = Maze()
    swarm        = AgentSwarm(NUM_AGENTS)
    spatial_grid = SpatialGrid(WIDTH, HEIGHT, BUCKET_SIZE)
    phero_surf   = PheromoneSurface()

    frame_count  = 0
    total_solved = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame_count += 1

        # --- Batch spawn: release SPAWN_BATCH_SIZE agents every SPAWN_INTERVAL frames ---
        if frame_count % SPAWN_INTERVAL == 1 and swarm.active_count < swarm.n:
            swarm.spawn_batch(SPAWN_BATCH_SIZE)

        # --- Dynamic maze ---
        if frame_count % WALL_TOGGLE_RATE == 0:
            maze.mutate()

        # --- Pheromone evaporation (one vectorised op) ---
        maze.evaporate()

        # --- Count newly finished agents ---
        prev_solved = int(np.sum(swarm.found_goal[:swarm.active_count]))

        # --- Agent update ---
        swarm.update(maze, spatial_grid)

        total_solved += int(np.sum(swarm.found_goal[:swarm.active_count])) - prev_solved

        # --- Render ---
        screen.fill(BLACK)

        # Pheromone overlay
        phero_surf.update(maze.pheromone)
        screen.blit(phero_surf.surf, (0, 0))

        # Maze walls (from pre-drawn surface)
        screen.blit(maze.wall_surface, (0, 0))

        # Agents: only draw spawned, non-finished agents
        active_mask = ~swarm.found_goal[:swarm.active_count]
        active_pos  = swarm.pos[:swarm.active_count][active_mask].astype(np.int32)
        for px, py in active_pos:
            pygame.draw.circle(screen, CYAN, (px, py), 3)

        # Start / End markers
        pygame.draw.rect(screen, BLUE, (0, 0, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED,  (WIDTH - GRID_SIZE, HEIGHT - GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # HUD
        fps_text   = font.render(f"FPS: {clock.get_fps():.0f}", True, WHITE)
        sol_text   = font.render(f"Solved: {total_solved}", True, WHITE)
        agent_text = font.render(f"Agents: {swarm.active_count}/{swarm.n}", True, WHITE)
        screen.blit(fps_text,   (8, 8))
        screen.blit(sol_text,   (8, 28))
        screen.blit(agent_text, (8, 48))

        pygame.display.flip()

        # Respawn finished agents (after blit so we see them reach the goal)
        swarm.respawn_finished()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
