"""
boids-sim.py
============
Classic Boids simulation in open space.

Features
--------
• Flock of 120 boids navigating toward rotating random goals
• Predator that crosses the screen at random intervals (H or V) –
  boids within the fear radius scatter away from it
• Interactive sliders (drag the knobs) to tune:
    Separation / Alignment / Cohesion in real time
• Boids rendered as small directional triangles
• Screen-wrapping so the flock never disappears off an edge
"""

import pygame
import random
import math
import numpy as np

# ---------------------------------------------------------------------------
# Window layout
# ---------------------------------------------------------------------------
WIN_W, WIN_H  = 960, 780
PANEL_H       = 110             # slider panel at bottom
SIM_W, SIM_H  = WIN_W, WIN_H - PANEL_H   # actual boid space

# ---------------------------------------------------------------------------
# Boids parameters (live-modified by sliders)
# ---------------------------------------------------------------------------
NUM_BOIDS   = 120
MAX_SPEED   = 4.0
MIN_SPEED   = 1.5
VEL_DAMP    = 0.97          # higher = less sudden slowdown → smoother glide
MAX_FORCE   = 0.25          # cap per-frame steering to prevent jerky snapping
BOID_RADIUS = 70.0              # neighbour detection radius

SEP_W_INIT  = 1.8
ALI_W_INIT  = 1.2
COH_W_INIT  = 0.9
GOAL_W      = 0.6               # goal attraction (fixed)

# ---------------------------------------------------------------------------
# Goal parameters
# ---------------------------------------------------------------------------
NUM_GOALS         = 2
GOAL_COLLECT_DIST = 35          # dist for a boid to "arrive" at goal
GOAL_COLLECT_COUNT= 5           # boids needed near goal to consume it

# ---------------------------------------------------------------------------
# Predator parameters
# ---------------------------------------------------------------------------
PREDATOR_SPEED      = 9.0
PREDATOR_FEAR_DIST  = 120.0     # flee radius
PREDATOR_FLEE_STR   = 5.0
PRED_INTERVAL_MIN   = 240       # frames between predators (min)
PRED_INTERVAL_MAX   = 480       # frames between predators (max)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BG_COLOR      = ( 10,  10,  22)
BOID_COLOR    = ( 80, 200, 255)
BOID_OUTLINE  = ( 40, 120, 180)
GOAL_COLOR    = (255, 210,  50)
GOAL_GLOW     = (255, 160,  20)
PRED_COLOR    = (220,  40,  40)
PRED_GLOW     = (160,  20,  20)
PANEL_BG      = ( 18,  18,  35)
WHITE         = (255, 255, 255)
GREY          = (140, 140, 160)
SLIDER_TRACK  = ( 50,  50,  80)
SLIDER_FILL   = ( 70, 140, 230)
SLIDER_KNOB   = (220, 230, 255)

# ---------------------------------------------------------------------------
# Spatial bucket grid
# ---------------------------------------------------------------------------

class SpatialGrid:
    def __init__(self, bucket: float):
        self.bs      = bucket
        self.buckets: dict = {}

    def clear(self):
        self.buckets.clear()

    def insert(self, idx: int, x: float, y: float):
        k = (int(x // self.bs), int(y // self.bs))
        if k not in self.buckets: self.buckets[k] = []
        self.buckets[k].append(idx)

    def query(self, x: float, y: float):
        bx, by = int(x // self.bs), int(y // self.bs)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (bx+dx, by+dy)
                if k in self.buckets:
                    out.extend(self.buckets[k])
        return out

# ---------------------------------------------------------------------------
# Slider UI
# ---------------------------------------------------------------------------

class Slider:
    KNOB_R = 10

    def __init__(self, cx: int, cy: int, width: int,
                 lo: float, hi: float, init: float, label: str):
        self.x     = cx - width // 2
        self.y     = cy
        self.w     = width
        self.lo    = lo
        self.hi    = hi
        self.value = init
        self.label = label
        self.drag  = False
        self.track = pygame.Rect(self.x, cy - 4, width, 8)

    @property
    def knob_x(self) -> int:
        t = (self.value - self.lo) / (self.hi - self.lo)
        return int(self.x + t * self.w)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            kx, ky = self.knob_x, self.y
            if math.hypot(event.pos[0]-kx, event.pos[1]-ky) < self.KNOB_R + 6:
                self.drag = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        elif event.type == pygame.MOUSEMOTION and self.drag:
            t = max(0.0, min(1.0, (event.pos[0] - self.x) / self.w))
            self.value = self.lo + t * (self.hi - self.lo)

    def draw(self, screen, font):
        # Track background
        pygame.draw.rect(screen, SLIDER_TRACK, self.track, border_radius=4)
        # Filled portion
        fill_w = max(0, self.knob_x - self.x)
        fill_r = pygame.Rect(self.x, self.y - 4, fill_w, 8)
        pygame.draw.rect(screen, SLIDER_FILL, fill_r, border_radius=4)
        # Knob
        kx = self.knob_x
        pygame.draw.circle(screen, SLIDER_KNOB, (kx, self.y), self.KNOB_R)
        pygame.draw.circle(screen, (180, 200, 255), (kx, self.y), self.KNOB_R, 2)
        # Label above
        txt = font.render(f"{self.label}  {self.value:.2f}", True, WHITE)
        screen.blit(txt, (self.x, self.y - self.KNOB_R - 24))

# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

class Goal:
    def __init__(self):
        self.respawn()
        self.pulse = random.uniform(0, math.tau)

    def respawn(self):
        margin = 60
        self.x  = random.randint(margin, SIM_W - margin)
        self.y  = random.randint(margin, SIM_H - margin)
        self.nearby_count = 0

    def update(self):
        self.pulse += 0.07

    def draw(self, screen):
        gx, gy = int(self.x), int(self.y)
        r_outer = int(22 + math.sin(self.pulse) * 5)
        # Outer glow rings
        for ring_r in range(r_outer + 18, r_outer, -4):
            alpha = max(0, 100 - (ring_r - r_outer) * 12)
            s = pygame.Surface((ring_r*2, ring_r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*GOAL_GLOW, alpha), (ring_r, ring_r), ring_r)
            screen.blit(s, (gx - ring_r, gy - ring_r))
        pygame.draw.circle(screen, GOAL_COLOR, (gx, gy), r_outer)
        pygame.draw.circle(screen, WHITE,      (gx, gy), r_outer, 2)

# ---------------------------------------------------------------------------
# Predator
# ---------------------------------------------------------------------------

class Predator:
    SIZE = 18

    def __init__(self):
        horiz = random.random() < 0.5
        if horiz:
            going_right  = random.random() < 0.5
            self.y       = random.randint(40, SIM_H - 40)
            if going_right:
                self.x, self.vx, self.vy = -30.0, PREDATOR_SPEED, 0.0
            else:
                self.x, self.vx, self.vy = float(WIN_W + 30), -PREDATOR_SPEED, 0.0
        else:
            going_down   = random.random() < 0.5
            self.x       = random.randint(40, SIM_W - 40)
            if going_down:
                self.y, self.vx, self.vy = -30.0, 0.0, PREDATOR_SPEED
            else:
                self.y, self.vx, self.vy = float(SIM_H + 30), 0.0, -PREDATOR_SPEED
        self.active = True
        self.pulse  = 0.0

    def update(self):
        self.x     += self.vx
        self.y     += self.vy
        self.pulse += 0.15
        if (self.x < -60 or self.x > WIN_W + 60 or
                self.y < -60 or self.y > SIM_H + 60):
            self.active = False

    def draw(self, screen):
        px, py  = int(self.x), int(self.y)
        angle   = math.atan2(self.vy, self.vx)
        S       = self.SIZE

        # Fear radius (subtle pulsing ring)
        ring_r = int(PREDATOR_FEAR_DIST)
        ring_s = pygame.Surface((ring_r*2, ring_r*2), pygame.SRCALPHA)
        alpha  = int(40 + 20 * math.sin(self.pulse))
        pygame.draw.circle(ring_s, (*PRED_GLOW, alpha), (ring_r, ring_r), ring_r)
        screen.blit(ring_s, (px - ring_r, py - ring_r))

        def pt(da, scale=1.0):
            a = angle + da
            return (int(px + math.cos(a)*S*scale), int(py + math.sin(a)*S*scale))

        # Predator body – sharp red triangle
        tri = [pt(0, 1.2), pt(math.pi*0.75, 0.7), pt(-math.pi*0.75, 0.7)]
        pygame.draw.polygon(screen, PRED_COLOR, tri)
        pygame.draw.polygon(screen, (255, 100, 100), tri, 2)

# ---------------------------------------------------------------------------
# Boids swarm (vectorised)
# ---------------------------------------------------------------------------

class BoidsSwarm:
    def __init__(self, n: int):
        self.n   = n
        # Spawn boids in small clusters so flocking rules fire immediately
        # (avoids the "frozen strangers" look at startup)
        num_clusters  = max(6, n // 12)
        cluster_r     = 60.0        # radius of each cluster
        cx = np.random.uniform(120, SIM_W - 120, num_clusters)
        cy = np.random.uniform(120, SIM_H - 120, num_clusters)
        cids = np.random.randint(0, num_clusters, n)
        offx = np.random.uniform(-cluster_r, cluster_r, n)
        offy = np.random.uniform(-cluster_r, cluster_r, n)
        px   = np.clip(cx[cids] + offx, 10, SIM_W - 10)
        py   = np.clip(cy[cids] + offy, 10, SIM_H - 10)
        self.pos = np.column_stack([px, py]).astype(np.float32)
        # All boids start at MAX_SPEED so they're moving right away
        angles   = np.random.uniform(0, math.tau, n)
        self.vel = np.column_stack([np.cos(angles) * MAX_SPEED,
                                    np.sin(angles) * MAX_SPEED]).astype(np.float32)
        self.acc = np.zeros((n, 2), dtype=np.float32)

    def update(self,
               sg: SpatialGrid,
               goals: list,
               predator,
               sep_w: float,
               ali_w: float,
               coh_w: float):

        n = self.n

        # Rebuild spatial grid
        sg.clear()
        for i in range(n):
            sg.insert(i, float(self.pos[i, 0]), float(self.pos[i, 1]))

        r2       = BOID_RADIUS * BOID_RADIUS
        pred_r2  = PREDATOR_FEAR_DIST ** 2

        for i in range(n):
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])

            # ---- Boids rules ----
            sep = np.zeros(2, np.float32)
            ali = np.zeros(2, np.float32)
            coh = np.zeros(2, np.float32)
            cnt = 0

            for j in sg.query(px, py):
                if j == i: continue
                dx = px - float(self.pos[j, 0])
                dy = py - float(self.pos[j, 1])
                d2 = dx*dx + dy*dy
                if 0 < d2 < r2:
                    d    = math.sqrt(d2)
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
                sn = math.hypot(sep[0], sep[1])
                if sn > 0:
                    f_sep = (sep / sn) * MAX_SPEED - self.vel[i]

                ali /= cnt
                an = math.hypot(ali[0], ali[1])
                if an > 0:
                    f_ali = (ali / an) * MAX_SPEED - self.vel[i]

                coh /= cnt
                desired = coh - self.pos[i]
                dn = math.hypot(desired[0], desired[1])
                if dn > 0:
                    f_coh = (desired / dn) * MAX_SPEED - self.vel[i]

            # ---- Goal attraction (nearest goal) ----
            f_goal = np.zeros(2, np.float32)
            best_d = float('inf')
            best_g = None
            for g in goals:
                gx, gy = g.x - px, g.y - py
                d = math.hypot(gx, gy)
                if d < best_d:
                    best_d, best_g = d, (gx, gy)
            if best_g is not None and best_d > 1:
                gx, gy = best_g
                f_goal = np.array([gx / best_d, gy / best_d], np.float32) * MAX_SPEED * GOAL_W

            # ---- Predator avoidance ----
            f_flee = np.zeros(2, np.float32)
            if predator is not None and predator.active:
                dx = px - predator.x
                dy = py - predator.y
                d2 = dx*dx + dy*dy
                if 0 < d2 < pred_r2:
                    d = math.sqrt(d2)
                    # Force increases sharply as predator gets closer
                    strength  = (PREDATOR_FEAR_DIST / d) ** 2
                    f_flee[0] = dx / d * strength * PREDATOR_FLEE_STR
                    f_flee[1] = dy / d * strength * PREDATOR_FLEE_STR

            # Accumulate steering forces
            raw = (f_sep * sep_w
                   + f_ali * ali_w
                   + f_coh * coh_w
                   + f_goal
                   + f_flee)
            # Clamp magnitude so no single frame can snap a boid sharply
            rmag = math.hypot(raw[0], raw[1])
            if rmag > MAX_FORCE:
                raw = raw * (MAX_FORCE / rmag)
            self.acc[i] += raw

        # ---- Vectorised physics ----
        self.vel *= VEL_DAMP
        self.vel += self.acc

        speeds = np.linalg.norm(self.vel, axis=1, keepdims=True)
        # Clamp to [MIN_SPEED, MAX_SPEED]
        too_slow = speeds[:, 0] < MIN_SPEED
        too_fast = speeds[:, 0] > MAX_SPEED
        if np.any(too_slow):
            ri = np.where(too_slow)[0]
            self.vel[ri] = (self.vel[ri] / np.maximum(speeds[ri], 1e-6)) * MIN_SPEED
        if np.any(too_fast):
            ri = np.where(too_fast)[0]
            self.vel[ri] = (self.vel[ri] / speeds[ri]) * MAX_SPEED

        self.pos += self.vel
        self.acc[:]  = 0.0

        # Screen wrap
        self.pos[:, 0] %= SIM_W
        self.pos[:, 1] %= SIM_H

    def draw(self, screen):
        for i in range(self.n):
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
            vx, vy = float(self.vel[i, 0]), float(self.vel[i, 1])
            spd    = math.hypot(vx, vy)
            angle  = math.atan2(vy, vx) if spd > 0.01 else 0.0

            # Triangle boid: tip in direction of travel, base behind
            TIP  = 8
            BASE = 6
            def pt(da, r):
                return (int(px + math.cos(angle+da)*r),
                        int(py + math.sin(angle+da)*r))

            tip  = pt(0,               TIP)
            bl   = pt(math.pi * 0.75,  BASE)
            br   = pt(-math.pi * 0.75, BASE)

            pygame.draw.polygon(screen, BOID_COLOR, [tip, bl, br])
            pygame.draw.polygon(screen, BOID_OUTLINE, [tip, bl, br], 1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Boids Simulation – Open Space")
    clock  = pygame.time.Clock()
    font_s = pygame.font.SysFont("consolas", 17)
    font_b = pygame.font.SysFont("consolas", 20, bold=True)

    # Pre-render a subtle radial gradient background
    bg = pygame.Surface((WIN_W, WIN_H))
    bg.fill(BG_COLOR)
    for r in range(460, 0, -1):
        alpha = int(30 * (1 - r/460))
        s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (30, 40, 80, alpha), (r, r), r)
        bg.blit(s, (WIN_W//2 - r, SIM_H//2 - r))

    # Slider panel background
    panel_rect = pygame.Rect(0, SIM_H, WIN_W, PANEL_H)

    # Sliders (centered vertically in the panel)
    sy      = SIM_H + PANEL_H // 2 + 10
    sw      = 220
    sliders = [
        Slider(WIN_W//2 - sw - 40, sy, sw, 0.0, 3.0, SEP_W_INIT, "Separation"),
        Slider(WIN_W//2,           sy, sw, 0.0, 3.0, ALI_W_INIT, "Alignment"),
        Slider(WIN_W//2 + sw + 40, sy, sw, 0.0, 3.0, COH_W_INIT, "Cohesion"),
    ]

    sg      = SpatialGrid(BOID_RADIUS)
    swarm   = BoidsSwarm(NUM_BOIDS)
    goals   = [Goal() for _ in range(NUM_GOALS)]
    predator = None
    next_pred_frame = random.randint(PRED_INTERVAL_MIN, PRED_INTERVAL_MAX)

    frame       = 0
    total_found = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            for sl in sliders:
                sl.handle(event)

        frame += 1

        # ---- Predator spawn ----
        if frame >= next_pred_frame and (predator is None or not predator.active):
            predator = Predator()
            next_pred_frame = frame + random.randint(PRED_INTERVAL_MIN, PRED_INTERVAL_MAX)

        if predator and not predator.active:
            predator = None

        if predator:
            predator.update()

        # ---- Goal logic ----
        for g in goals:
            g.update()
            g.nearby_count = 0

        # Count boids near each goal
        for i in range(NUM_BOIDS):
            px, py = float(swarm.pos[i, 0]), float(swarm.pos[i, 1])
            for g in goals:
                if math.hypot(px - g.x, py - g.y) < GOAL_COLLECT_DIST:
                    g.nearby_count += 1

        for g in goals:
            if g.nearby_count >= GOAL_COLLECT_COUNT:
                total_found += 1
                g.respawn()

        # ---- Swarm update ----
        swarm.update(sg, goals, predator,
                     sliders[0].value,
                     sliders[1].value,
                     sliders[2].value)

        # ---- Render ----
        screen.blit(bg, (0, 0))

        # Goals
        for g in goals:
            g.draw(screen)

        # Predator
        if predator and predator.active:
            predator.draw(screen)

        # Boids
        swarm.draw(screen)

        # Panel
        pygame.draw.rect(screen, PANEL_BG, panel_rect)
        pygame.draw.line(screen, (60, 60, 100), (0, SIM_H), (WIN_W, SIM_H), 2)

        for sl in sliders:
            sl.draw(screen, font_s)

        # HUD
        fps_t  = font_s.render(f"FPS: {clock.get_fps():.0f}  |  "
                               f"Boids: {NUM_BOIDS}  |  "
                               f"Goals collected: {total_found}", True, GREY)
        pred_t = font_s.render(
            "PREDATOR INCOMING!" if (predator and predator.active) else "",
            True, (255, 80, 80))
        esc_t  = font_s.render("ESC to quit", True, (80, 80, 100))
        screen.blit(fps_t,  (10, 6))
        screen.blit(pred_t, (WIN_W // 2 - pred_t.get_width() // 2, 32))
        screen.blit(esc_t,  (WIN_W - esc_t.get_width() - 10, 6))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
