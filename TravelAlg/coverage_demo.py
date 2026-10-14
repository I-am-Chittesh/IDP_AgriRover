#!/usr/bin/env python3
import argparse
import heapq
import math
import os
import sys
from dataclasses import dataclass

import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle

EPS = 1e-6

BG, PANEL, LIME, AMBER = "#121417", "#191D21", "#CCFF00", "#FFB000"
OFF, MUTED, OBST, RED = "#F2F1EC", "#9BA3AB", "#3A4047", "#FF5C5C"
CELL_COLORS = ["#CCFF00", "#FFB000", "#2EC4B6", "#8B7CFF", "#FF6B6B", "#6CB4FF",
               "#F4A6D7", "#B5E48C", "#FFD166", "#06D6A0", "#EF476F", "#118AB2"]


@dataclass
class Params:
    swath_w: float = 1.5      # sensor footprint width on the ground (m)
    overlap: float = 0.10     # fraction of footprint that adjacent swaths share
    robot_r: float = 0.4      # robot radius, used for headland and obstacle inflation (m)
    min_len: float = 0.8      # free pockets thinner than this are treated as blocked (m)
    turn_cost: float = 2.0    # one skid-steer pivot is scored like this many meters of driving
    grid_res: float = 0.2     # A* grid resolution (m)

    @property
    def d(self):
        """Distance between neighbouring swath lines."""
        return self.swath_w * (1.0 - self.overlap)


@dataclass
class Problem:
    W: float
    H: float
    obstacles: list
    home: tuple

    def transposed(self):
        """Swap x and y. Planning with vertical swaths on the transposed problem
        gives horizontal swaths on the real one."""
        return Problem(self.H, self.W,
                       [(y0, x0, y1, x1) for x0, y0, x1, y1 in self.obstacles],
                       list(self.names), (self.home[1], self.home[0]))


@dataclass
class Cell:
    id: int
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def width(self):
        return self.x1 - self.x0


@dataclass
class Plan:
    sweep_axis: str
    cells: list
    crit: list
    route: list               
    segments: list
    skipped: list


def inflate(obstacles, r):
    return [(x0 - r, y0 - r, x1 + r, y1 + r) for x0, y0, x1, y1 in obstacles]

class Space:
    """Usable rectangle minus inflated obstacles. Points on the boundary count as free."""

    def __init__(self, prob, robot_r):
        self.xlo, self.xhi = robot_r, prob.W - robot_r
        self.ylo, self.yhi = robot_r, prob.H - robot_r
        self.infl = inflate(prob.obstacles, robot_r)
        self._grids = {}

    def free_points(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        ok = ((x >= self.xlo - EPS) & (x <= self.xhi + EPS) &
              (y >= self.ylo - EPS) & (y <= self.yhi + EPS))
        for ox0, oy0, ox1, oy1 in self.infl:
            ok &= ~((x > ox0 + EPS) & (x < ox1 - EPS) & (y > oy0 + EPS) & (y < oy1 - EPS))
        return ok

    def line_free(self, a, b, step=0.05):
        n = max(2, int(math.dist(a, b) / step) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = np.asarray(a, float)[None, :] * (1 - t) + np.asarray(b, float)[None, :] * t
        return bool(self.free_points(pts).all())

    def grid(self, res):
        if res not in self._grids:
            nx = int((self.xhi - self.xlo) / res + 1e-9) + 1
            ny = int((self.yhi - self.ylo) / res + 1e-9) + 1
            xs = self.xlo + np.arange(nx) * res
            ys = self.ylo + np.arange(ny) * res
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            blocked = ~self.free_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
            self._grids[res] = (xs, ys, blocked)
        return self._grids[res]


def free_intervals(xa, xb, infl, ylo, yhi, min_len):
    """Free y-intervals on a sweep line anywhere inside the slab [xa, xb]."""
    blocked = []
    for ox0, oy0, ox1, oy1 in infl:
        if ox0 < xb - EPS and ox1 > xa + EPS:
            lo, hi = max(oy0, ylo), min(oy1, yhi)
            if hi > lo:
                blocked.append([lo, hi])
    blocked.sort()
    merged = []
    for lo, hi in blocked:
        if merged and lo <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    free, cur = [], ylo
    for lo, hi in merged:
        if lo - cur >= min_len:
            free.append((cur, lo))
        cur = max(cur, hi)
    if yhi - cur >= min_len:
        free.append((cur, yhi))
    return free


def decompose(prob, P):
    """Sweep a vertical line left to right.
    Between two obstacle edges the free intervals on the line stay the same (a slab).
    If a slab's interval equals the previous slab's interval, the cell just grows.
    Otherwise (obstacle edge = split or merge event) a new cell starts."""
    space = Space(prob, P.robot_r)
    xs = {space.xlo, space.xhi}
    for ox0, _, ox1, _ in space.infl:
        for x in (ox0, ox1):
            if space.xlo + EPS < x < space.xhi - EPS:
                xs.add(x)
    xs = sorted(xs)
    cells, open_cells = [], []
    for xa, xb in zip(xs[:-1], xs[1:]):
        free = free_intervals(xa, xb, space.infl, space.ylo, space.yhi, P.min_len)
        nxt = []
        for y0, y1 in free:
            same = next((c for c in open_cells
                         if abs(c.y0 - y0) < EPS and abs(c.y1 - y1) < EPS), None)
            if same is not None:
                same.x1 = xb
                nxt.append(same)
            else:
                c = Cell(len(cells), xa, xb, y0, y1)
                cells.append(c)
                nxt.append(c)
        open_cells = nxt
    return cells, [x for x in xs if space.xlo + EPS < x < space.xhi - EPS]


def swath_positions(cell, d):
    """Evenly spread swath lines so the spacing is never larger than d."""
    n = max(1, math.ceil(cell.width / d - 1e-9))
    s = cell.width / n
    return [cell.x0 + (k + 0.5) * s for k in range(n)]


def snake(cell, xs, start_bottom, reverse):
    """Up the first line, down the next, and so on."""
    order = xs[::-1] if reverse else xs
    pts, up = [], start_bottom
    for x in order:
        a, b = (cell.y0, cell.y1) if up else (cell.y1, cell.y0)
        pts += [(x, a), (x, b)]
        up = not up
    return pts


def order_cells(cells, home, d):
    """Greedy nearest neighbour: always go to the cell whose cheapest entry
    corner is closest. Each cell can be entered from either side and either end."""
    pos, remaining, route = home, list(cells), []
    while remaining:
        best = None
        for c in remaining:
            xs = swath_positions(c, d)
            for rev in (False, True):
                for bottom in (True, False):
                    pts = snake(c, xs, bottom, rev)
                    dist = math.dist(pos, pts[0])
                    if best is None or dist < best[0]:
                        best = (dist, c, pts)
        _, c, pts = best
        route.append((c, pts))
        remaining.remove(c)
        pos = pts[-1]
    return route


def astar(space, start, goal, res):
    """8-connected grid A* on the configuration space."""
    xs, ys, blocked = space.grid(res)
    nx, ny = blocked.shape
    free_idx = np.argwhere(~blocked)
    free_xy = np.c_[xs[free_idx[:, 0]], ys[free_idx[:, 1]]]

    def snap(p):
        k = int(np.argmin(np.hypot(free_xy[:, 0] - p[0], free_xy[:, 1] - p[1])))
        return tuple(int(v) for v in free_idx[k])

    s, g = snap(start), snap(goal)

    def h(n):  # octile distance
        dx, dy = abs(n[0] - g[0]), abs(n[1] - g[1])
        return res * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))

    heap, gcost, came = [(h(s), 0.0, s)], {s: 0.0}, {}
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur == g:
            out = [cur]
            while cur in came:
                cur = came[cur]
                out.append(cur)
            return [(xs[i], ys[j]) for i, j in out[::-1]]
        if gc > gcost.get(cur, 1e18):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = cur[0] + di, cur[1] + dj
                if not (0 <= ni < nx and 0 <= nj < ny) or blocked[ni, nj]:
                    continue
                if di and dj and (blocked[cur[0] + di, cur[1]] or blocked[cur[0], cur[1] + dj]):
                    continue  # no corner cutting
                ng = gc + res * (math.sqrt(2) if di and dj else 1.0)
                if ng < gcost.get((ni, nj), 1e18):
                    gcost[(ni, nj)] = ng
                    came[(ni, nj)] = cur
                    heapq.heappush(heap, (ng + h((ni, nj)), ng, (ni, nj)))
    return None


def smooth(space, pts):
    """String pulling: skip ahead to the farthest point still visible in a straight line."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not space.line_free(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def connect(space, a, b, res):
    if math.dist(a, b) < 1e-9:
        return [a]
    if space.line_free(a, b):
        return [a, b]
    nodes = astar(space, a, b, res)
    if nodes is None:
        return None
    return smooth(space, [a] + nodes + [b])


def plan_mission(prob, P):
    space = Space(prob, P.robot_r)
    cells, crit = decompose(prob, P)
    route_all = order_cells(cells, prob.home, P.d)
    route, segments, skipped, pos = [], [], [], prob.home
    for c, pts in route_all:
        link = connect(space, pos, pts[0], P.grid_res)
        if link is None:
            skipped.append(c)
            continue
        if len(link) > 1:
            segments.append(("transit", link))
        segments.append(("coverage", pts))
        route.append((c, pts))
        pos = pts[-1]
    back = connect(space, pos, prob.home, P.grid_res)
    if back is not None and len(back) > 1:
        segments.append(("return", back))
    return Plan("x", cells, crit, route, segments, skipped)


def to_world(plan):
    """Undo the transpose so every coordinate is in the real field frame."""
    sw = lambda p: (p[1], p[0])
    cmap = {c.id: Cell(c.id, c.y0, c.y1, c.x0, c.x1) for c in plan.cells}
    return Plan("y", list(cmap.values()), plan.crit,
                [(cmap[c.id], [sw(p) for p in pts]) for c, pts in plan.route],
                [(k, [sw(p) for p in pts]) for k, pts in plan.segments],
                [cmap[c.id] for c in plan.skipped])


def plan_oriented(prob, P, sweep):
    if sweep == "x":
        return plan_mission(prob, P)
    return to_world(plan_mission(prob.transposed(), P))


def polyline(segments):
    out = []
    for _, pts in segments:
        for p in pts:
            if not out or math.dist(out[-1], p) > 1e-9:
                out.append(p)
    return out


def seg_length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))


def count_turns(poly, thresh_deg=20.0):
    heads = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(poly[:-1], poly[1:])]
    n = 0
    for h0, h1 in zip(heads[:-1], heads[1:]):
        dh = abs((h1 - h0 + math.pi) % (2 * math.pi) - math.pi)
        if dh > math.radians(thresh_deg):
            n += 1
    return n


def mission_metrics(plan, P):
    by = {"transit": 0.0, "coverage": 0.0, "return": 0.0}
    for kind, pts in plan.segments:
        by[kind] += seg_length(pts)
    total = sum(by.values())
    turns = count_turns(polyline(plan.segments))
    n_swaths = sum(len(pts) // 2 for _, pts in plan.route)
    return dict(by=by, total=total, turns=turns, swaths=n_swaths,
                cost=total + P.turn_cost * turns)


def resample(poly, step):
    a = np.asarray(poly, float)
    keep = [0] + [i for i in range(1, len(a)) if np.hypot(*(a[i] - a[i - 1])) > 1e-12]
    a = a[keep]
    if len(a) < 2:
        return a
    seg = np.hypot(*np.diff(a, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.c_[np.interp(t, s, a[:, 0]), np.interp(t, s, a[:, 1])]


def coverage_check(prob, P, plan, res=0.1):
    W, H, r = prob.W, prob.H, P.robot_r
    nx, ny = int(round(W / res)), int(round(H / res))
    xs, ys = (np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    reach = (X >= r) & (X <= W - r) & (Y >= r) & (Y <= H - r)
    for ox0, oy0, ox1, oy1 in inflate(prob.obstacles, r):
        reach &= ~((X > ox0) & (X < ox1) & (Y > oy0) & (Y < oy1))
    covered = np.zeros_like(reach)
    h = P.swath_w / 2.0
    k = int(math.ceil(h / res))
    for kind, pts in plan.segments:
        if kind != "coverage":
            continue
        for px, py in resample(pts, res):
            ix, iy = int(px / res), int(py / res)
            i0, i1 = max(0, ix - k), min(nx, ix + k + 1)
            j0, j1 = max(0, iy - k), min(ny, iy + k + 1)
            sub = (X[j0:j1, i0:i1] - px) ** 2 + (Y[j0:j1, i0:i1] - py) ** 2 <= h * h
            covered[j0:j1, i0:i1] |= sub
    raster = np.zeros(reach.shape, int)
    raster[reach & covered] = 1
    raster[reach & ~covered] = 2
    return dict(pct=100.0 * (reach & covered).sum() / reach.sum(),
                area=reach.sum() * res * res, raster=raster)


def moisture_truth(x, y):
    """Made-up soil moisture (percent). Wet patch, wetter patch, dry patch."""
    g = lambda cx, cy, s: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    return 100 * (0.30 + 0.22 * g(7, 5, 3.0) + 0.30 * g(23, 16, 4.0) - 0.14 * g(26, 4, 2.5)
                  + 0.03 * y / 20.0)


def idw(sxy, sv, gx, gy, k=8, power=2.0):
    """Inverse distance weighting from the k nearest samples."""
    P = np.c_[gx.ravel(), gy.ravel()]
    out = np.empty(len(P))
    for i in range(0, len(P), 800):
        chunk = P[i:i + 800]
        D = np.linalg.norm(chunk[:, None, :] - sxy[None, :, :], axis=2)
        idx = np.argpartition(D, k, axis=1)[:, :k]
        dk = np.take_along_axis(D, idx, axis=1)
        w = 1.0 / (dk ** power + 1e-9)
        out[i:i + 800] = (w * sv[idx]).sum(1) / w.sum(1)
    return out.reshape(gx.shape)

def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "text.color": OFF, "axes.labelcolor": MUTED, "axes.edgecolor": "#2B3136",
        "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "monospace",
        "font.size": 8.5, "hatch.color": "#566069",
    })


def draw_base(ax, prob, title, labels=True):
    ax.add_patch(Rectangle((0, 0), prob.W, prob.H, fill=False, ec=OFF, lw=1.3))
    for (x0, y0, x1, y1), name in zip(prob.obstacles, prob.names):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=OBST, ec=MUTED, lw=1, hatch="///"))
        if labels:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=7,
                    color=OFF, bbox=dict(fc=PANEL, ec="none", pad=1.2, alpha=0.85), zorder=7)
    ax.plot(*prob.home, "s", color=LIME, ms=7, zorder=8)
    ax.text(prob.home[0] + 0.7, prob.home[1] + 1.0, "HOME", color=LIME, fontsize=7.5, zorder=8)
    ax.set_xlim(-1, prob.W + 1)
    ax.set_ylim(-2.2, prob.H + 1.4)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", color=LIME, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)


def fig_steps(prob, P, plan, mets, cov, path):
    fig, axs = plt.subplots(2, 3, figsize=(19, 11.8))
    ax = axs.ravel()
    r, W, H = P.robot_r, prob.W, prob.H
    along = "x" if plan.sweep_axis == "x" else "y"

    # 1 inputs
    draw_base(ax[0], prob, "1. INPUTS")
    ax[0].text(W / 2, -1.2, f"field {W:g} m x {H:g} m   |   robot radius {r} m   |   footprint {P.swath_w} m",
               ha="center", color=MUTED, fontsize=8)

    # 2 configuration space
    draw_base(ax[1], prob, "2. CONFIGURATION SPACE (robot becomes a point)", labels=False)
    ax[1].add_patch(Rectangle((r, r), W - 2 * r, H - 2 * r, fill=False, ls="--", ec=LIME, lw=1.2))
    for x0, y0, x1, y1 in inflate(prob.obstacles, r):
        ax[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=AMBER, alpha=0.22, ec=AMBER, ls="--", lw=1.2))
    ax[1].text(W / 2, -1.2, "lime: robot-centre limits (headland)   amber: obstacles inflated by r",
               ha="center", color=MUTED, fontsize=8)

    # 3 decomposition
    draw_base(ax[2], prob, f"3. CELL DECOMPOSITION: {len(plan.cells)} cells (sweep along {along})", labels=False)
    for c in plan.cells:
        col = CELL_COLORS[c.id % len(CELL_COLORS)]
        ax[2].add_patch(Rectangle((c.x0, c.y0), c.x1 - c.x0, c.y1 - c.y0, fc=col, alpha=0.30, ec=col, lw=1.3))
        ax[2].text((c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2, str(c.id + 1), ha="center", va="center",
                   fontsize=10, fontweight="bold", color=OFF)
    for v in plan.crit:
        if plan.sweep_axis == "x":
            ax[2].plot([v, v], [r, H - r], ls=":", color=OFF, lw=0.9)
        else:
            ax[2].plot([r, W - r], [v, v], ls=":", color=OFF, lw=0.9)
    if plan.sweep_axis == "x":
        ax[2].annotate("", xy=(W - r, H + 0.7), xytext=(r, H + 0.7), arrowprops=dict(arrowstyle="->", color=AMBER))
        ax[2].text(W / 2, H + 0.85, "sweep line direction", color=AMBER, ha="center", fontsize=7.5)
    else:
        ax[2].annotate("", xy=(W + 0.7, H - r), xytext=(W + 0.7, r), arrowprops=dict(arrowstyle="->", color=AMBER))
    ax[2].text(W / 2, -1.2, "dotted: events where free space changes (split / merge)",
               ha="center", color=MUTED, fontsize=8)

    # 4 swaths
    draw_base(ax[3], prob, f"4. SWATHS: spacing d = {P.d:.2f} m, {mets['swaths']} lines", labels=False)
    for i, (c, pts) in enumerate(plan.route):
        col = CELL_COLORS[c.id % len(CELL_COLORS)]
        ax[3].add_patch(Rectangle((c.x0, c.y0), c.x1 - c.x0, c.y1 - c.y0, fc=col, alpha=0.10, ec=col, lw=0.8))
        for k in range(0, len(pts), 2):
            (xa, ya), (xb, yb) = pts[k], pts[k + 1]
            ax[3].plot([xa, xb], [ya, yb], color=col, lw=1.3)
    ax[3].text(W / 2, -1.2, "d = footprint x (1 - overlap), evenly spread so spacing never exceeds d",
               ha="center", color=MUTED, fontsize=8)

    # 5 mission
    draw_base(ax[4], prob, "5. MISSION: snake, A* links, return home", labels=False)
    for kind, pts in plan.segments:
        xs_, ys_ = zip(*pts)
        if kind == "coverage":
            ax[4].plot(xs_, ys_, color=LIME, lw=1.1)
        elif kind == "transit":
            ax[4].plot(xs_, ys_, color=AMBER, lw=1.7, ls="--")
            ax[4].annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.2))
        else:
            ax[4].plot(xs_, ys_, color=OFF, lw=2.0)
            ax[4].annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=OFF, lw=1.4))
    for i, (c, pts) in enumerate(plan.route):
        ax[4].text(pts[0][0], pts[0][1], str(i + 1), fontsize=7, fontweight="bold", color=BG, ha="center",
                   va="center", zorder=9, bbox=dict(boxstyle="circle,pad=0.25", fc=AMBER, ec="none"))
    from matplotlib.lines import Line2D
    ax[4].legend(handles=[Line2D([], [], color=LIME, lw=1.5, label="coverage swaths"),
                          Line2D([], [], color=AMBER, lw=1.5, ls="--", label="A* transit"),
                          Line2D([], [], color=OFF, lw=2, label="return home"),
                          Line2D([], [], marker="o", color=AMBER, lw=0, label="visit order")],
                 loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7.5,
                 facecolor=PANEL, edgecolor="#2B3136", labelcolor=OFF)

    # 6 verification
    draw_base(ax[5], prob, f"6. VERIFY: {cov['pct']:.1f}% of reachable area covered", labels=False)
    ax[5].imshow(cov["raster"], origin="lower", extent=(0, W, 0, H),
                 cmap=ListedColormap([PANEL, "#7FA000", RED]), vmin=0, vmax=2, interpolation="nearest")
    for (x0, y0, x1, y1) in prob.obstacles:
        ax[5].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=OBST, ec=MUTED, lw=1, hatch="///"))
    ax[5].text(W / 2, -1.2, "green: swept by sensor footprint   red: missed   gray: obstacle",
               ha="center", color=MUTED, fontsize=8)

    b = mets["by"]
    ideal = cov["area"] / P.d
    fig.text(0.5, 0.035,
             f"cells {len(plan.cells)}   swaths {mets['swaths']}   coverage path {b['coverage']:.0f} m   "
             f"transit {b['transit']:.0f} m   return {b['return']:.0f} m   turns {mets['turns']}   "
             f"ideal length (area/d) {ideal:.0f} m   efficiency {100 * ideal / b['coverage']:.0f}%",
             ha="center", color=OFF, fontsize=10)
    fig.suptitle("BOUSTROPHEDON COVERAGE PLANNING, STEP BY STEP", color=OFF, fontsize=14,
                 fontweight="bold", x=0.5, y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=130)


def fig_orientation(prob, P, results, chosen, path):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, (sweep, plan, m) in zip(axs, results):
        tag = "  [SELECTED]" if sweep == chosen else ""
        swath_dir = "vertical" if sweep == "x" else "horizontal"
        draw_base(ax, prob, f"sweep along {sweep}: {swath_dir} swaths{tag}", labels=False)
        for kind, pts in plan.segments:
            xs_, ys_ = zip(*pts)
            col, ls, lw = {"coverage": (LIME, "-", 1.0), "transit": (AMBER, "--", 1.6), "return": (OFF, "-", 1.8)}[kind]
            ax.plot(xs_, ys_, color=col, ls=ls, lw=lw)
        ax.text(prob.W / 2, -1.2,
                f"cells {len(plan.cells)}  length {m['total']:.0f} m  turns {m['turns']}  cost {m['cost']:.0f}",
                ha="center", color=OFF, fontsize=9.5)
    fig.suptitle(f"Same field, same coverage, different turns. cost = length + {P.turn_cost:g} m per turn",
                 color=OFF, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)


def fig_heatmap(prob, P, plan, rng, path):
    W, H = prob.W, prob.H
    samples = np.vstack([resample(pts, 1.0) for kind, pts in plan.segments if kind == "coverage"])
    meas = moisture_truth(samples[:, 0], samples[:, 1]) + rng.normal(0, 1.0, len(samples))
    res = 0.4
    gx, gy = np.meshgrid(np.arange(res / 2, W, res), np.arange(res / 2, H, res))
    truth = moisture_truth(gx, gy)
    est = idw(samples, meas, gx, gy)
    mask = np.zeros(gx.shape, bool)
    for x0, y0, x1, y1 in prob.obstacles:
        mask |= (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1)
    rmse = float(np.sqrt(np.mean((est[~mask] - truth[~mask]) ** 2)))
    cmap = LinearSegmentedColormap.from_list("moist", [AMBER, LIME, "#2EC4B6"])
    cmap.set_bad("#2a2f35")
    vmin, vmax = truth.min(), truth.max()

    fig, axs = plt.subplots(1, 3, figsize=(19, 6.4))
    draw_base(axs[0], prob, "GROUND TRUTH (unknown to the rover)", labels=False)
    axs[0].imshow(np.ma.array(truth, mask=mask), origin="lower", extent=(0, W, 0, H), cmap=cmap, vmin=vmin, vmax=vmax)
    draw_base(axs[1], prob, f"SAMPLES TAKEN ALONG THE PATH ({len(samples)} points, 1 m apart)", labels=False)
    sc = axs[1].scatter(samples[:, 0], samples[:, 1], c=meas, s=9, cmap=cmap, vmin=vmin, vmax=vmax, zorder=5)
    draw_base(axs[2], prob, f"IDW HEATMAP SENT TO THE TWIN (RMSE {rmse:.1f} % moisture)", labels=False)
    axs[2].imshow(np.ma.array(est, mask=mask), origin="lower", extent=(0, W, 0, H), cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(sc, ax=axs, fraction=0.018, pad=0.02)
    cb.set_label("soil moisture (%)", color=MUTED)
    cb.ax.tick_params(colors=MUTED)
    fig.suptitle("Why coverage matters: the heatmap only exists where the rover actually sampled",
                 color=OFF, fontsize=12.5, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return rmse, len(samples)


def default_problem(P):
    obs = [(6, 10, 9, 13), (14, 3, 18, 7), (16, 14, 18, 17), (21, 11, 25, 15)]
    names = ["well", "tree cluster", "post", "pump shed"]
    return Problem(30.0, 20.0, obs, names, (P.robot_r, P.robot_r))


def random_problem(P, n, seed):
    rng = np.random.default_rng(seed)
    W, H, obs, tries = 30.0, 20.0, [], 0
    while len(obs) < n and tries < 1000:
        tries += 1
        w, h = rng.uniform(1.5, 4.0, 2)
        x0, y0 = rng.uniform(2.0, W - 2.0 - w), rng.uniform(2.0, H - 2.0 - h)
        if x0 < 4 and y0 < 4:
            continue  # keep the home corner clear
        a = (x0 - 1.0, y0 - 1.0, x0 + w + 1.0, y0 + h + 1.0)
        if any(a[0] < o[2] and a[2] > o[0] and a[1] < o[3] and a[3] > o[1] for o in obs):
            continue  # keep at least 2 m between obstacles
        obs.append((x0, y0, x0 + w, y0 + h))
    return Problem(W, H, obs, [f"obs {i + 1}" for i in range(len(obs))], (P.robot_r, P.robot_r))


def write_waypoints(plan, path):
    poly, kinds = [], []
    for kind, pts in plan.segments:
        for p in pts:
            if not poly or math.dist(poly[-1], p) > 1e-9:
                poly.append(p)
                kinds.append(kind)
    with open(path, "w") as f:
        f.write("x_m,y_m,heading_deg,phase\n")
        for i, (p, k) in enumerate(zip(poly, kinds)):
            q = poly[min(i + 1, len(poly) - 1)]
            hd = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) if q != p else 0.0
            f.write(f"{p[0]:.3f},{p[1]:.3f},{hd:.1f},{k}\n")
    return len(poly)



def main():
    ap = argparse.ArgumentParser(description="Boustrophedon coverage planning demo")
    ap.add_argument("--no-show", action="store_true", help="save figures only")
    ap.add_argument("--random", type=int, default=0, help="N random obstacles instead of the default field")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sweep", choices=["auto", "x", "y"], default="auto")
    ap.add_argument("--swath", type=float, default=1.5, help="sensor footprint width (m)")
    ap.add_argument("--outdir", default="coverage_demo_out")
    args = ap.parse_args()

    P = Params(swath_w=args.swath)
    prob = random_problem(P, args.random, args.seed) if args.random else default_problem(P)
    os.makedirs(args.outdir, exist_ok=True)
    style()

    print("=" * 70)
    print("STEP 1  INPUTS")
    print(f"  field {prob.W:g} x {prob.H:g} m, {len(prob.obstacles)} obstacles, home {prob.home}")
    print(f"  footprint {P.swath_w} m, overlap {P.overlap:.0%}  ->  swath spacing d = {P.d:.2f} m")
    print("STEP 2  CONFIGURATION SPACE")
    print(f"  usable region shrinks by {P.robot_r} m on every side, obstacles grow by {P.robot_r} m")
    print("STEP 3  to 5  DECOMPOSE, SWATH, ORDER, CONNECT (both sweep directions)")

    results = []
    for sweep in ("x", "y"):
        plan = plan_oriented(prob, P, sweep)
        m = mission_metrics(plan, P)
        results.append((sweep, plan, m))
        print(f"  sweep {sweep}: {len(plan.cells):2d} cells, {m['swaths']:3d} swaths, "
              f"length {m['total']:6.1f} m, turns {m['turns']:3d}, cost {m['cost']:6.1f}")
    if args.sweep == "auto":
        chosen = min(results, key=lambda r: r[2]["cost"])[0]
    else:
        chosen = args.sweep
    _, plan, mets = next(r for r in results if r[0] == chosen)
    print(f"  selected sweep direction: {chosen}")
    if plan.skipped:
        print(f"  WARNING: {len(plan.skipped)} cell(s) unreachable and skipped")

    print("STEP 6  VERIFY")
    cov = coverage_check(prob, P, plan)
    ideal = cov["area"] / P.d
    print(f"  reachable area {cov['area']:.1f} m2, covered {cov['pct']:.2f} %")
    print(f"  coverage path {mets['by']['coverage']:.1f} m vs ideal (area / d) {ideal:.1f} m "
          f"-> efficiency {100 * ideal / mets['by']['coverage']:.0f} %")
    print(f"  transit {mets['by']['transit']:.1f} m, return {mets['by']['return']:.1f} m, turns {mets['turns']}")

    n_wp = write_waypoints(plan, os.path.join(args.outdir, "waypoints.csv"))
    print(f"  wrote {n_wp} waypoints to {args.outdir}/waypoints.csv (x, y, heading, phase)")

    fig_steps(prob, P, plan, mets, cov, os.path.join(args.outdir, "1_step_by_step.png"))
    fig_orientation(prob, P, results, chosen, os.path.join(args.outdir, "2_sweep_orientation.png"))
    rmse, n = fig_heatmap(prob, P, plan, np.random.default_rng(args.seed),
                          os.path.join(args.outdir, "3_heatmap.png"))
    print("STEP 7  HEATMAP")
    print(f"  {n} samples along the path, IDW reconstruction RMSE {rmse:.2f} % moisture")
    print("=" * 70)
    print(f"figures saved in {args.outdir}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
