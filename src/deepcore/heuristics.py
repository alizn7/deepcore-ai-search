"""
heuristics.py
Student heuristic starter file for DeepCore:
  - h_naive_manhattan : provided baseline
  - h1_basic          : required student implementation
  - h2_advanced       : required student implementation
"""

from itertools import combinations, permutations
from typing import Dict, FrozenSet, List, Tuple
from collections import deque
import heapq
import weakref

from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos

INF = float("inf")

# Cache wall-aware shortest path distances per environment instance
_WALL_DIST_CACHE: "weakref.WeakKeyDictionary[DeepCoreEnv, Dict[Tuple[Pos, Pos], int]]" = weakref.WeakKeyDictionary()
_H2_CACHE: "weakref.WeakKeyDictionary[DeepCoreEnv, _H2Tables]" = weakref.WeakKeyDictionary()


def _get_wall_aware_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """Precomputes all-pairs shortest path steps ignoring dynamic obstacles (only walls # block)."""
    cached = _WALL_DIST_CACHE.get(env)
    if cached is not None:
        return cached

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]
    dist_map: Dict[Tuple[Pos, Pos], int] = {}

    for src in walkable:
        q = deque([(src, 0)])
        visited = {src}
        while q:
            curr, d = q.popleft()
            dist_map[(src, curr)] = d
            cr, cc = curr
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if env.in_bounds(nxt) and nxt not in env.walls and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))

    _WALL_DIST_CACHE[env] = dist_map
    return dist_map


def _prim_mst_cost(nodes: List[Pos], dist_fn) -> float:
    """Computes Minimum Spanning Tree weight over `nodes` using `dist_fn(u, v)`."""
    if len(nodes) <= 1:
        return 0.0
    in_mst = {nodes[0]}
    total_weight = 0.0

    while len(in_mst) < len(nodes):
        best_edge = float("inf")
        best_node = None
        for u in in_mst:
            for v in nodes:
                if v not in in_mst:
                    w = dist_fn(u, v)
                    if w < best_edge:
                        best_edge = w
                        best_node = v
        if best_node is None:
            break
        in_mst.add(best_node)
        total_weight += best_edge

    return total_weight


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Baseline Naive Heuristic: Manhattan distance to nearest waiting miner (or Elevator if none).
    """
    if env.is_goal(state):
        return 0.0
    r, c = state.bot_pos
    if state.miners:
        return float(min(abs(r - mr) + abs(c - mc) for mr, mc in state.miners))
    er, ec = env.elevator_pos
    return float(abs(r - er) + abs(c - ec))


def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:
    if env.is_goal(state):
        return 0.0
    elevator = env.elevator_pos
    nodes = [state.bot_pos, elevator, *state.miners]
    mst = _prim_mst_cost(nodes, _manhattan)
    payload = 2 * state.carried * _manhattan(state.bot_pos, elevator)
    payload += sum(2 * _manhattan(m, elevator) for m in state.miners)
    return float(mst + payload)


def _is_corner_deadlock(pos: Pos, env: DeepCoreEnv) -> bool:
    """Returns True if a boulder at `pos` is wedged against perpendicular rock walls (#)."""
    r, c = pos
    n_wall = (r - 1, c) in env.walls
    s_wall = (r + 1, c) in env.walls
    e_wall = (r, c + 1) in env.walls
    w_wall = (r, c - 1) in env.walls
    return (n_wall or s_wall) and (e_wall or w_wall)


# =============================================================================
# H2 precomputation
# =============================================================================
class _H2Tables:
    def __init__(self, env: DeepCoreEnv):
        self.env = env
        self.cells: List[Pos] = [
            (r, c) for r in range(env.rows) for c in range(env.cols) if (r, c) not in env.walls
        ]
        self.cell_set = set(self.cells)
        self.weight: Dict[Pos, int] = {
            p: (env.COST_SLUDGE if p in env.sludge else env.COST_CLEAR_FLOOR) for p in self.cells
        }
        self.neighbors: Dict[Pos, List[Pos]] = {
            p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRECTION_DELTAS.values()
                if (p[0] + dr, p[1] + dc) in self.cell_set]
            for p in self.cells
        }
        self.steps = _get_wall_aware_distances(env)
        self.terrain = self._all_pairs_terrain()
        self.push_cost: Dict[Pos, Dict[Pos, int]] = {
            c: self._boulder_to_chasm_costs(c) for c in env.initial_chasms
        }
        self._labels_cache: Dict[Tuple[FrozenSet[Pos], FrozenSet[Pos]], list] = {}
        self._feasible_cache: Dict[tuple, List[Tuple[int, Tuple[Pos, ...]]]] = {}
        self._tour_cache: Dict[tuple, float] = {}
        self._assign_cache: Dict[tuple, List[Tuple[float, FrozenSet[Pos]]]] = {}

    def _all_pairs_terrain(self) -> Dict[Tuple[Pos, Pos], float]:
        directed: Dict[Tuple[Pos, Pos], int] = {}
        for src in self.cells:
            dist = {src: 0}
            pq = [(0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v in self.neighbors[u]:
                    nd = d + self.weight[v]
                    if nd < dist.get(v, INF):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            for dst, d in dist.items():
                directed[(src, dst)] = d
        sym: Dict[Tuple[Pos, Pos], float] = {}
        for (a, b), d in directed.items():
            sym[(a, b)] = min(d, directed.get((b, a), INF))
        return sym

    def _boulder_to_chasm_costs(self, chasm: Pos) -> Dict[Pos, int]:
        env = self.env
        dist = {chasm: 0}
        pq = [(0, chasm)]
        while pq:
            d, q = heapq.heappop(pq)
            if d > dist[q]:
                continue
            for dr, dc in DIRECTION_DELTAS.values():
                p = (q[0] - dr, q[1] - dc)
                robot = (p[0] - dr, p[1] - dc)
                if (
                    p not in self.cell_set
                    or robot not in self.cell_set
                    or p == env.elevator_pos
                    or p == chasm
                ):
                    continue
                nd = d + env.COST_PUSH_BASE - self.weight[p]
                if nd < dist.get(p, INF):
                    dist[p] = nd
                    heapq.heappush(pq, (nd, p))
        del dist[chasm]
        return dist

    def opening_options(self, rubble: FrozenSet[Pos], chasms: FrozenSet[Pos]) -> list:
        key = (rubble, chasms)
        options = self._labels_cache.get(key)
        if options is not None:
            return options
        barriers = list(rubble | chasms)
        options = []
        for size in range(len(barriers) + 1):
            for subset in combinations(barriers, size):
                opened = set(subset)
                blocked = (rubble | chasms) - opened
                labels: Dict[Pos, int] = {}
                label = 0
                for start in self.cells:
                    if start in labels or start in blocked:
                        continue
                    labels[start] = label
                    q = deque([start])
                    while q:
                        u = q.popleft()
                        for v in self.neighbors[u]:
                            if v not in labels and v not in blocked:
                                labels[v] = label
                                q.append(v)
                    label += 1
                drills = sum(1 for p in subset if p in rubble)
                bridge_set = frozenset(p for p in subset if p in chasms)
                options.append((drills * self.env.COST_DRILL, bridge_set, labels))
        self._labels_cache[key] = options
        return options

    def feasible_openings(
        self, bot: Pos, miners: FrozenSet[Pos], rubble: FrozenSet[Pos], chasms: FrozenSet[Pos]
    ) -> List[Tuple[int, Tuple[Pos, ...]]]:
        key = (bot, miners, rubble, chasms)
        cached = self._feasible_cache.get(key)
        if cached is not None:
            return cached
        targets = [self.env.elevator_pos, *miners]
        feasible = []
        for drill_cost, bridge_set, labels in self.opening_options(rubble, chasms):
            bot_label = labels.get(bot)
            if bot_label is not None and all(labels.get(t) == bot_label for t in targets):
                feasible.append((drill_cost, tuple(sorted(bridge_set))))
        self._feasible_cache[key] = feasible
        return feasible

    def assignments(
        self, boulders: FrozenSet[Pos], bridges: Tuple[Pos, ...]
    ) -> List[Tuple[float, FrozenSet[Pos]]]:
        key = (boulders, bridges)
        cached = self._assign_cache.get(key)
        if cached is not None:
            return cached
        usable = sorted(b for b in boulders if not _is_corner_deadlock(b, self.env))
        options = []
        for chosen in permutations(usable, len(bridges)):
            push = 0.0
            for chasm, boulder in zip(bridges, chosen):
                push += self.push_cost[chasm].get(boulder, INF)
            if push < INF:
                options.append((push, frozenset(chosen)))
        options.sort(key=lambda item: item[0])
        self._assign_cache[key] = options
        return options

    def tour_cost(self, bot: Pos, miners: FrozenSet[Pos], carried: int, extra: FrozenSet[Pos]) -> float:
        key = (bot, miners, carried, extra)
        cached = self._tour_cache.get(key)
        if cached is not None:
            return cached
        steps, terrain = self.steps, self.terrain
        elevator = self.env.elevator_pos

        def step_dist(a: Pos, b: Pos) -> float:
            return steps.get((a, b), INF)

        def terrain_dist(a: Pos, b: Pos) -> float:
            return terrain.get((a, b), INF)

        node_sets = [[bot, elevator, *miners]]
        if extra:
            node_sets.append([bot, elevator, *miners, *extra])
        step_lb = max(_prim_mst_cost(nodes, step_dist) for nodes in node_sets)
        if step_lb == INF:
            value = INF
        else:
            base = max(step_lb, max(_prim_mst_cost(nodes, terrain_dist) for nodes in node_sets))
            payload = 2 * carried * step_lb + sum(2 * step_dist(m, elevator) for m in miners)
            value = float(base + payload)
        self._tour_cache[key] = value
        return value


def _h2_tables(env: DeepCoreEnv) -> _H2Tables:
    tables = _H2_CACHE.get(env)
    if tables is None:
        tables = _H2Tables(env)
        _H2_CACHE[env] = tables
    return tables


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    if env.is_goal(state):
        return 0.0
    tables = _h2_tables(env)
    bot, miners, carried = state.bot_pos, state.miners, state.carried
    tour_v = tables.tour_cost(bot, miners, carried, frozenset())
    if tour_v == INF:
        return INF
    best = INF
    for drill_cost, bridges in tables.feasible_openings(bot, miners, state.rubble, state.chasms):
        lower = drill_cost + tour_v
        if lower >= best:
            continue
        if not bridges:
            best = lower
            continue
        for push, chosen in tables.assignments(state.boulders, bridges):
            if lower + push >= best:
                break
            total = drill_cost + push + tables.tour_cost(bot, miners, carried, chosen)
            if total < best:
                best = total
    return best
