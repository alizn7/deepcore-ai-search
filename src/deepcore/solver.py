"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""

from collections import deque
from itertools import combinations
import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
    parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# Shared helpers
# =============================================================================
Heuristic = Callable[[DeepCoreState, DeepCoreEnv], float]
ParentMap = Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]]

LIMIT_MESSAGE = "Expansion limit reached."
NO_SOLUTION_MESSAGE = "No solution found."


def _path_cost(env: DeepCoreEnv, actions: List[str]) -> int:
    state = env.get_initial_state()
    total = 0
    for act in actions:
        state, step_cost = env.transition(state, act)
        total += step_cost
    return total


def _finish(metrics: SearchMetrics, actions: List[str], cost: int, message: str = "Solved") -> SearchMetrics:
    metrics.solved = True
    metrics.solution_actions = actions
    metrics.total_cost = cost
    metrics.status_message = message
    return metrics


def _limit_hit(metrics: SearchMetrics, max_expansions: Optional[int]) -> bool:
    return max_expansions is not None and metrics.nodes_expanded >= max_expansions


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS)
# =============================================================================
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()
    stack: List[DeepCoreState] = [start]
    parent: ParentMap = {start: (None, None)}
    g: Dict[DeepCoreState, int] = {start: 0}
    explored: Set[DeepCoreState] = set()

    while stack:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
        metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(explored))
        state = stack.pop()
        if state in explored:
            continue
        if env.is_goal(state):
            return _finish(metrics, _reconstruct_path(parent, state), g[state])
        if _limit_hit(metrics, max_expansions):
            metrics.status_message = LIMIT_MESSAGE
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        successors = env.get_successors(state)
        metrics.nodes_generated += len(successors)
        for action, child, step_cost in reversed(successors):
            if child not in explored:
                parent[child] = (state, action.value)
                g[child] = g[state] + step_cost
                stack.append(child)

    metrics.max_memory_size = max(metrics.max_memory_size, len(explored))
    metrics.status_message = NO_SOLUTION_MESSAGE
    return metrics


def _depth_limited_search(
    env: DeepCoreEnv,
    limit: int,
    metrics: SearchMetrics,
    max_expansions: Optional[int],
) -> Tuple[str, Optional[List[str]], Optional[int]]:
    start = env.get_initial_state()
    best_depth: Dict[DeepCoreState, int] = {start: 0}
    on_path: Set[DeepCoreState] = {start}
    path_actions: List[str] = []
    path_costs: List[int] = []
    stack: List[list] = []
    cutoff = False

    if env.is_goal(start):
        return "found", [], 0
    if limit == 0:
        return "cutoff", None, None

    metrics.nodes_expanded += 1
    metrics.explored_order.append(start.bot_pos)
    succ = env.get_successors(start)
    metrics.nodes_generated += len(succ)
    stack.append([start, 0, succ, 0])

    while stack:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
        metrics.max_memory_size = max(metrics.max_memory_size, len(best_depth) + len(stack))
        frame = stack[-1]
        state, depth, successors, idx = frame
        if idx >= len(successors):
            stack.pop()
            on_path.discard(state)
            if path_actions:
                path_actions.pop()
                path_costs.pop()
            continue
        frame[3] = idx + 1
        action, child, step_cost = successors[idx]
        child_depth = depth + 1
        if child in on_path or best_depth.get(child, child_depth + 1) <= child_depth:
            continue
        best_depth[child] = child_depth

        if env.is_goal(child):
            actions = path_actions + [action.value]
            return "found", actions, sum(path_costs) + step_cost
        if child_depth >= limit:
            cutoff = True
            continue
        if _limit_hit(metrics, max_expansions):
            return "limit", None, None

        metrics.nodes_expanded += 1
        metrics.explored_order.append(child.bot_pos)
        child_succ = env.get_successors(child)
        metrics.nodes_generated += len(child_succ)
        on_path.add(child)
        path_actions.append(action.value)
        path_costs.append(step_cost)
        stack.append([child, child_depth, child_succ, 0])

    return ("cutoff" if cutoff else "failure"), None, None


def iterative_deepening_search(
    env: DeepCoreEnv,
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="IDS")
    limit = 0
    while max_depth is None or limit <= max_depth:
        status, actions, cost = _depth_limited_search(env, limit, metrics, max_expansions)
        if status == "found":
            return _finish(metrics, actions, cost, f"Solved at depth limit {limit}")
        if status == "limit":
            metrics.status_message = LIMIT_MESSAGE
            return metrics
        if status == "failure":
            metrics.status_message = NO_SOLUTION_MESSAGE
            return metrics
        limit += 1
    metrics.status_message = LIMIT_MESSAGE
    return metrics


# =============================================================================
# 3. Bidirectional Search
# =============================================================================
def _canonical_goal_states(env: DeepCoreEnv) -> List[DeepCoreState]:
    boulders = sorted(env.initial_boulders)
    chasms = sorted(env.initial_chasms)
    rubble = sorted(env.initial_rubble)
    roots: List[DeepCoreState] = []
    for used in range(min(len(boulders), len(chasms)) + 1):
        for spent in combinations(boulders, used):
            for filled in combinations(chasms, used):
                for drilled_count in range(len(rubble) + 1):
                    for drilled in combinations(rubble, drilled_count):
                        roots.append(DeepCoreState(
                            bot_pos=env.elevator_pos,
                            carried=env.total_miners,
                            boulders=frozenset(boulders) - frozenset(spent),
                            chasms=frozenset(chasms) - frozenset(filled),
                            rubble=frozenset(rubble) - frozenset(drilled),
                            miners=frozenset(),
                        ))
    return roots


def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()
    if env.is_goal(start):
        return _finish(metrics, [], 0)

    fwd_parent: ParentMap = {start: (None, None)}
    fwd_depth: Dict[DeepCoreState, int] = {start: 0}
    bwd_next: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {}
    bwd_depth: Dict[DeepCoreState, int] = {}
    fwd_layer: List[DeepCoreState] = [start]
    bwd_layer: List[DeepCoreState] = []
    boulder_budget = len(start.boulders) + len(env.initial_chasms) - len(start.chasms)
    for goal in [*env.get_goal_states(), *_canonical_goal_states(env)]:
        # free boulders + bridged chasms is invariant; other roots are unreachable
        bridged = len(env.initial_chasms) - len(goal.chasms)
        if len(goal.boulders) + bridged != boulder_budget:
            continue
        if goal not in bwd_next:
            bwd_next[goal] = (None, None)
            bwd_depth[goal] = 0
            bwd_layer.append(goal)

    def join(meet: DeepCoreState) -> List[str]:
        actions = _reconstruct_path(fwd_parent, meet)
        curr = meet
        while True:
            nxt, act = bwd_next[curr]
            if nxt is None:
                break
            actions.append(act)
            curr = nxt
        return actions

    def track() -> None:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(fwd_layer) + len(bwd_layer))
        metrics.max_memory_size = max(metrics.max_memory_size, len(fwd_parent) + len(bwd_next))

    if start in bwd_next:
        actions = join(start)
        return _finish(metrics, actions, _path_cost(env, actions))

    while fwd_layer:
        track()
        best_len = math.inf
        best_actions: Optional[List[str]] = None

        if bwd_layer and len(bwd_layer) < len(fwd_layer):
            next_layer: List[DeepCoreState] = []
            for state in bwd_layer:
                metrics.nodes_expanded += 1
                metrics.explored_order.append(state.bot_pos)
                for action, prev, _ in env.get_predecessors(state):
                    metrics.nodes_generated += 1
                    if prev in bwd_next:
                        continue
                    bwd_next[prev] = (state, action.value)
                    bwd_depth[prev] = bwd_depth[state] + 1
                    next_layer.append(prev)
                    if prev in fwd_depth:
                        length = fwd_depth[prev] + bwd_depth[prev]
                        if length < best_len:
                            best_len, best_actions = length, join(prev)
            bwd_layer = next_layer
        else:
            next_layer = []
            for state in fwd_layer:
                metrics.nodes_expanded += 1
                metrics.explored_order.append(state.bot_pos)
                for action, child, _ in env.get_successors(state):
                    metrics.nodes_generated += 1
                    if child in fwd_parent:
                        continue
                    fwd_parent[child] = (state, action.value)
                    fwd_depth[child] = fwd_depth[state] + 1
                    next_layer.append(child)
                    if child in bwd_next:
                        length = fwd_depth[child] + bwd_depth[child]
                        if length < best_len:
                            best_len, best_actions = length, join(child)
                    elif env.is_goal(child) and fwd_depth[child] < best_len:
                        best_len, best_actions = fwd_depth[child], _reconstruct_path(fwd_parent, child)
            fwd_layer = next_layer

        track()
        if best_actions is not None:
            return _finish(metrics, best_actions, _path_cost(env, best_actions))

    metrics.status_message = NO_SOLUTION_MESSAGE
    return metrics


# =============================================================================
# Generic best-first graph search (UCS, GBFS, A*, Weighted A*)
# =============================================================================
def _best_first_search(
    env: DeepCoreEnv,
    name: str,
    priority: Callable[[int, float], Tuple[float, ...]],
    heuristic: Optional[Heuristic],
    reopen: bool,
) -> SearchMetrics:
    metrics = SearchMetrics(algorithm_name=name)
    start = env.get_initial_state()
    h_cache: Dict[DeepCoreState, float] = {}

    def h_of(state: DeepCoreState) -> float:
        if heuristic is None:
            return 0.0
        value = h_cache.get(state)
        if value is None:
            value = heuristic(state, env)
            h_cache[state] = value
        return value

    h0 = h_of(start)
    if h0 == math.inf:
        metrics.status_message = "Initial state is a proven deadlock."
        return metrics

    counter = 0
    best_g: Dict[DeepCoreState, int] = {start: 0}
    parent: ParentMap = {start: (None, None)}
    closed: Set[DeepCoreState] = set()
    heap = [(*priority(0, h0), counter, 0, start)]

    while heap:
        open_count = len(best_g) - len(closed)
        metrics.max_frontier_size = max(metrics.max_frontier_size, open_count)
        metrics.max_memory_size = max(metrics.max_memory_size, len(best_g))

        *_, g, state = heapq.heappop(heap)
        if g != best_g[state] or state in closed:
            continue
        if env.is_goal(state):
            return _finish(metrics, _reconstruct_path(parent, state), g)

        closed.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, child, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_g = g + step_cost
            old_g = best_g.get(child)
            if old_g is not None:
                if new_g >= old_g:
                    continue
                if child in closed:
                    if not reopen:
                        continue
                    closed.discard(child)
            h = h_of(child)
            if h == math.inf:
                continue
            best_g[child] = new_g
            parent[child] = (state, action.value)
            counter += 1
            heapq.heappush(heap, (*priority(new_g, h), counter, new_g, child))

    metrics.status_message = NO_SOLUTION_MESSAGE
    return metrics


# =============================================================================
# 4. Uniform Cost Search (UCS)
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    return _best_first_search(env, "UCS", lambda g, h: (g,), None, reopen=False)


# =============================================================================
# 5. Greedy Best-First Search (GBFS)
# =============================================================================
def greedy_best_first_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    return _best_first_search(env, "GBFS", lambda g, h: (h,), heuristic, reopen=False)


# =============================================================================
# 6. A* Search
# =============================================================================
def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    return _best_first_search(env, "A*", lambda g, h: (g + h,), heuristic, reopen=True)


# =============================================================================
# BONUS 1: Weighted A* Search
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    if weight < 1:
        raise ValueError("Weighted A* needs weight >= 1.")
    return _best_first_search(
        env, f"WA*(W={weight:g})", lambda g, h: (g + weight * h,), heuristic, reopen=True
    )


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*)
# =============================================================================
IDA_TT_CAPACITY = 1 << 14


def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="IDA*")
    start = env.get_initial_state()
    h_start = heuristic(start, env)
    if h_start == math.inf:
        metrics.status_message = "Initial state is a proven deadlock."
        return metrics
    bound = h_start

    def ordered_children(state: DeepCoreState, g: int) -> List[Tuple[float, str, DeepCoreState, int]]:
        children = []
        for action, child, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            h = heuristic(child, env)
            if h == math.inf:
                continue
            children.append((g + step_cost + h, action.value, child, step_cost))
        children.sort(key=lambda item: item[0])
        return children

    while True:
        if env.is_goal(start):
            return _finish(metrics, [], 0)
        next_bound = math.inf
        on_path: Set[DeepCoreState] = {start}
        table: Dict[DeepCoreState, int] = {start: 0}
        path_actions: List[str] = []
        metrics.nodes_expanded += 1
        metrics.explored_order.append(start.bot_pos)
        # frame: [state, g, children, next index]
        stack: List[list] = [[start, 0, ordered_children(start, 0), 0]]

        while stack:
            depth = len(stack)
            metrics.max_frontier_size = max(metrics.max_frontier_size, depth)
            metrics.max_memory_size = max(metrics.max_memory_size, depth + len(table))
            frame = stack[-1]
            state, g, children, idx = frame
            if idx >= len(children):
                stack.pop()
                on_path.discard(state)
                if path_actions:
                    path_actions.pop()
                continue
            frame[3] = idx + 1
            f, action, child, step_cost = children[idx]
            if f > bound:
                next_bound = min(next_bound, f)
                # children are sorted by f
                frame[3] = len(children)
                continue
            if child in on_path:
                continue
            child_g = g + step_cost
            if env.is_goal(child):
                return _finish(metrics, path_actions + [action], child_g, f"Solved with f-bound {bound:g}")
            seen_g = table.get(child)
            if seen_g is not None and seen_g <= child_g:
                continue
            if seen_g is None and len(table) >= IDA_TT_CAPACITY:
                del table[next(iter(table))]
            table[child] = child_g
            if _limit_hit(metrics, max_expansions):
                metrics.status_message = LIMIT_MESSAGE
                return metrics
            metrics.nodes_expanded += 1
            metrics.explored_order.append(child.bot_pos)
            on_path.add(child)
            path_actions.append(action)
            stack.append([child, child_g, ordered_children(child, child_g), 0])

        if next_bound == math.inf:
            metrics.status_message = NO_SOLUTION_MESSAGE
            return metrics
        bound = next_bound


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*)
# =============================================================================
SMA_DEFAULT_MEMORY = 5000


class _SMANode:
    __slots__ = (
        "state", "parent", "action", "g", "f", "depth", "succ", "next_idx",
        "children", "forgotten", "alive", "seq",
    )

    def __init__(self, state, parent, action, g, f, depth, seq):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f
        self.depth = depth
        self.succ: Optional[List[Tuple[str, DeepCoreState, int, float]]] = None
        self.next_idx = 0
        self.children: Dict[int, "_SMANode"] = {}
        self.forgotten: Dict[int, float] = {}
        self.alive = True
        self.seq = seq

    def can_generate(self) -> bool:
        return self.succ is None or self.next_idx < len(self.succ) or bool(self.forgotten)

    def fully_generated(self) -> bool:
        return self.succ is not None and self.next_idx >= len(self.succ)


def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    limit = SMA_DEFAULT_MEMORY if memory_limit is None else memory_limit
    if limit < 2:
        raise ValueError("SMA* needs memory_limit >= 2.")
    metrics = SearchMetrics(algorithm_name="SMA*")
    start = env.get_initial_state()
    h_start = heuristic(start, env)
    if h_start == math.inf:
        metrics.status_message = "Initial state is a proven deadlock."
        return metrics

    seq = 0
    root = _SMANode(start, None, None, 0, h_start, 0, seq)
    used = 1
    open_count = 1
    best_heap: List[Tuple[float, int, int, _SMANode]] = [(root.f, 0, root.seq, root)]
    worst_heap: List[Tuple[float, int, int, _SMANode]] = []
    is_open = {root: True}
    in_tree: Dict[DeepCoreState, _SMANode] = {start: root}

    def push_best(node: _SMANode) -> None:
        heapq.heappush(best_heap, (node.f, -node.depth, node.seq, node))

    def push_worst(node: _SMANode) -> None:
        if node.parent is not None:
            heapq.heappush(worst_heap, (-node.f, node.depth, -node.seq, node))

    def set_open(node: _SMANode, value: bool) -> None:
        nonlocal open_count
        if is_open.get(node, False) != value:
            open_count += 1 if value else -1
            if value:
                is_open[node] = True
            else:
                is_open.pop(node, None)
        if value:
            push_best(node)

    def backup(node: Optional[_SMANode]) -> None:
        while node is not None and node.fully_generated():
            values = [c.f for c in node.children.values()]
            values.extend(node.forgotten.values())
            new_f = min(values) if values else math.inf
            if new_f == node.f:
                return
            node.f = new_f
            if is_open.get(node):
                push_best(node)
            if not node.children:
                push_worst(node)
            node = node.parent

    def evict_worst(protect: _SMANode) -> bool:
        nonlocal used
        while worst_heap:
            neg_f, _, _, leaf = heapq.heappop(worst_heap)
            if not leaf.alive or leaf.children or -neg_f != leaf.f or leaf is protect:
                continue
            parent = leaf.parent
            idx = next(i for i, c in parent.children.items() if c is leaf)
            del parent.children[idx]
            parent.forgotten[idx] = leaf.f
            leaf.alive = False
            if in_tree.get(leaf.state) is leaf:
                del in_tree[leaf.state]
            set_open(leaf, False)
            used -= 1
            set_open(parent, True)
            if not parent.children:
                push_worst(parent)
            return True
        return False

    def path_of(node: _SMANode) -> List[str]:
        actions: List[str] = []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions

    def compact() -> None:
        nonlocal best_heap, worst_heap
        best_heap = [e for e in best_heap if e[3].alive and is_open.get(e[3]) and e[0] == e[3].f]
        worst_heap = [e for e in worst_heap if e[3].alive and not e[3].children and -e[0] == e[3].f]
        heapq.heapify(best_heap)
        heapq.heapify(worst_heap)

    while True:
        metrics.max_frontier_size = max(metrics.max_frontier_size, open_count)
        metrics.max_memory_size = max(metrics.max_memory_size, used)
        if len(best_heap) + len(worst_heap) > 8 * used + 1024:
            compact()
        node = None
        while best_heap:
            f, _, _, cand = heapq.heappop(best_heap)
            if cand.alive and is_open.get(cand) and f == cand.f:
                node = cand
                break
        if node is None or node.f == math.inf:
            metrics.status_message = (
                f"No solution within memory_limit={limit}." if node is not None else NO_SOLUTION_MESSAGE
            )
            return metrics
        if env.is_goal(node.state):
            return _finish(metrics, path_of(node), node.g, f"Solved within memory_limit={limit}")

        if node.succ is None:
            metrics.nodes_expanded += 1
            metrics.explored_order.append(node.state.bot_pos)
            ancestors = set()
            walker = node
            while walker is not None:
                ancestors.add(walker.state)
                walker = walker.parent
            node.succ = []
            for action, child_state, step_cost in env.get_successors(node.state):
                if child_state in ancestors:
                    continue
                h = heuristic(child_state, env)
                if h == math.inf:
                    continue
                node.succ.append((action.value, child_state, step_cost, h))

        if node.next_idx < len(node.succ):
            idx = node.next_idx
            node.next_idx += 1
            action, child_state, step_cost, h = node.succ[idx]
            child_f = max(node.f, node.g + step_cost + h)
        elif node.forgotten:
            idx = min(node.forgotten, key=node.forgotten.get)
            child_f = node.forgotten.pop(idx)
            action, child_state, step_cost, h = node.succ[idx]
        else:
            set_open(node, False)
            node.f = math.inf
            push_worst(node)
            backup(node.parent)
            continue

        child_g = node.g + step_cost
        twin = in_tree.get(child_state)
        if twin is not None and twin.alive and twin.g <= child_g:
            if not node.can_generate():
                set_open(node, False)
            else:
                push_best(node)
            if node.fully_generated() and not node.children and not node.forgotten:
                node.f = math.inf
                push_worst(node)
                backup(node.parent)
            else:
                backup(node)
            continue

        child_depth = node.depth + 1
        if child_depth >= limit - 1 and not env.is_goal(child_state):
            child_f = math.inf

        if used >= limit:
            evict_worst(protect=node)
            if not node.alive:
                continue

        seq += 1
        child = _SMANode(child_state, node, action, child_g, child_f, child_depth, seq)
        node.children[idx] = child
        in_tree[child_state] = child
        used += 1
        metrics.nodes_generated += 1
        set_open(child, True)
        push_worst(child)

        if not node.can_generate():
            set_open(node, False)
        else:
            push_best(node)
        backup(node)
