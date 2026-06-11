#!/usr/bin/env python3
"""Maze version of an AlphaDev-like RL demo.

The contrast:

1. Ordinary RL learns a policy for one maze: state -> move.
2. AlphaDev-like RL searches for a reusable program: partial program -> append
   one rule, then execute the candidate program on several mazes.

This keeps the AlphaDev idea while replacing x86 assembly with tiny maze rules.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Point = Tuple[int, int]
Maze = Sequence[str]
Program = List[str]

MAZES: Sequence[Maze] = (
    (
        "#######",
        "#S....#",
        "#.###.#",
        "#...#.#",
        "###.#.#",
        "#....G#",
        "#######",
    ),
    (
        "#######",
        "#S#...#",
        "#.#.#G#",
        "#...#.#",
        "###.#.#",
        "#.....#",
        "#######",
    ),
    (
        "#######",
        "#S....#",
        "###.#.#",
        "#...#.#",
        "#.###.#",
        "#....G#",
        "#######",
    ),
)

MOVE_ACTIONS: Sequence[Point] = ((-1, 0), (0, 1), (1, 0), (0, -1))
MOVE_NAMES: Sequence[str] = ("UP", "RIGHT", "DOWN", "LEFT")

RULES: Sequence[str] = (
    "GOAL_AHEAD",
    "RIGHT_OPEN",
    "FRONT_OPEN",
    "LEFT_OPEN",
    "TURN_BACK",
)
MAX_PROGRAM_LEN = 5
MAX_STEPS = 80

DEMO_SEARCH_TRACE: Sequence[Program] = (
    ["GOAL_AHEAD"],
    ["FRONT_OPEN"],
    ["FRONT_OPEN", "LEFT_OPEN"],
    ["LEFT_OPEN"],
)


def find_tile(maze: Maze, tile: str) -> Point:
    for r, row in enumerate(maze):
        for c, value in enumerate(row):
            if value == tile:
                return r, c
    raise ValueError(f"tile {tile!r} not found")


def is_open(maze: Maze, point: Point) -> bool:
    r, c = point
    return maze[r][c] != "#"


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def train_ordinary_q_learning(maze: Maze, episodes: int = 1200) -> List[Point]:
    """Learns a movement policy for one maze."""
    rng = random.Random(4)
    start = find_tile(maze, "S")
    goal = find_tile(maze, "G")
    q: Dict[Point, List[float]] = defaultdict(lambda: [0.0] * len(MOVE_ACTIONS))
    alpha = 0.35
    gamma = 0.92

    for episode in range(episodes):
        state = start
        epsilon = max(0.05, 0.35 * (1.0 - episode / episodes))
        for _ in range(MAX_STEPS):
            if rng.random() < epsilon:
                action_id = rng.randrange(len(MOVE_ACTIONS))
            else:
                action_id = max(range(len(MOVE_ACTIONS)), key=lambda i: q[state][i])

            candidate = add(state, MOVE_ACTIONS[action_id])
            next_state = candidate if is_open(maze, candidate) else state
            done = next_state == goal
            step_reward = 1.0 if done else -0.02
            target = step_reward + (0.0 if done else gamma * max(q[next_state]))
            q[state][action_id] += alpha * (target - q[state][action_id])
            state = next_state
            if done:
                break

    path = [start]
    state = start
    seen = set()
    for _ in range(MAX_STEPS):
        seen.add(state)
        action_id = max(range(len(MOVE_ACTIONS)), key=lambda i: q[state][i])
        candidate = add(state, MOVE_ACTIONS[action_id])
        state = candidate if is_open(maze, candidate) else state
        path.append(state)
        if state == goal or state in seen:
            break
    return path


def rotate(direction: int, amount: int) -> int:
    return (direction + amount) % 4


def execute_rule(program: Program, maze: Maze, position: Point, direction: int) -> Tuple[Point, int]:
    goal = find_tile(maze, "G")
    directions = MOVE_ACTIONS

    def ahead(dir_id: int) -> Point:
        return add(position, directions[dir_id])

    for rule in program:
        if rule == "GOAL_AHEAD" and ahead(direction) == goal:
            return ahead(direction), direction
        if rule == "RIGHT_OPEN":
            right = rotate(direction, 1)
            if is_open(maze, ahead(right)):
                return ahead(right), right
        if rule == "FRONT_OPEN" and is_open(maze, ahead(direction)):
            return ahead(direction), direction
        if rule == "LEFT_OPEN":
            left = rotate(direction, -1)
            if is_open(maze, ahead(left)):
                return ahead(left), left
        if rule == "TURN_BACK":
            back = rotate(direction, 2)
            if is_open(maze, ahead(back)):
                return ahead(back), back
    return position, rotate(direction, 1)


def run_program(program: Program, maze: Maze) -> Tuple[bool, List[Point]]:
    start = find_tile(maze, "S")
    goal = find_tile(maze, "G")
    position = start
    direction = 1
    path = [position]
    for _ in range(MAX_STEPS):
        position, direction = execute_rule(program, maze, position, direction)
        path.append(position)
        if position == goal:
            return True, path
    return False, path


def program_score(program: Program) -> float:
    results = [run_program(program, maze) for maze in MAZES]
    solved = sum(ok for ok, _path in results)
    avg_steps = sum(len(path) for _ok, path in results) / len(results)
    return solved / len(MAZES) - 0.01 * len(program) - 0.001 * avg_steps


def softmax(logits: Sequence[float]) -> List[float]:
    peak = max(logits)
    exp = [math.exp(x - peak) for x in logits]
    total = sum(exp)
    return [x / total for x in exp]


def sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    cursor = rng.random()
    running = 0.0
    for index, probability in enumerate(probabilities):
        running += probability
        if cursor <= running:
            return index
    return len(probabilities) - 1


def train_random_search(budget: int = 200, seed: int = 7) -> Tuple[Program, List[Tuple[int, float, int]]]:
    """Pure random baseline: uniformly sample programs and keep the best.

    Returns (best_program, history) where history entries are
    (candidates_tried, best_score, best_program_length).
    """
    rng = random.Random(seed)
    best_program: Program = []
    best_score = -999.0
    history: List[Tuple[int, float, int]] = []

    for tried in range(1, budget + 1):
        length = rng.randint(1, MAX_PROGRAM_LEN)
        program: Program = [RULES[rng.randrange(len(RULES))] for _ in range(length)]
        score = program_score(program)
        if score > best_score:
            best_score = score
            best_program = program[:]
        if tried % 25 == 0 or tried == budget:
            history.append((tried, best_score, len(best_program)))

    return best_program, history


def train_alphadev_like(iterations: int = 160) -> Program:
    """Cross-entropy style policy search over rule programs."""
    rng = random.Random(9)
    logits = [[0.0 for _ in RULES] for _ in range(MAX_PROGRAM_LEN)]
    best_program: Program = []
    best_score = -999.0

    for _ in range(iterations):
        candidates = []
        for _sample in range(48):
            program = []
            program_len = rng.randint(1, MAX_PROGRAM_LEN)
            for position in range(program_len):
                rule_id = sample_index(softmax(logits[position]), rng)
                program.append(RULES[rule_id])
            candidates.append((program_score(program), program))

        candidates.sort(reverse=True, key=lambda item: item[0])
        if candidates[0][0] > best_score:
            best_score, best_program = candidates[0]

        for _score, program in candidates[:8]:
            for position, rule in enumerate(program):
                logits[position][RULES.index(rule)] += 0.22

        if all(run_program(best_program, maze)[0] for maze in MAZES):
            break

    return best_program


def render_maze(maze: Maze, path: Iterable[Point]) -> str:
    path_set = set(path)
    lines = []
    for r, row in enumerate(maze):
        chars = []
        for c, char in enumerate(row):
            if (r, c) in path_set and char == ".":
                chars.append("*")
            else:
                chars.append(char)
        lines.append("".join(chars))
    return "\n".join(lines)


TEST_MAZES: Sequence[Maze] = (
    # 測試迷宮 1 — 螺旋型路徑（訓練時從未見過）
    (
        "#######",
        "#S....#",
        "#####.#",
        "#.....#",
        "#.###.#",
        "#....G#",
        "#######",
    ),
    # 測試迷宮 2 — 起點在左側中間（訓練時從未見過）
    (
        "#######",
        "#.....#",
        "#S###.#",
        "#.....#",
        "###.#.#",
        "#....G#",
        "#######",
    ),
)


def _sep(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    # ── 1. Ordinary RL ──────────────────────────────────────
    _sep("1. Ordinary RL：學習單一迷宮的 policy")
    ordinary_path = train_ordinary_q_learning(MAZES[0])
    solved = ordinary_path[-1] == find_tile(MAZES[0], "G")
    print(f"  Solved maze 1: {solved}  (path length: {len(ordinary_path) - 1} steps)")
    print(render_maze(MAZES[0], ordinary_path))
    print()
    print("  注意：只學了這一張迷宮，換張迷宮就需要重新訓練。")

    # ── 2. Candidate evaluation trace ───────────────────────
    _sep("2. AlphaDev-like 候選程式搜尋過程")
    print(f"  {'program':28s}  solved  score")
    print(f"  {'-'*28}  ------  ------")
    for candidate in DEMO_SEARCH_TRACE:
        n = sum(run_program(candidate, maze)[0] for maze in MAZES)
        s = program_score(candidate)
        print(f"  {' → '.join(candidate):28s}  {n}/{len(MAZES)}     {s:+.3f}")

    # ── 3. Search strategy comparison ───────────────────────
    _sep("3. 搜尋策略對比：Random Search vs AlphaDev-like")

    print("\n  [Random Search]  budget=200 programs")
    rnd_best, rnd_hist = train_random_search(budget=200)
    print(f"  {'tried':>6}  {'best_score':>10}  {'rules':>5}")
    for tried, score, rules in rnd_hist:
        print(f"  {tried:>6}  {score:>10.4f}  {rules:>5}")
    rnd_solved = sum(run_program(rnd_best, m)[0] for m in MAZES)
    print(f"\n  Final best: {' → '.join(rnd_best)}")
    print(f"  Solves {rnd_solved}/{len(MAZES)} mazes  |  {len(rnd_best)} rule(s)")

    print()
    print("  [AlphaDev-like]  cross-entropy, up to 160 iterations × 48 samples")
    adv_best = train_alphadev_like()
    adv_solved = sum(run_program(adv_best, m)[0] for m in MAZES)
    print(f"\n  Final best: {' → '.join(adv_best)}")
    print(f"  Solves {adv_solved}/{len(MAZES)} mazes  |  {len(adv_best)} rule(s)")

    print()
    print(f"  ┌────────────────────┬──────────────────────────────┬────────┬───────┐")
    print(f"  │ Strategy           │ Best Program                 │ Solved │ Rules │")
    print(f"  ├────────────────────┼──────────────────────────────┼────────┼───────┤")
    print(f"  │ Random Search      │ {' → '.join(rnd_best):28s} │ {rnd_solved}/{len(MAZES)}    │   {len(rnd_best)}   │")
    print(f"  │ AlphaDev-like      │ {' → '.join(adv_best):28s} │ {adv_solved}/{len(MAZES)}    │   {len(adv_best)}   │")
    print(f"  └────────────────────┴──────────────────────────────┴────────┴───────┘")
    print()
    print("  補充說明：此玩具迷宮的程式空間僅約 3,900 種組合（5 rules × 最長 5 條），")
    print("  因此隨機搜尋也能找到最優方案。")
    print("  真實 AlphaDev 面對的 x86 組語空間超過 10^14，隨機搜尋在此規模下")
    print("  幾乎不可能在合理時間內找到有效程式，這正是 MCTS+cross-entropy 的價值所在。")

    # ── 4. Generalization test ───────────────────────────────
    _sep("4. 程式泛化能力：把訓練到的程式跑在從未見過的迷宮")
    print(f"  Program: {' → '.join(adv_best)}")
    print()
    print("  Training mazes (seen during search):")
    for i, maze in enumerate(MAZES, start=1):
        ok, path = run_program(adv_best, maze)
        print(f"    Maze {i}: solved={ok}, steps={len(path) - 1}")
    print()
    print("  Test mazes (NEVER seen during search):")
    for i, maze in enumerate(TEST_MAZES, start=1):
        ok, path = run_program(adv_best, maze)
        status = "✓ solved" if ok else "✗ failed"
        print(f"    Test maze {i}: {status}, steps={len(path) - 1}")
        print(render_maze(maze, path))


if __name__ == "__main__":
    main()
