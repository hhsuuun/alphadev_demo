# MazeDev: AlphaDev-like RL Demo

This folder contains a small, executable demo for a reinforcement-learning
course report. It uses a maze game to show the difference between ordinary RL
and AlphaDev-like program discovery.

## What The Demo Shows

Ordinary RL:

- State: current maze position.
- Action: move up, right, down, or left.
- Reward: reach the goal in one maze.
- Result: a policy that must keep acting online.

AlphaDev-like RL:

- State: the partial program discovered so far.
- Action: append one maze rule.
- Reward: execute the candidate program on several mazes and score success,
  with a penalty for longer programs.
- Result: a reusable maze-solving program.

In this simplified demo, the discovered program is:

```text
LEFT_OPEN
```

The interpreter tries the program repeatedly. If the left side is open, it turns
left and moves. If no rule fires, it rotates right. This is a tiny wall-following
algorithm discovered as a program rather than a state-by-state policy.

The browser demo also shows the decision/search process:

```text
GOAL_AHEAD              -> solves 0/3 mazes
FRONT_OPEN              -> solves 2/3 mazes
FRONT_OPEN + LEFT_OPEN  -> solves 3/3 mazes
LEFT_OPEN               -> solves 3/3 mazes with a shorter program
```

This mirrors AlphaDev's idea: sample or search candidate programs, execute each
candidate on a task distribution, score it, and keep the best program.

## Files

- `index.html`: browser visualization comparing ordinary RL and AlphaDev-like
  program discovery.
- `toy_alphadev_rl.py`: command-line training/demo code.

## Run

Browser demo:

```bash
python3 -m http.server 8765
```

Then open:

```text
http://localhost:8765/alphadev/rl_extension_demo/index.html
```

Command-line demo:

```bash
python3 alphadev/rl_extension_demo/toy_alphadev_rl.py
```

## How This Maps To AlphaDev

Original AlphaDev components in this repository:

- `../alphadev.py`
  - `AssemblyGame`: the RL environment.
  - `TaskSpec`: action/state/reward configuration.
  - `AlphaDevConfig`: AlphaZero/MCTS/training hyperparameters.
  - `play_game`, `run_mcts`, `train_network`: pseudocode for self-play,
    search, and learning.
- `../sort_functions_test.cc`
  - Released assembly programs found by AlphaDev.
  - Correctness tests that execute each discovered sorting program.

Mapping:

| AlphaDev paper/code | MazeDev demo |
| --- | --- |
| x86 assembly instruction | Maze rule such as `LEFT_OPEN` |
| Partial assembly program | Partial maze-solving program |
| CPU simulator state | Maze interpreter state |
| Test inputs for sorting | Multiple maze maps |
| Correctness + latency reward | Success rate + program-length reward |
| Discovered sort routine | Discovered maze-solving routine |

## Three-Minute Report Message

The key line:

Ordinary RL trains a player. AlphaDev-like RL trains an algorithm discoverer.
After discovery, the final deployed object is a program, not the neural network.
