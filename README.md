# Candy Crush AI

**A Python match-three game and AI experimentation project** comparing heuristic search, supervised learning, and deep reinforcement learning.

This project implements a configurable match-three environment from scratch, then uses it to study how different agents select moves under limited turns and candy-collection objectives. The goal is to build, evaluate, and understand the agents—not to claim that a neural network necessarily outperforms a carefully designed heuristic.

## Highlights

- **Custom game engine:** configurable 5×5 and 9×9 boards, multiple candy colors, legal swaps, cascades, gravity and refill, reshuffling, and win/loss objectives
- **Special candies:** horizontal/vertical striped candies, wrapped candies, color bombs, and related activation/chain-reaction logic
- **Three AI approaches:** greedy and objective-aware heuristics, supervised PyTorch move evaluation, and a deep Q-learning agent (DQN / Double DQN experiments)
- **Reproducible evaluation:** seeded gameplay, separate training and evaluation runs, checkpoint comparisons, and confidence intervals for win rates
- **Automated testing:** pytest tests for core game mechanics, move selection, and ML/RL components

## Results at a glance

In a **500-game evaluation on the same starting seeds (80000–80499)**, the best selected Double DQN checkpoint was compared with the objective-aware heuristic:

| Agent | Wins | Win rate | Average episode reward |
|---|---:|---:|---:|
| Objective-aware heuristic | 307/500 | **61.4%** | 16.36 |
| Double DQN (300-episode checkpoint) | 180/500 | **36.0%** | 11.89 |

The selected Double DQN checkpoint is `models/double_dqn_long_seed42_ep300.pth`. It was selected from a preceding checkpoint benchmark; the 500-game comparison above used a fresh seed range.

**Finding:** the learned agent developed a better-than-random policy in earlier evaluations, but the objective-aware heuristic remained substantially stronger. Training was not monotonic: performance deteriorated at the 200-episode checkpoint and recovered at 300 episodes. An opening-move diagnostic also found that the Double DQN's immediate heuristic move ranking was similar to random selection on 100 test boards. See [Experiment Results](docs/EXPERIMENTS.md) for full tables, seed ranges, and limitations.

## How the project works

### Game environment

The environment is implemented using separate Python classes for the board, candies, game state, and agents. Moves are legal adjacent swaps that produce a match or activate supported special candies. After a move, matches resolve, candies fall, missing positions refill, and cascades can trigger additional clears.

Levels can specify board dimensions, number of candy colors, a turn limit, a target candy type, and a collection goal. The reinforcement learning environment (`rl_environment.py`) exposes `reset()`, `get_valid_moves()`, and `step()`.

### Agents

| Approach | How moves are selected |
|---|---|
| **Greedy heuristic** | Scores candidate moves by immediate matching opportunities |
| **Objective-aware heuristic** | Adds value for clearing candy types required by the current goal |
| **Supervised ML** | Uses a PyTorch multilayer perceptron to predict heuristic move quality from board-and-move features |
| **DQN / Double DQN** | Uses learned action values to rank legal swaps, with experience replay and a target network |

### Model inputs and training

The supervised pipeline supports one-hot board representations and optional target-candy masks. The reinforcement learning model uses an **817-feature state-action vector** for the 9×9, six-color configuration, comprising encoded board contents, a target mask, move coordinates, target type, and remaining turns/collection progress.

The reinforcement learning network has hidden layers of 64 and 32 units. Its training pipeline includes epsilon-greedy exploration, an experience replay buffer, a target network, Adam optimization, and Huber loss. Experiments compare standard DQN, Double DQN, learning rates, normalized versus original input features, and training duration.

**RL reward definition:** the environment rewards each target candy collected on a move and adds a +10 bonus on winning a game. The environment also tracks game score separately.

## Project layout

```text
candy-crush-ai/
├── board.py                  # Board rules, valid moves, matches, power-ups
├── candy.py                  # Candy representation
├── game.py                   # Objectives, score, turns, win/loss
├── agent.py                  # Greedy and objective-aware heuristic logic
├── ml_agent.py               # Supervised inference agent
├── ml_model.py               # PyTorch move-evaluation network
├── ml_features.py            # State/move feature encoders
├── generate_dataset.py       # Heuristic-labeled examples
├── prepare_dataset.py        # Dataset splits / preparation
├── train_model.py            # Supervised model training
├── dqn_agent.py              # Q-learning agent
├── rl_environment.py         # RL environment and rewards
├── train_dqn.py              # DQN training and checkpoints
├── main.py                   # Console gameplay demo
├── benchmarks/               # Evaluations, comparisons, diagnostics
├── tests/                    # pytest tests
├── models/                   # Local model checkpoints (not committed)
└── docs/EXPERIMENTS.md       # Results and experimental limitations
```

## Getting started

From the project root, install the Python dependencies used across the project:

```powershell
py -m pip install numpy pandas matplotlib torch scikit-learn pytest
```

Run the console game demo (currently uses the objective-aware heuristic):

```powershell
py main.py
```

Run the test suite:

```powershell
py -m pytest tests/ -q
```

For model development and evaluation, the repository includes these commands:

```powershell
py generate_dataset.py
py prepare_dataset.py
py train_model.py
py train_dqn.py
py -m benchmarks.benchmark_dqn
py -m benchmarks.compare_best_dqn
```

**Checkpoint note:** `.pth` model weights are stored locally and are not included in version control. Benchmarks that load a checkpoint require the matching `.pth` file under `models/`. Some commands generate datasets, retrain models, or replace checkpoints with the same filenames; inspect their configurations before rerunning a historical experiment. Some tests also require local model checkpoints.

## Testing and reproducibility

The project uses seeded simulations for reproducible comparisons. Reported win rates should be interpreted with their board configurations, seed ranges, and evaluation-game counts; results from different configurations are not directly interchangeable. Checkpoint evaluation uses `epsilon=0` to measure the learned policy without exploratory random actions.

The project has tests for board behavior, swap validity, power-ups, objectives, feature encoding, and agent logic. **Run the test command above on your local checkout to verify the current branch**; the numerical results in this README document completed experiments, not a fresh CI run.

## Findings and limitations

- Objective-aware heuristics are strong baselines when game goals are explicit.
- More DQN training did **not** guarantee higher evaluation win rates; checkpoint selection matters.
- Input normalization did not show a consistent benefit in the 100-episode comparison.
- The best evaluated Double DQN still lagged behind the objective-aware heuristic, so these results do not demonstrate that reinforcement learning is superior for this task.
- The reported experiments were primarily single-seed training runs, and the model's ability to rank immediate rewards remains an area for future investigation.

Further research could explore improved action representations, more training seeds, alternative network architectures, and stronger Q-value diagnostics. These are **future experiments**, not prerequisites for the project's current functionality.

## Technology

Python · PyTorch · NumPy · pandas · Matplotlib · scikit-learn · pytest · Git

## Status

**Core implementation and comparative experiments complete.** The project is being packaged as a reproducible, documented engineering portfolio project. Further research directions are optional.
