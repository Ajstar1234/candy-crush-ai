
# Candy Crush AI

A Python match-three game and AI experimentation platform for comparing heuristic search, supervised machine learning, and deep reinforcement learning.

## Project Overview

Candy Crush AI is a custom match-three environment where agents select legal swaps to collect target-colored candies before running out of moves. The project emphasizes **agent design, reproducible experiments, and software engineering** rather than recreating every feature of the commercial game.

The project currently includes:

- A configurable, object-oriented match-three game engine
- Greedy and objective-aware heuristic agents
- Supervised PyTorch move-evaluation models, including target-aware features and pairwise ranking experiments
- A Deep Q-Network (DQN) agent with experience replay, a target network, and epsilon-greedy exploration
- Automated unit tests and agent benchmarks

**Current focus:** Standardizing evaluation across agents and investigating why DQN performance improves early in training but deteriorates with additional updates.

## Features

### Game Environment

- Object-oriented board, candy, game, and agent classes
- Configurable board sizes (including 5×5 and 9×9) and candy counts (including six-color boards)
- Horizontal and vertical matches; legal swaps and move generation
- Candy removal, gravity, refill, and cascading matches
- Horizontal/vertical striped candies, wrapped candies, and color bombs
- Special-candy activation and chain reactions
- Reshuffling when no legal moves remain
- Configurable move limits and candy-collection objectives
- Score, objective-progress, and win/loss tracking
- Automated tests with `pytest`

### AI Agents

| Agent | Approach |
|---|---|
| Greedy | Selects moves based on immediate matching opportunities |
| Objective-aware | Prioritizes moves that collect candies required by the level objective |
| Supervised ML | Uses a PyTorch neural network to score candidate moves |
| DQN | Learns state-action values from gameplay rewards and selects moves using an epsilon-greedy policy |

The supervised experiments also compare **mean squared error (MSE)** training with **pairwise move-ranking** objectives.

## Supervised Machine Learning

### Feature Engineering

| Feature representation | Board | Input features |
|---|---|---:|
| Original one-hot encoding | 5×5, four colors | 180 |
| Expanded encoding | 9×9, six colors | 734 |
| Target-mask encoding | 9×9, six colors | 815 |
| RL state-action encoding | 9×9, six colors | 817 |

The feature vector represents candy colors, special-candy information, the candidate swap, and the objective. The target-mask representation explicitly marks cells containing the target candy. The RL representation extends it with remaining turns and remaining target candies.

### Neural Network and Training

The move-evaluation model uses a fully connected PyTorch network with two hidden layers (64 and 32 units), ReLU activations, and a scalar output.

Supervised training includes:

- Mini-batch optimization
- Adam optimizer
- Validation-based checkpoint selection
- Separate training, validation, and test data
- Multiple feature representations
- MSE regression and pairwise ranking experiments

The initial 5×5 supervised model achieved **6.81 test MSE**, versus **6.40** for a constant-mean baseline, motivating work on more informative features and the distinction between score regression and move selection.

In a subsequent 9×9 target-mask experiment, the best validation MSE was **2.7996**, versus **2.9126** for a constant-mean validation baseline.

### Corrected Move-Ranking Comparison

The following results were measured on **500 generated 9×9 boards** using a corrected six-color heuristic and the same evaluation board seeds for each model.

"Optimal-move accuracy" counts a move as correct if it ties for the heuristic's highest-scoring legal action. The score gap measures the average difference from the heuristic's best move.

| Move-selection method | Heuristic-optimal move accuracy | Mean heuristic score gap |
|---|---:|---:|
| Corrected 734-feature MSE model | 12.8% | 6.90 |
| **815-feature target-mask MSE model** | **17.2%** | **6.02** |
| Pairwise ranking model (learning rate 0.001) | 13.2% | 6.90 |
| Pairwise ranking model (learning rate 0.0001) | 13.2% | 6.92 |
| Random legal move | 14.24% | 6.60 |

The target-mask MSE model performed best among these learned variants on these metrics.

Pairwise ranking training did not improve move-selection performance in these experiments.

These metrics evaluate agreement with the **heuristic**, not necessarily the best action under the full game simulator.

### Gameplay Benchmarks

The following historical gameplay results use different board configurations and model versions. They should **not** be interpreted as a controlled comparison of model training improvements.

#### Original 5×5 Benchmark

Four candy colors:

| Difficulty | Greedy | Objective-aware | Supervised ML |
|---|---:|---:|---:|
| Easy | 84% | 92% | 68% |
| Medium | 57% | 63% | 33% |
| Hard | 22% | 21% | 7% |

#### Earlier Expanded 9×9 Benchmark

Six candy colors, using the original 734-feature ML model:

| Difficulty | Greedy | Objective-aware | Supervised ML |
|---|---:|---:|---:|
| Easy | 33% | 75% | 16% |
| Medium | 22% | 46% | 4% |
| Hard | 7% | 18% | 1% |

For these benchmark difficulties, the configurations were:

| Difficulty | Move limit | Target candies |
|---|---:|---:|
| Easy | 8 | 10 |
| Medium | 6 | 12 |
| Hard | 5 | 15 |

Each agent was evaluated on 100 games per difficulty.

#### Corrected Model vs. Target-Mask Model

A later 9×9 comparison of the corrected 734-feature and target-mask MSE models yielded the following win rates, again with 100 games per difficulty:

| Difficulty | Corrected 734-feature ML | Target-mask ML |
|---|---:|---:|
| Easy | 19% | 25% |
| Medium | 9% | 9% |
| Hard | 2% | 1% |

The objective-aware heuristic remains a substantially stronger gameplay baseline than the current supervised models.

## Deep Reinforcement Learning

### DQN Implementation

The reinforcement learning agent operates on the **9×9, six-color** environment.

Its state-action encoder returns **817 features**, including:

- Candy colors and special-candy information
- Proposed move coordinates
- Target-color information
- Remaining turns
- Remaining target candies

The DQN implementation includes:

- An online/policy Q-network and a periodically synchronized target network
- Epsilon-greedy action selection over legal moves
- A replay buffer with a capacity of 2,000 experiences
- Mini-batch Bellman updates (batch size 32) using Huber loss
- Discount factor `gamma = 0.99`
- Adam optimization
- Epsilon decay from `1.0` toward a minimum of `0.05`
- Reproducible training seeds and checkpoints at episodes 0, 25, 50, 75, and 100

### Reward Function

The current reward function prioritizes completion of candy-collection objectives.

For each move:

- The agent receives a reward equal to the number of target-colored candies collected.
- Successfully completing the level provides an additional reward of 10.

The reward function does not currently include a general score reward or a per-move efficiency penalty.

### Training Experiments

Initial DQN experiments were conducted over 25 and 100 training episodes.

The agent successfully collected gameplay experiences, updated its policy network using experience replay, and periodically synchronized its target network.

However, additional training did not consistently improve gameplay performance.

To investigate this, checkpoints were saved throughout a single reproducible 100-episode training run.

### DQN Checkpoint Evaluation

Each checkpoint was evaluated using a greedy policy (`epsilon = 0`) on 500 starting boards per evaluation set.

The random policy selects uniformly from legal moves.

| Agent / checkpoint | Seeds 20000–20499 | Fresh seeds 30000–30499 |
|---|---:|---:|
| Untrained (episode 0) | 20.8% | 19.8% |
| Episode 25 | 30.6% | Not evaluated |
| **Episode 50** | **38.2%** | **31.4%** |
| Episode 75 | 18.6% | Not evaluated |
| Episode 100 | 22.4% | 18.2% |
| Random legal move | 27.2% | 25.4% |

On the fresh 500-game evaluation set:

| Metric | DQN (50 episodes) | Random |
|---|---:|---:|
| Games | 500 | 500 |
| Wins | 157 | 127 |
| Win rate | **31.4%** | 25.4% |
| Average reward | **11.04** | 9.90 |

The episode-50 DQN outperformed random selection on both evaluation sets.

However, the episode-100 DQN fell to **18.2%** on the fresh evaluation set.

This demonstrates that **additional training updates did not reliably improve policy performance in this experiment**.

Potential explanations include unstable or overestimated Q-values, although the precise cause has not yet been established.

Because the episode-50 checkpoint was selected after inspecting the earlier evaluation results, the fresh evaluation provides a useful follow-up check. Independent training runs, broader target-color tests, and statistical comparisons are still needed.

The current DQN has not yet demonstrated performance comparable to the objective-aware heuristic.

### Planned RL Improvements

Future experiments will investigate:

- Double DQN to reduce Q-value overestimation
- More extensive training with intermediate checkpoints
- Larger experience replay buffers
- Input and reward normalization
- Performance evaluation across multiple random seeds
- Alternative reward functions
- Training stability and generalization

Potential extensions include convolutional neural networks and supervised pretraining.

## Development Roadmap

### Phase 1: Game Environment — Complete

- [x] Board generation and legal swaps
- [x] Horizontal and vertical matching
- [x] Gravity, refilling, and cascades
- [x] Special candies and chain reactions

### Phase 2: Heuristic Agents — Complete

- [x] Greedy agent
- [x] Objective-aware agent
- [x] Automated gameplay evaluation

### Phase 3: Objective-Based Gameplay — Complete

- [x] Configurable move limits
- [x] Candy-collection objectives
- [x] Win/loss conditions
- [x] Board reshuffling

### Phase 4: Supervised Machine Learning — Complete

- [x] Supervised dataset generation
- [x] Neural network training
- [x] Feature encoding experiments
- [x] Corrected model comparisons
- [x] Pairwise ranking experiments
- [x] Move-ranking and gameplay benchmarks

### Phase 5: Reinforcement Learning — In Progress

- [x] Reinforcement learning environment
- [x] State-action feature encoding
- [x] Deep Q-Network agent
- [x] Experience replay
- [x] Target network synchronization
- [x] Epsilon-greedy exploration
- [x] Training and checkpoint saving
- [x] Initial training and evaluation experiments
- [ ] Improve training stability
- [ ] Evaluate longer training runs
- [ ] Compare RL against other agents under unified benchmark conditions

### Final Project Improvements — Planned

- [ ] Unified evaluation framework
- [ ] Per-game results and confidence intervals
- [ ] Evaluation across multiple target colors
- [ ] Stronger lookahead heuristic baseline
- [ ] Final performance charts
- [ ] GitHub Actions CI
- [ ] Final project documentation and analysis

## Technologies

- **Python:** Game environment, algorithms, simulation, and AI agents
- **PyTorch:** Neural network training and inference
- **NumPy:** Feature engineering and state encoding
- **pandas:** Benchmark analysis and data processing
- **Matplotlib:** Performance visualization
- **pytest:** Automated testing
- **Git/GitHub:** Version control

## Running the Project

Commands below are run from the repository root.

On Windows, `py` uses the Python launcher. On other systems, replace it with the appropriate Python command.

### Gameplay

```bash
py main.py
```

The gameplay configuration and selected agent can be adjusted in `main.py`.

### Supervised ML Training

Generate supervised training data:

```bash
py generate_dataset.py
```

Prepare datasets:

```bash
py prepare_dataset.py
```

Train the configured supervised model:

```bash
py train_model.py
```

The target-mask training configuration saves its selected model to:

```text
models/best_move_model_9x9_target_mask.pth
```

### DQN Training

Train the reinforcement learning agent:

```bash
py train_dqn.py
```

The current training script runs a seeded experiment and saves intermediate checkpoints, including:

```text
models/dqn_seed42_ep0.pth
models/dqn_seed42_ep25.pth
models/dqn_seed42_ep50.pth
models/dqn_seed42_ep75.pth
models/dqn_seed42_ep100.pth
```

### Benchmarking

Run the heuristic-agent benchmark:

```bash
py -m benchmarks.benchmark
```

Run the supervised ML gameplay benchmark:

```bash
py -m benchmarks.benchmark_ml
```

Evaluate supervised move-ranking performance:

```bash
py -m benchmarks.evaluate_move_rankings
```

Evaluate DQN checkpoints:

```bash
py -m benchmarks.benchmark_dqn
```

Each benchmark currently has its own agent and model configuration.

**Results from different scripts are not necessarily directly comparable yet.**

A unified evaluation framework is a planned improvement.

### Model Checkpoints

PyTorch checkpoint files (`.pth`) are excluded from Git version control.

Models must be trained or supplied separately before loading them for inference.

### Testing

Run the test suite from the project root:

```bash
py -m pytest tests/ -v
```

The test suite covers:

- Board initialization and match detection
- Legal and invalid swaps
- Special-candy creation and activation
- Power-up chain reactions
- Objective tracking and win/loss conditions
- Configurable board generation
- ML feature encoding
- Supervised model behavior
- RL environment transitions
- Replay memory
- DQN action selection
- Network training and target-network synchronization

Some model-loading tests require local checkpoint files.

## Findings and Limitations

### Objective-Aware Decision Making

Maximizing immediate score does not necessarily maximize objective completion.

The objective-aware heuristic consistently demonstrated the importance of selecting moves based on the level's candy-collection requirements.

### Feature Engineering

Explicit target-candy masks improved heuristic-aligned move ranking in the corrected supervised experiments.

This suggests that providing the network with objective-relevant information can improve its decision-making.

### Regression Accuracy vs. Move Quality

Low prediction error does not guarantee that a model selects the strongest available action.

The supervised experiments also demonstrated that pairwise ranking objectives do not automatically outperform regression-based models.

### Reinforcement Learning Stability

The DQN learned a policy that outperformed random selection during early training, but additional updates reduced its gameplay performance.

This highlights the importance of evaluating intermediate checkpoints rather than assuming that lower training loss or longer training produces better policies.

### Evaluation Methodology

Reliable comparisons require consistent board configurations, objective settings, evaluation seeds, and sufficient games to measure uncertainty.

A unified evaluation framework is a major planned improvement.

## Project Status

**Actively in Development**

The core game environment, heuristic agents, supervised machine learning pipeline, and initial DQN training system are implemented.

Current development focuses on:

- Standardizing benchmarks across agents
- Improving reinforcement learning stability
- Investigating training generalization
- Developing stronger performance baselines
- Preparing final experimental comparisons and documentation

The project serves as an ongoing exploration of artificial intelligence, machine learning experimentation, and software engineering through a custom match-three game environment.
