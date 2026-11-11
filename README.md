# Candy Crush AI

An AI-focused match-three game built in Python to explore heuristic decision-making, supervised machine learning, and eventually reinforcement learning through automated gameplay.

## Project Overview

Candy Crush AI is a custom match-three game environment where artificial intelligence agents select moves to complete objectives within configurable move limits.

The project focuses on developing and comparing AI approaches rather than simply recreating Candy Crush. It currently includes greedy and objective-aware heuristic agents, a supervised PyTorch neural network, automated gameplay benchmarks, and experiments investigating how feature engineering affects AI decision-making.

The long-term goal is to develop a reinforcement learning agent and evaluate its performance against the existing approaches.

## Current Features

### Game Environment

- Object-oriented Python architecture with separate board, candy, game, and agent classes
- Configurable board dimensions, including 5×5 and 9×9 environments
- Configurable candy types, including six-color boards
- Horizontal and vertical match detection
- Legal move validation and automated move generation
- Match removal, gravity, refilling, and cascading
- Horizontal and vertical striped candies, wrapped candies, and color bombs
- Special-candy activation and chain reactions
- Board reshuffling when no valid moves are available
- Configurable move limits and candy collection objectives
- Score tracking, objective progress, and win/loss conditions
- Automated unit tests using pytest

### AI Agents

**Greedy Agent**

Evaluates legal moves using immediate matching opportunities.

**Objective-Aware Agent**

Prioritizes moves that collect candies required by the current level objective.

**Supervised ML Agent**

Uses a fully connected PyTorch neural network to predict heuristic move scores and select the highest-scoring legal move.

Supports configurable model input sizes, checkpoint loading, and multiple feature encoders.

## Machine Learning Pipeline

The supervised learning pipeline uses the objective-aware heuristic to generate labeled move-evaluation examples from simulated gameplay.

### Feature Engineering

Three feature representations have been implemented:

| Representation | Board Configuration | Input Features |
|---|---|---:|
| Original one-hot encoding | 5×5, 4 colors | 180 |
| Expanded one-hot encoding | 9×9, 6 colors | 734 |
| Target-mask encoding | 9×9, 6 colors | 815 |

Features include:

- One-hot encoded candy colors
- Special-candy information
- Candidate move coordinates
- Target candy objective
- An optional target-candy mask identifying objective-colored cells

The target-mask representation explicitly identifies candies relevant to the current objective, reducing the need for the model to infer this relationship solely from the original feature representation.

### Neural Network

Implemented using PyTorch with:

- Configurable input dimensions
- Two fully connected hidden layers (64 and 32 neurons)
- ReLU activation functions
- Mean squared error loss
- Adam optimizer
- Mini-batch training
- Validation-based checkpoint selection
- Separate training, validation, and test datasets

The pipeline supports multiple feature representations while preserving separately trained model checkpoints.

## AI Agent Benchmark Results

Automated benchmarks evaluate agent performance across three difficulty levels.

| Difficulty | Move Limit | Target Candies | Games per Agent |
|---|---:|---:|---:|
| Easy | 8 | 10 | 100 |
| Medium | 6 | 12 | 100 |
| Hard | 5 | 15 | 100 |

### Original 5×5 Benchmark

Win rates on 5×5 boards with four candy colors:

| Difficulty | Greedy | Objective-Aware | ML |
|---|---:|---:|---:|
| Easy | 84% | 92% | 68% |
| Medium | 57% | 63% | 33% |
| Hard | 22% | 21% | 7% |

These results established the initial performance baseline for the three agents.

### Expanded 9×9 Benchmark

Win rates on 9×9 boards with six candy colors:

| Difficulty | Greedy | Objective-Aware | ML |
|---|---:|---:|---:|
| Easy | 33% | 75% | 16% |
| Medium | 22% | 46% | 4% |
| Hard | 7% | 18% | 1% |

The objective-aware heuristic substantially outperformed the greedy and supervised ML agents.

Interestingly, the greedy agent sometimes achieved higher average scores despite winning fewer games, demonstrating that maximizing immediate points does not necessarily maximize objective completion.

The 9×9 ML gameplay results use the original 734-feature model. The newer target-mask model has not yet completed equivalent full-game benchmarks.

Results across board sizes are not directly comparable as measures of model improvement because the board dimensions and candy counts differ.

## Supervised ML Evaluation

### Original Model Evaluation

The initial 5×5 neural network achieved:

- Test MSE: 6.81
- Constant-mean test baseline MSE: 6.40

The model did not outperform the constant-mean baseline on prediction error.

This motivated further investigation into move-ranking accuracy and feature representation.

### Move-Ranking Evaluation

A dedicated evaluation pipeline compares neural network selections against the objective-aware heuristic's highest-scoring legal moves.

The evaluation measures:

- **Optimal-move accuracy:** Percentage of boards where the model selects a heuristic-optimal move, including ties
- **Average heuristic score gap:** Difference between the best available heuristic score and the score of the model's selected move
- **Random baseline:** Expected performance when selecting uniformly from legal moves

Results on 100 reproducibly generated 9×9 board states:

| Metric | Original ML | Target-Mask ML | Random |
|---|---:|---:|---:|
| Optimal-move accuracy | 11.00% | 18.00% | 15.61% |
| Average heuristic score gap | 7.53 | 6.46 | 6.85 |

The target-mask model improved optimal-move accuracy by 7 percentage points and reduced the average heuristic score gap by approximately 14.2% relative to the original model.

The target-mask network also achieved a best validation MSE of **2.7996**, compared with a constant-mean validation baseline of **2.9126**.

These results are encouraging but preliminary. The original model was trained before a correction to six-color heuristic evaluation, so the observed differences cannot yet be attributed solely to the target mask.

A controlled experiment using both encoders and the corrected heuristic is planned.

## AI Development Roadmap

### Phase 1: Game Environment — Implemented

Developed the match-three engine, board generation, swapping, match detection, gravity, cascades, and special candies.

### Phase 2: Heuristic AI Agents — Implemented

Developed greedy and objective-aware agents to evaluate legal moves using matching opportunities and level objectives.

### Phase 3: Objective-Based Gameplay — Implemented

Added configurable move limits, candy collection objectives, objective tracking, win/loss conditions, and board reshuffling.

### Phase 4: Supervised Machine Learning — Implemented, Improving Performance

Developed the supervised training pipeline and neural-network-powered agent.

Current work includes:

- Evaluating move-ranking accuracy
- Improving feature representations
- Comparing neural network variants
- Investigating model generalization
- Improving gameplay performance against heuristic baselines

**Next milestone:** Train a corrected 734-feature baseline to isolate the effect of target-mask encoding, followed by further model evaluation.

### Phase 5: Reinforcement Learning — Planned

Train an agent through simulated gameplay using rewards based on objective completion, scoring, and move efficiency.

Compare its performance against the heuristic and supervised ML agents.

## Technologies

- **Python:** Game environment, algorithms, simulation, and AI development
- **PyTorch:** Neural network training and inference
- **NumPy:** State encoding and feature engineering
- **pandas:** Benchmark analysis and result aggregation
- **Matplotlib:** Performance visualization
- **pytest:** Automated unit testing
- **Git/GitHub:** Version control

## Running the Project

Run the main gameplay simulation from the project root:

```bash
py main.py
```

The desired agent and board configuration can be selected in `main.py`.

### Training

Generate sample supervised training data:

```bash
py generate_dataset.py
```

Run dataset preparation:

```bash
py prepare_dataset.py
```

Train the currently configured neural network:

```bash
py train_model.py
```

The current training configuration uses the 9×9 target-mask representation and saves its best checkpoint to:

```text
models/best_move_model_9x9_target_mask.pth
```

PyTorch checkpoint files (`.pth`) are excluded from Git version control. Models must be trained or supplied separately before loading them for inference.

## Benchmarking

Run the heuristic-agent benchmark:

```bash
py -m benchmarks.benchmark
```

Run the original 9×9 ML-agent gameplay benchmark:

```bash
py -m benchmarks.benchmark_ml
```

Evaluate move-ranking performance:

```bash
py -m benchmarks.evaluate_move_rankings
```

The move-ranking evaluator compares both trained 9×9 models and a random-move baseline on identical generated board states.

The benchmark scripts require their corresponding model checkpoints to be available locally.

Benchmark outputs include gameplay statistics, CSV summaries, and Matplotlib visualizations.

## Testing

The project includes automated tests covering:

- Board initialization and match detection
- Legal and invalid swaps
- Special-candy creation and activation
- Color bomb interactions with configurable candy types
- Power-up chain reactions
- Objective tracking and win/loss conditions
- Heuristic agent decisions
- Configurable board generation
- ML feature encoding and target-mask correctness
- Dataset generation and reproducibility
- Neural network input and output dimensions
- ML agent move selection

Run the test suite from the project root:

```bash
py -m pytest tests/ -v
```

Tests that load trained checkpoints require the corresponding model files to be available locally.

## Project Status

**Actively in Development**

The core game environment, heuristic agents, supervised ML pipeline, and automated evaluation systems are implemented.

Current development focuses on improving the supervised model's decision-making through feature engineering, controlled experiments, and move-ranking evaluation.

Future work will investigate additional learning approaches, including reinforcement learning.

The project serves as an ongoing exploration of AI decision-making, machine learning evaluation, and software engineering through a custom interactive game environment.