# Candy Crush AI

This project focuses on developing a match-three game environment and an AI agent that selects moves under configurable move limits, with the long-term goal of completing levels with specific objectives.

## Project Overview

Rather than simply recreating Candy Crush, the goal is to explore different AI approaches, beginning with heuristic-based decision-making and progressing toward machine learning and reinforcement learning.

## Current Features

- Object-oriented Python architecture with separate game, board, candy, and agent classes
- Randomized 5×5 game board generation
- Horizontal and vertical match detection
- Candy swapping and legal move validation
- Automatic match removal, gravity, and board refilling
- Cascading matches after moves
- Automated generation of available legal moves
- Heuristic AI agent that evaluates moves based on immediate matches
- Automated move selection using heuristic scores
- Scoring system awarding points for candies cleared, including cascades
- Score and turn tracking with win/loss conditions
- Configurable move limits and target scores
- Refactored game logic with modular functions for match detection, power-up creation, and activation
- Automated unit tests using pytest for board generation, match detection, and multiple-match scenarios
- Initial special candy implementations:
  - Horizontal and vertical striped candies
  - Wrapped candies
  - Color bombs

## Planned Features

- Additional level objectives, including clearing specific candy types
- Complete and test special candy mechanics:
  - Striped candy activation and chain reactions
  - Wrapped candy explosions and interactions
  - Color bomb combinations and interactions
  - Fish candies
- Comprehensive automated tests for special candy mechanics
- Improved heuristic evaluation considering objectives, power-ups, and future moves
- Game-state representation for machine learning
- Gameplay data collection and performance evaluation
- Neural network-based move evaluation
- Reinforcement learning agent trained through simulated gameplay
- Comparison of heuristic and learned agent performance across multiple levels

## AI Development Roadmap

### Phase 1: Game Environment — Implemented

Develop the core mechanics required for automated gameplay, including board generation, match detection, swapping, gravity, refilling, and cascading matches.

### Phase 2: Heuristic AI Agent — In Progress

Develop an agent capable of identifying valid moves, evaluating potential matches, and selecting moves based on heuristic scores.

Current progress includes immediate match evaluation, automated move selection, and integration with the game's scoring and win conditions.

### Phase 3: Objective-Based Gameplay — Planned

Implement multiple level objectives and move constraints. Improve the agent's decision-making to prioritize completing objectives rather than only maximizing immediate matches.

### Phase 4: Machine Learning — Planned

Collect gameplay data and experiment with supervised learning models that predict move quality from board states and available actions.

### Phase 5: Reinforcement Learning — Planned

Train an agent through simulated gameplay using rewards based on scoring, objective completion, and move efficiency.

Evaluate performance against the heuristic-based agent.

## Technologies

- Python
- Object-Oriented Programming
- Heuristic Search and Decision-Making
- Artificial Intelligence
- Machine Learning / Reinforcement Learning (planned)
- pytest (Unit Testing)

## Testing

The project uses pytest for automated unit testing to validate core game mechanics.

Current test coverage includes:
- Board initialization and dimensions
- Horizontal match detection
- Vertical match detection
- Detection of multiple matches on the same board

Run tests from the project root: 

```bash
py -m pytest tests/ -v
```

Additional tests for power-up creation, activation, and game-state transitions are planned.

## Project Status

**Actively in Development**

The core match-three environment, basic heuristic AI agent, scoring system, and score-based victory conditions are implemented.

Current development focuses on refining special candy mechanics, refactoring game logic for maintainability, expanding automated test coverage, and preparing for objective-based gameplay and future machine learning experiments.
