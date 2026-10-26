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
- Objective-based levels requiring collection of specific candy types
- Tracking of collected candies across matches, cascades, and power-ups
- Board reshuffling when no valid moves are available
- Automated tests for objective completion, power-ups, and cascades
- Heuristic AI agent that evaluates valid moves based on immediate matches and level objectives
- Automated move selection using heuristic scores

## Planned Features

- Additional level objectives and more complex win conditions
- Advanced special candy combinations and interactions
- Improved heuristic evaluation considering future moves
- Automated gameplay benchmarking and performance analysis
- Game-state representation for machine learning
- Neural network-based move evaluation using PyTorch
- Reinforcement learning through simulated gameplay
- Comparison of heuristic and learned agent performance
- Optional graphical user interface

## AI Development Roadmap

### Phase 1: Game Environment — Implemented

Develop the core mechanics required for automated gameplay, including board generation, match detection, swapping, gravity, refilling, and cascading matches.

### Phase 2: Heuristic AI Agent — Implemented

Developed an AI agent capable of identifying valid moves,
evaluating matches, and selecting moves based on heuristic scores.

The agent supports objective-aware evaluation, prioritizing
candies needed to complete collection objectives.

### Phase 3: Objective-Based Gameplay — Implemented

Implemented candy collection objectives, configurable move limits,
objective tracking, and win/loss conditions.

Added support for board reshuffling and automated tests
covering objective tracking, power-ups, and cascades.

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
- pytest (Automated Unit Testing)

### Planned Technologies
- NumPy
- pandas
- PyTorch
- Reinforcement Learning

## Testing

The project uses pytest to validate game mechanics and AI behavior.

Current tests cover:
- Board initialization and match detection
- Valid and invalid swaps
- Special candy creation and activation
- Power-up chain reactions
- Color bomb activation
- Board reshuffling
- Candy collection through cascades
- Objective-based win/loss conditions
- Agent move evaluation and objective prioritization

Run tests from the project root:

```bash
py -m pytest tests/ -v
```

## Project Status

**Actively in Development**

The core match-three environment is implemented, including candy swapping, match detection, cascades, special candies, scoring, and objective-based win/loss conditions.

The heuristic AI agent evaluates available moves and prioritizes candies required for collection objectives. Automated tests cover core game mechanics, special candy activation, objective tracking, and board reshuffling.

Current development focuses on improving gameplay reliability and building an automated benchmarking system to evaluate agent performance across multiple games.

Future development will introduce gameplay data collection, supervised machine learning with PyTorch, and reinforcement learning.
