# Candy Crush AI

An AI-focused Candy Crush-style game environment built in Python, designed to explore intelligent decision-making, game-state evaluation, and eventually machine learning and reinforcement learning.

## Project Overview

This project focuses on developing a match-three game environment and an AI agent capable of selecting moves to complete levels with specific objectives and move constraints.

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
- Score and turn tracking
- Configurable target score with win detection
- Basic 10-move level limit and win/loss reporting
- Initial power-up implementations (currently being refined)

## Planned Features

- Configurable move limits and additional level settings
- Fully functional special candies and power-ups
  - Striped candies
  - Wrapped candies
  - Color bombs
  - Fish candies
- Objective-based levels, including clearing specific candy types
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

## Project Status

**Actively in Development**

The core match-three environment, basic heuristic AI agent, scoring system, and score-based victory conditions are implemented.

Current development focuses on completing special candy mechanics, expanding level objectives, and preparing the environment for machine learning and reinforcement learning experiments.
