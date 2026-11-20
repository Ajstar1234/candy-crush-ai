import torch

from agent import Agent
from dqn_agent import DQNAgent
from rl_environment import RLEnvironment


dqn = DQNAgent(epsilon=0.0, use_normalized_features=False)

dqn.policy_net.load_state_dict(
    torch.load(
        "models/double_dqn_long_seed42_ep300.pth",
        weights_only=True,
        map_location="cpu"
    )
)
dqn.policy_net.eval()

heuristic = Agent()

total_boards = 0
optimal_choices = 0
total_score_gap = 0
total_dqn_score = 0
total_best_score = 0
total_random_score = 0
expected_random_optimal = 0

for seed in range(90000, 90100):
    env = RLEnvironment()
    env.reset(seed=seed)

    valid_moves = env.get_valid_moves()

    if not valid_moves:
        continue

    dqn_move = dqn.choose_move(env.board, env.game)

    move_scores = {
        move: heuristic.evaluate_move(env.board, move, env.game)
        for move in valid_moves
    }

    best_score = max(move_scores.values())
    dqn_score = move_scores[dqn_move]

    scores = list(move_scores.values())

    total_random_score += sum(scores) / len(scores)

    expected_random_optimal += (
        sum(score == best_score for score in scores)
        / len(scores)
    )

    if dqn_score == best_score:
        optimal_choices += 1

    total_score_gap += best_score - dqn_score
    total_dqn_score += dqn_score
    total_best_score += best_score
    total_boards += 1

print(f"Boards evaluated: {total_boards}")
print(f"Heuristic-optimal DQN choices: {optimal_choices}")
print(f"Optimal-move accuracy: {optimal_choices / total_boards:.2%}")
print(f"Average DQN heuristic score: {total_dqn_score / total_boards:.2f}")
print(f"Average best heuristic score: {total_best_score / total_boards:.2f}")
print(f"Average heuristic score gap: {total_score_gap / total_boards:.2f}")
print(
    f"Expected random optimal-move accuracy: "
    f"{expected_random_optimal / total_boards:.2%}"
)

print(
    f"Expected random heuristic score: "
    f"{total_random_score / total_boards:.2f}"
)

print(
    f"DQN score advantage over random: "
    f"{(total_dqn_score - total_random_score) / total_boards:.2f}"
)