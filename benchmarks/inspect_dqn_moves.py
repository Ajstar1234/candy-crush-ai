import torch

from agent import Agent
from dqn_agent import DQNAgent
from ml_features import encode_rl_state_move
from rl_environment import RLEnvironment


env = RLEnvironment()
env.reset(seed=80000)

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
results = []

with torch.no_grad():
    for move in env.get_valid_moves():
        features = encode_rl_state_move(env.board, move, env.game)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        q_value = dqn.policy_net(tensor).item()
        heuristic_score = heuristic.evaluate_move(env.board, move, env.game)

        results.append((move, q_value, heuristic_score))

by_q = sorted(results, key=lambda x: x[1], reverse=True)
by_heuristic = sorted(results, key=lambda x: x[2], reverse=True)

print("Target candy:", env.game.target_candy_type)
print("Valid moves:", len(results))

print("\nTop 5 moves ranked by Double DQN:")
for move, q, score in by_q[:5]:
    print(f"Move: {move} | Q-value: {q:.3f} | Heuristic: {score}")

print("\nTop 5 moves ranked by objective-aware heuristic:")
for move, q, score in by_heuristic[:5]:
    print(f"Move: {move} | Q-value: {q:.3f} | Heuristic: {score}")

print("\nSame preferred move:", by_q[0][0] == by_heuristic[0][0])