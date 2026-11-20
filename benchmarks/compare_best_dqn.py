import torch

from agent import Agent
from dqn_agent import DQNAgent
from benchmarks.evaluate_shared import evaluate_agent, summarize_results


dqn = DQNAgent(epsilon=0.0, use_normalized_features=False)

dqn.policy_net.load_state_dict(
    torch.load(
        "models/double_dqn_long_seed42_ep300.pth",
        weights_only=True,
        map_location="cpu"
    )
)
dqn.policy_net.eval()

objective_agent = Agent()

agents = [
    ("Double DQN - 300", dqn.choose_move),
    ("Objective-Aware", objective_agent.choose_move)
]

results_by_agent = {}

for name, choose_move in agents:
    results = evaluate_agent(
        choose_move,
        num_games=500,
        seed_start=80000
    )

    results_by_agent[name] = results
    summary = summarize_results(results)

    print(f"\n{name}")
    print(f"Wins: {summary['wins']}/{summary['games']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Average reward: {summary['average_reward']:.2f}")

dqn_results = results_by_agent["Double DQN - 300"]
objective_results = results_by_agent["Objective-Aware"]

dqn_only = sum(
    d["won"] and not h["won"]
    for d, h in zip(dqn_results, objective_results)
)

objective_only = sum(
    h["won"] and not d["won"]
    for d, h in zip(dqn_results, objective_results)
)

print(f"\nDQN-only wins: {dqn_only}")
print(f"Objective-Aware-only wins: {objective_only}")