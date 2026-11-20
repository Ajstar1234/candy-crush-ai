import random
import torch
import csv

from agent import Agent
from ml_agent import MLAgent
from dqn_agent import DQNAgent
from benchmarks.evaluate_shared import evaluate_agent, summarize_results


def main():
    heuristic = Agent()

    ml_agent = MLAgent(
        input_size=815,
        checkpoint_path="models/best_move_model_9x9_target_mask.pth",
        use_target_mask=True
    )

    dqn_agent = DQNAgent(epsilon=0.0)

    dqn_agent.policy_net.load_state_dict(
        torch.load(
            "models/dqn_seed42_ep50.pth",
            weights_only=True,
            map_location="cpu"
        )
    )
    dqn_agent.policy_net.eval()

    agents = [
        ("Random", lambda board, game: random.choice(board.get_valid_moves())),
        ("Greedy", lambda board, game: heuristic.choose_move(board)),
        ("Objective-Aware", heuristic.choose_move),
        ("Supervised ML", ml_agent.choose_move),
        ("DQN - 50 Episodes", dqn_agent.choose_move)
    ]

    all_results = {}

    for name, choose_move in agents:
        results = evaluate_agent(
            choose_move,
            num_games=500,
            seed_start=30000
        )

        all_results[name] = results
        summary = summarize_results(results)

        print(f"\n{name}")
        print(f"Win rate: {summary['win_rate'] * 100:.2f}%")
        print(f"Average reward: {summary['average_reward']:.2f}")
        print(
            f"95% CI: {summary['ci_low'] * 100:.2f}% - "
            f"{summary['ci_high'] * 100:.2f}%"
        )

    with open(
        "benchmarks/all_agents_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "agent",
                "seed",
                "won",
                "reward",
                "turns",
                "target_candy_type"
            ]
        )

        writer.writeheader()

        for name, results in all_results.items():
            for result in results:
                writer.writerow({
                    "agent": name,
                    **result
                })

    print("\nResults saved to benchmarks/all_agents_results.csv")

    return all_results


if __name__ == "__main__":
    main()