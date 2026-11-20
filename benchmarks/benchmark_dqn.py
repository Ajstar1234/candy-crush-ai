import torch

from dqn_agent import DQNAgent
from benchmarks.evaluate_shared import evaluate_agent, summarize_results


def evaluate_dqn(
    num_games=20,
    seed_start=20000,
    epsilon=0.0,
    checkpoint_path="models/dqn_25episodes.pth",
    use_normalized_features=False
):
    agent = DQNAgent(
        epsilon=epsilon,
        use_normalized_features=use_normalized_features
    )

    agent.policy_net.load_state_dict(
        torch.load(
            checkpoint_path,
            weights_only=True,
            map_location="cpu"
        )
    )

    agent.policy_net.eval()

    results = evaluate_agent(
        agent.choose_move,
        num_games=num_games,
        seed_start=seed_start
    )

    return summarize_results(results)

if __name__ == "__main__":
    num_games = 500

    models = [
        ("Double DQN - 100", "models/double_dqn_long_seed42_ep100.pth", 0.0, False),
        ("Double DQN - 200", "models/double_dqn_long_seed42_ep200.pth", 0.0, False),
        ("Double DQN - 300", "models/double_dqn_long_seed42_ep300.pth", 0.0, False),
        ("Double DQN - 400", "models/double_dqn_long_seed42_ep400.pth", 0.0, False),
        ("Double DQN - 500", "models/double_dqn_long_seed42_ep500.pth", 0.0, False),

        ("Random", "models/dqn_seed42_ep0.pth", 1.0, False)
    ]

    for model_name, checkpoint_path, epsilon, normalized in models:
        summary = evaluate_dqn(
            num_games=num_games,
            seed_start=70000,
            epsilon=epsilon,
            checkpoint_path=checkpoint_path,
            use_normalized_features=normalized
        )

        print(f"\n{model_name}")
        print(f"Games evaluated: {summary['games']}")
        print(f"Total wins: {summary['wins']}")
        print(f"Win rate: {summary['win_rate'] * 100:.2f}%")
        print(f"Average reward: {summary['average_reward']:.2f}")
        print(
            f"95% Confidence Interval: "
            f"{summary['ci_low'] * 100:.2f}% - "
            f"{summary['ci_high'] * 100:.2f}%"
        )