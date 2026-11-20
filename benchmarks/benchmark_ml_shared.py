from ml_agent import MLAgent
from benchmarks.evaluate_shared import evaluate_agent, summarize_results


def main():
    agent = MLAgent(
        input_size=815,
        checkpoint_path="models/best_move_model_9x9_target_mask.pth",
        use_target_mask=True
    )

    results = evaluate_agent(
        agent.choose_move,
        num_games=500,
        seed_start=30000
    )

    summary = summarize_results(results)

    print("\nSupervised ML - Target Mask")
    print(f"Games evaluated: {summary['games']}")
    print(f"Wins: {summary['wins']}")
    print(f"Win rate: {summary['win_rate'] * 100:.2f}%")
    print(f"Average reward: {summary['average_reward']:.2f}")
    print(
        f"95% Confidence Interval: "
        f"{summary['ci_low'] * 100:.2f}% - "
        f"{summary['ci_high'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()