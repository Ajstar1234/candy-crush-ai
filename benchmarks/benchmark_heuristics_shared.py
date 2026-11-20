from agent import Agent
from benchmarks.evaluate_shared import evaluate_agent, summarize_results


def main():
    agent = Agent()

    agents = [
        ("Greedy", lambda board, game: agent.choose_move(board)),
        ("Objective-Aware", agent.choose_move)
    ]

    for name, choose_move in agents:
        results = evaluate_agent(
            choose_move,
            num_games=500,
            seed_start=30000
        )

        summary = summarize_results(results)

        print(f"\n{name}")
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