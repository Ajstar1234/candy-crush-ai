import math

from rl_environment import RLEnvironment


def evaluate_agent(choose_move, num_games=20, seed_start=30000, **env_settings):
    results = []

    for game_index in range(num_games):
        seed = seed_start + game_index

        env = RLEnvironment(**env_settings)
        env.reset(seed=seed)

        episode_reward = 0.0

        while not env.done and not env.game.is_game_over():
            move = choose_move(env.board, env.game)

            if move is None:
                raise RuntimeError("No legal moves in an active episode.")

            _, reward, _, _ = env.step(move)
            episode_reward += reward

        results.append({
            "seed": seed,
            "won": env.game.is_won(),
            "reward": episode_reward,
            "turns": env.game.turn,
            "target_candy_type": env.game.target_candy_type
        })

    return results

def summarize_results(results):
    n = len(results)

    if n == 0:
        raise ValueError("Cannot summarize empty results.")

    wins = sum(result["won"] for result in results)
    win_rate = wins / n
    average_reward = sum(result["reward"] for result in results) / n

    z = 1.96
    denominator = 1 + z**2 / n

    center = (win_rate + z**2 / (2 * n)) / denominator

    margin = (
        z * math.sqrt(
            win_rate * (1 - win_rate) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    return {
        "games": n,
        "wins": wins,
        "win_rate": win_rate,
        "average_reward": average_reward,
        "ci_low": center - margin,
        "ci_high": center + margin
    }