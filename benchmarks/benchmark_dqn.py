import torch

from dqn_agent import DQNAgent
from rl_environment import RLEnvironment


def evaluate_dqn(
        num_games=20, 
        seed_start=20000, 
        epsilon=0.0,
        checkpoint_path="models/dqn_25episodes.pth"
        ):
    env = RLEnvironment()

    agent = DQNAgent(epsilon=epsilon)

    agent.policy_net.load_state_dict(
        torch.load(
            checkpoint_path,
            weights_only=True,
            map_location="cpu"
        )
    )

    agent.policy_net.eval()

    total_wins = 0
    total_reward = 0.0

    for game_index in range(num_games):
        env.reset(seed=seed_start + game_index)

        while not env.done and not env.game.is_game_over():
            move = agent.choose_move(env.board, env.game)

            if move is None:
                raise RuntimeError("No legal moves in an active episode.")

            _, reward, done, _ = env.step(move)
            total_reward += reward

        if env.game.is_won():
            total_wins += 1

    return total_wins, total_reward

if __name__ == "__main__":
    num_games = 500

    models = [
        ("DQN - Untrained", "models/dqn_seed42_ep0.pth", 0.0),
        ("DQN - 50 Episodes", "models/dqn_seed42_ep50.pth", 0.0),
        ("DQN - 100 Episodes", "models/dqn_seed42_ep100.pth", 0.0),
        ("Random Agent", "models/dqn_seed42_ep0.pth", 1.0)
    ]

    for model_name, checkpoint_path, epsilon in models:
        total_wins, total_reward = evaluate_dqn(
            num_games=num_games,
            seed_start=30000,
            epsilon=epsilon,
            checkpoint_path=checkpoint_path
        )

        print(f"\n{model_name}")
        print(f"Games evaluated: {num_games}")
        print(f"Total wins: {total_wins}")
        print(f"Win rate: {total_wins / num_games * 100:.2f}%")
        print(f"Average reward: {total_reward / num_games:.2f}")