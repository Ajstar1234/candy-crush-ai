import numpy as np
import torch
import random

from dqn_agent import DQNAgent
from rl_environment import RLEnvironment
from ml_features import encode_rl_state_move, encode_normalized_rl_state_move

def run_training_episode(env, agent, seed=None):
    env.reset(seed=seed)

    encode_features = (
        encode_normalized_rl_state_move
        if agent.use_normalized_features
        else encode_rl_state_move
    )

    total_reward = 0.0

    training_losses = []

    while not env.done and not env.game.is_game_over():
        move = agent.choose_move(env.board, env.game)

        if move is None:
            raise RuntimeError("No legal moves in an active episode.")

        features = encode_features(env.board, move, env.game)

        observation, reward, done, info = env.step(move)

        if done:
            next_action_features = np.empty(
                (0, len(features)), dtype=np.float32
            )
        else:
            next_moves = env.get_valid_moves()

            next_action_features = np.array([
                encode_features(env.board, next_move, env.game)
                for next_move in next_moves
            ], dtype=np.float32)

        agent.remember(
            state_action_features=features,
            reward=reward,
            next_action_features=next_action_features,
            done=done
        )

        train_loss = agent.train_step(batch_size=32)

        if train_loss is not None:
            training_losses.append(train_loss)

        total_reward += reward

    return total_reward, training_losses

def train_dqn(num_episodes=5):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = RLEnvironment()
    agent = DQNAgent(
        learning_rate=0.00005,
        use_normalized_features=False
    )

    torch.save(
        agent.policy_net.state_dict(),
        "models/double_dqn_long_seed42_ep0.pth"
    )

    for episode in range(num_episodes):
        reward, losses = run_training_episode(
            env,
            agent,
            seed=10000 + episode
        )

        agent.decay_epsilon()

        if (episode + 1) % 5 == 0:
            agent.update_target_network()

        average_loss = (
            sum(losses) / len(losses)
            if losses else 0.0
        )

        if (episode + 1) in (100, 200, 300, 400, 500):
            torch.save(
                agent.policy_net.state_dict(),
                f"models/double_dqn_long_seed42_ep{episode + 1}.pth"
            )
    

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Reward: {reward:.2f} | "
            f"Won: {env.game.is_won()} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Memory: {len(agent.memory)} | "
            f"Train Updates: {len(losses)} | "
            f"Avg Loss: {average_loss:.4f}"
        )

    torch.save(
        agent.policy_net.state_dict(),
        "models/double_dqn_long_500episodes.pth"
    )

    print("DQN checkpoint saved.")

    return agent

if __name__ == "__main__":
    agent = train_dqn(num_episodes=500)