import torch
import random
import numpy as np

from ml_features import encode_rl_state_move
from ml_model import MoveEvaluationModel
from collections import deque


class DQNAgent:
    def __init__(
        self,
        input_size=817,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05

    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.policy_net = MoveEvaluationModel(input_size)

        self.target_net = MoveEvaluationModel(input_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()

        self.target_net.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.memory = deque(maxlen=2000)

    def choose_move(self, board, game):
        valid_moves = board.get_valid_moves()

        if not valid_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            with torch.no_grad():
                best_move = None
                best_value = float('-inf')

                for move in valid_moves:
                    features = encode_rl_state_move(board, move, game)
                    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                    value = self.policy_net(features).item()

                    if value > best_value:
                        best_value = value
                        best_move = move

                return best_move

    def remember(self, state_action_features, reward, next_action_features, done):
        state_action_features = np.array(
            state_action_features, dtype=np.float32, copy=True
        )

        next_action_features = np.array(
            next_action_features, dtype=np.float32, copy=True
        )

        self.memory.append((
            state_action_features,
            float(reward),
            next_action_features,
            bool(done)
        ))

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        state_action_batch = []
        target_q_batch = []

        for state_action_features, reward, next_action_features, done in batch:

            target_q = float(reward)

            with torch.no_grad():
                if not done and len(next_action_features) > 0:
                    next_action_features_tensor = torch.tensor(
                        next_action_features,
                        dtype=torch.float32
                    )

                    next_q_values = self.target_net(
                        next_action_features_tensor
                    ).squeeze(-1)

                    max_next_q = next_q_values.max().item()

                    target_q += self.gamma * max_next_q

            state_action_batch.append(state_action_features)
            target_q_batch.append(target_q)

        state_action_tensor = torch.tensor(
            np.stack(state_action_batch),
            dtype=torch.float32
        )

        target_q_tensor = torch.tensor(
            target_q_batch,
            dtype=torch.float32
        )

        predicted_q_values = self.policy_net(
            state_action_tensor
        ).squeeze(-1)

        loss = torch.nn.functional.smooth_l1_loss(
            predicted_q_values,
            target_q_tensor
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)