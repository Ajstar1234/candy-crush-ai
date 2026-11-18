import torch
import numpy as np

from dqn_agent import DQNAgent
from rl_environment import RLEnvironment
from ml_features import encode_rl_state_move


def test_dqn_network_initialization():
    agent = DQNAgent(input_size=817)

    assert agent.policy_net is not agent.target_net
    assert agent.target_net.training is False

    for policy_param, target_param in zip(
        agent.policy_net.parameters(),
        agent.target_net.parameters()
    ):
        assert torch.equal(policy_param, target_param)
        assert policy_param is not target_param
        assert target_param.requires_grad is False

def test_dqn_exploration(monkeypatch):
    env = RLEnvironment()
    env.reset(seed=42)

    agent = DQNAgent(epsilon=1.0)

    valid_moves = env.get_valid_moves()

    monkeypatch.setattr(
        "dqn_agent.random.choice",
        lambda moves: moves[-1]
    )

    selected_move = agent.choose_move(env.board, env.game)

    assert selected_move == valid_moves[-1]

def test_dqn_exploitation():
    env = RLEnvironment()
    env.reset(seed=42)

    agent = DQNAgent(epsilon=0.0)

    valid_moves = env.get_valid_moves()

    q_values = []

    with torch.no_grad():
        for move in valid_moves:
            features = encode_rl_state_move(
                env.board, move, env.game
            )

            features_tensor = torch.tensor(
                features,
                dtype=torch.float32
            ).unsqueeze(0)

            q_value = agent.policy_net(features_tensor).item()
            q_values.append(q_value)

    expected_move = valid_moves[q_values.index(max(q_values))]

    selected_move = agent.choose_move(env.board, env.game)

    assert selected_move == expected_move

def test_dqn_remember():
    agent = DQNAgent()

    features = np.ones(817, dtype=np.float32)
    next_features = np.zeros((3, 817), dtype=np.float32)

    agent.remember(features, 5.0, next_features, False)

    assert len(agent.memory) == 1

    stored_features, reward, stored_next, done = agent.memory[0]

    assert stored_features.shape == (817,)
    assert stored_next.shape == (3, 817)
    assert reward == 5.0
    assert done is False

    # Verify stored arrays are independent copies.
    features[0] = 999
    next_features[0, 0] = 999

    assert stored_features[0] == 1.0
    assert stored_next[0, 0] == 0.0

def test_dqn_train_step():
    torch.manual_seed(42)

    agent = DQNAgent()

    for _ in range(4):
        features = np.ones(817, dtype=np.float32)
        next_features = np.empty((0, 817), dtype=np.float32)

        agent.remember(
            features,
            reward=10.0,
            next_action_features=next_features,
            done=True
        )

    policy_before = [
        param.clone().detach()
        for param in agent.policy_net.parameters()
    ]

    target_before = [
        param.clone().detach()
        for param in agent.target_net.parameters()
    ]

    loss = agent.train_step(batch_size=4)

    assert isinstance(loss, float)
    assert loss >= 0

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            policy_before, agent.policy_net.parameters()
        )
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(
            target_before, agent.target_net.parameters()
        )
    )

def test_dqn_target_network_update():
    torch.manual_seed(42)
    agent = DQNAgent()

    for _ in range(4):
        features = np.ones(817, dtype=np.float32)
        next_features = np.empty((0, 817), dtype=np.float32)

        agent.remember(features, 10.0, next_features, True)

    agent.train_step(batch_size=4)

    # Networks should differ after training the policy network.
    assert any(
        not torch.equal(p, t)
        for p, t in zip(
            agent.policy_net.parameters(),
            agent.target_net.parameters()
        )
    )

    agent.update_target_network()

    # After synchronization, their weights should match.
    assert all(
        torch.equal(p, t)
        for p, t in zip(
            agent.policy_net.parameters(),
            agent.target_net.parameters()
        )
    )

def test_dqn_epsilon_decay():
    agent = DQNAgent(
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.2
    )

    agent.decay_epsilon()
    assert agent.epsilon == 0.5

    agent.decay_epsilon()
    assert agent.epsilon == 0.25

    agent.decay_epsilon()
    assert agent.epsilon == 0.2

    agent.decay_epsilon()
    assert agent.epsilon == 0.2