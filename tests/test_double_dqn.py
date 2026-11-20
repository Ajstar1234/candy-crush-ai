import pytest
import torch

from dqn_agent import DQNAgent


def test_double_dqn_selects_with_policy_evaluates_with_target():
    agent = DQNAgent(input_size=2, gamma=1.0)

    agent.policy_net = torch.nn.Linear(2, 1, bias=False)
    agent.target_net = torch.nn.Linear(2, 1, bias=False)
    agent.optimizer = torch.optim.SGD(
        agent.policy_net.parameters(), lr=0.01
    )

    with torch.no_grad():
        agent.policy_net.weight.copy_(torch.tensor([[2.0, 1.0]]))
        agent.target_net.weight.copy_(torch.tensor([[5.0, 10.0]]))

    agent.remember(
        state_action_features=[0.0, 0.0],
        reward=0.0,
        next_action_features=[[1.0, 0.0], [0.0, 1.0]],
        done=False
    )

    loss = agent.train_step(batch_size=1)

    assert loss == pytest.approx(4.5)