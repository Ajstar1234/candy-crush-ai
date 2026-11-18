import torch

from train_model import pairwise_ranking_loss


def test_pairwise_ranking_loss():
    targets = torch.tensor([3.0, 9.0, 5.0])

    correct_predictions = torch.tensor([1.0, 3.0, 2.0])
    incorrect_predictions = torch.tensor([3.0, 1.0, 2.0])

    correct_loss = pairwise_ranking_loss(
        correct_predictions, targets
    )

    incorrect_loss = pairwise_ranking_loss(
        incorrect_predictions, targets
    )

    assert correct_loss < incorrect_loss


def test_pairwise_ranking_loss_equal_scores():
    targets = torch.tensor([5.0, 5.0, 5.0])
    predictions = torch.tensor([1.0, 2.0, 3.0])

    loss = pairwise_ranking_loss(predictions, targets)

    assert loss.item() == 0.0