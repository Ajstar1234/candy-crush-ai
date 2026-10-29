import torch
from ml_model import MoveEvaluationModel


def test_model_output_shape():
    model = MoveEvaluationModel()

    sample_input = torch.randn(10, 55)
    output = model(sample_input)

    assert output.shape == (10, 1)