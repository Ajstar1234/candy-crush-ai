import torch
from ml_model import MoveEvaluationModel


def test_model_output_shape():
    model = MoveEvaluationModel()

    sample_input = torch.randn(10, 180)
    output = model(sample_input)

    assert output.shape == (10, 1)


def test_model_dynamic_input():
    model = MoveEvaluationModel(input_size=734)

    sample_input = torch.randn(4, 734)
    output = model(sample_input)

    assert output.shape == (4, 1)