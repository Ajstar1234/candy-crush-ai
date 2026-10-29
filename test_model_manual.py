import torch
from ml_model import MoveEvaluationModel

model = MoveEvaluationModel()

sample_input = torch.randn(1, 55)
prediction = model(sample_input)

print(prediction)
print(prediction.shape)