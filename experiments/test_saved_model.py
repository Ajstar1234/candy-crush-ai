import torch
from ml_model import MoveEvaluationModel
from prepare_dataset import prepare_datasets

model = MoveEvaluationModel()
model.load_state_dict(
    torch.load("models/best_move_model.pth", weights_only=True, map_location="cpu")
)

X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

loss_fn = torch.nn.MSELoss()

model.eval()

with torch.no_grad():
    X_pred = model(X_test_tensor).squeeze(-1)

    test_loss = loss_fn(X_pred, y_test_tensor)

    print("Test MSE:", test_loss.item())

    mean_train_score = torch.tensor(y_train, dtype=torch.float32).mean()

    baseline_loss = ((y_test_tensor - mean_train_score) ** 2).mean()

    print("Baseline Loss:", baseline_loss.item())


