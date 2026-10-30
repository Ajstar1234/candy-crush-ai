import torch

from ml_model import MoveEvaluationModel
from prepare_dataset import prepare_datasets
from torch.utils.data import TensorDataset, DataLoader

def train_model():
    model = MoveEvaluationModel()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    mean_train_score = y_train_tensor.mean()
    baseline_loss = ((y_val_tensor - mean_train_score) ** 2).mean()
    print(f"Baseline Validation MSE: {baseline_loss.item():.4f}")

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()        

            predictions = model(X_batch).squeeze(-1)
            loss = loss_fn(predictions, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.shape[0]

        average_loss = total_loss / len(train_dataset)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch).squeeze(-1)
                loss = loss_fn(predictions, y_batch)
                val_total_loss += loss.item() * X_batch.shape[0]

        val_average_loss = val_total_loss / len(val_dataset)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Val Loss: {val_average_loss:.4f}"
        )

    model.eval()

    with torch.no_grad():
        sample_features = X_val_tensor[:10]
        actual_scores = y_val_tensor[:10]

        predicted_scores = model(sample_features).squeeze(-1)

        for i in range(10):
            print(f"Actual: {actual_scores[i]:.2f}, Predicted: {predicted_scores[i]:.2f}")

if __name__ == "__main__":
    train_model()
