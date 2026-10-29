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

        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {average_loss:.4f}")

if __name__ == "__main__":
    train_model()
            