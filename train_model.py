import torch

from ml_model import MoveEvaluationModel
from prepare_dataset import prepare_datasets
from torch.utils.data import TensorDataset, DataLoader

def pairwise_ranking_loss(predictions, targets):
    better_moves = targets[:, None] > targets[None, :]

    if not better_moves.any():
        return predictions.sum() * 0.0

    predicted_differences = (
        predictions[:, None] - predictions[None, :]
    )

    loss = torch.nn.functional.softplus(
        -predicted_differences[better_moves]
    ).mean()

    return loss

def train_model(
    use_target_mask=True,
    checkpoint_path="models/best_move_model_9x9_target_mask.pth"
    ):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(
        rows=9,
        cols=9,
        num_candy_types=6,
        use_target_mask=use_target_mask,
        return_group_sizes=False
    )

    torch.manual_seed(42)

    input_size = X_train.shape[1]
    model = MoveEvaluationModel(input_size)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    best_val_loss = float("inf")

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

        if val_average_loss < best_val_loss:
            best_val_loss = val_average_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved! Validation MSE: {best_val_loss:.4f}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Val Loss: {val_average_loss:.4f}"
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            weights_only=True,
            map_location="cpu"
        )
    )

    model.eval()

    with torch.no_grad():
        sample_features = X_val_tensor[:10]
        actual_scores = y_val_tensor[:10]

        predicted_scores = model(sample_features).squeeze(-1)

        for i in range(10):
            print(f"Actual: {actual_scores[i]:.2f}, Predicted: {predicted_scores[i]:.2f}")

def train_ranking_model():
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_groups, val_groups, test_groups
    ) = prepare_datasets(
        rows=9,
        cols=9,
        num_candy_types=6,
        use_target_mask=True,
        return_group_sizes=True
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_feature_groups = torch.split(X_train_tensor, train_groups)
    train_target_groups = torch.split(y_train_tensor, train_groups)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    val_feature_groups = torch.split(X_val_tensor, val_groups)
    val_target_groups = torch.split(y_val_tensor, val_groups)

    torch.manual_seed(42)

    input_size = X_train.shape[1]

    model = MoveEvaluationModel(input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = float("inf")
    checkpoint_path = "models/best_move_model_9x9_ranking_lr0001.pth"

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        trained_groups = 0

        shuffled_indices = torch.randperm(len(train_feature_groups))

        for index in shuffled_indices:
            X_group = train_feature_groups[index]
            y_group = train_target_groups[index]

            if len(torch.unique(y_group)) < 2:
                continue

            optimizer.zero_grad()

            predictions = model(X_group).squeeze(-1)
            loss = pairwise_ranking_loss(predictions, y_group)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            trained_groups += 1

        if trained_groups == 0:
            raise ValueError("No valid ranking groups in training data")

        average_train_loss = total_train_loss / trained_groups

        model.eval()

        total_val_loss = 0
        evaluated_groups = 0

        with torch.no_grad():
            for X_group, y_group in zip(val_feature_groups, val_target_groups):

                if len(torch.unique(y_group)) < 2:
                    continue

                predictions = model(X_group).squeeze(-1)
                loss = pairwise_ranking_loss(predictions, y_group)

                total_val_loss += loss.item()
                evaluated_groups += 1

        if evaluated_groups == 0:
            raise ValueError("No valid ranking groups in validation data")

        average_val_loss = total_val_loss / evaluated_groups

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss

            torch.save(model.state_dict(), checkpoint_path)

            print(f"New best ranking model saved! Validation Loss: {best_val_loss:.4f}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Ranking Loss: {average_train_loss:.4f} | "
            f"Val Ranking Loss: {average_val_loss:.4f}"
        )

    print(f"Training board states: {len(train_feature_groups)}")
    print(f"Total training moves: {len(X_train_tensor)}")

if __name__ == "__main__":
    train_ranking_model()
