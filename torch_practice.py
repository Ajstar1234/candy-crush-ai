import numpy as np
import torch

from prepare_dataset import prepare_datasets
from torch.utils.data import TensorDataset, DataLoader

numbers = np.array([1, 2, 3, 4, 5])

numbers_tensor = torch.tensor(numbers, dtype=torch.float32)

print(numbers_tensor)
print(numbers_tensor.shape)
print(numbers_tensor.dtype)

X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

print(X_train_tensor.shape)
print(y_train_tensor.shape)

print(X_train_tensor.dtype)
print(y_train_tensor.dtype)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

feature, label = train_dataset[0]

print(feature.shape)
print(label)
print(len(train_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

for batch in train_loader:
    X_batch, y_batch = batch
    print(X_batch.shape)
    print(y_batch.shape)
    break

predictions = torch.tensor([5.0, 7.0, 3.0])
targets = torch.tensor([9.0, 6.0, 3.0])

mse_loss = torch.nn.MSELoss()
loss = mse_loss(predictions, targets)

print(loss)