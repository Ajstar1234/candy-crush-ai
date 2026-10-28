import numpy as np
from generate_dataset import generate_training_data


def test_generate_training_data():
    X, y = generate_training_data(num_games=10)

    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0
    assert X.shape[1] == 55

def test_generate_training_data_reproducible():
    X1, y1 = generate_training_data(num_games=10, seed=42)
    X2, y2 = generate_training_data(num_games=10, seed=42)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)

def test_generate_training_data_multiple_states():
    X, y = generate_training_data(num_games=10, seed=42)

    board_states = X[:, :50]

    unique_boards = np.unique(board_states, axis=0)

    assert X.shape[0] > 0
    assert unique_boards.shape[0] > 1