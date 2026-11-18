import numpy as np
from generate_dataset import generate_training_data


def test_generate_training_data():
    X, y = generate_training_data(num_games=10)

    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0
    assert X.shape[1] == 180

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

def test_generate_group_sizes():
    X, y, group_sizes = generate_training_data(
        num_games=2,
        seed=42,
        rows=5,
        cols=5,
        num_candy_types=4,
        use_target_mask=True,
        return_group_sizes=True
    )

    assert len(group_sizes) > 0
    assert sum(group_sizes) == len(X)
    assert len(X) == len(y)
    assert X.shape[1] == 205
    assert all(size > 0 for size in group_sizes)