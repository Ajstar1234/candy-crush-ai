import numpy as np

from board import Board
from ml_features import encode_state_move

def test_encode_state_move():
    board = Board(5, 5)
    board.create_board()

    move = ((1, 2), (1, 3))

    features = encode_state_move(board, move)

    combined_array = board.to_numpy()

    flattened_board = combined_array.flatten()

    assert features.shape == (54,)
    assert np.array_equal(features[-4:], [1, 2, 1, 3])
    assert np.array_equal(features[:50], flattened_board)