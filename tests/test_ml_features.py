import numpy as np

from board import Board
from ml_features import encode_state_move

def test_encode_state_move():
    board = Board(5, 5)
    board.create_board()

    move = ((1, 2), (1, 3))

    features = encode_state_move(board, move)

    assert features.shape == (180,)
    assert np.array_equal(features[175:179], [1, 2, 1, 3])
    assert np.array_equal(features[179:], [0])

    encoded_colors = features[:150].reshape(25, 6)

    assert np.all(encoded_colors.sum(axis=1) == 1)