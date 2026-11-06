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

def test_dynamic_feature_encoding():
    board = Board(9, 9, num_candy_types=6)
    board.create_board()

    move = ((0, 0), (0, 1))

    features = encode_state_move(board, move)

    assert features.shape == (734,)

    color_features = features[:648].reshape(81, 8)

    assert np.all(color_features.sum(axis=1) == 1)
    assert np.array_equal(features[-5:-1], [0, 0, 0, 1])
