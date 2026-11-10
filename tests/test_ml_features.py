import numpy as np

from board import Board
from candy import Candy
from game import Game
from ml_features import encode_state_move, encode_state_move_with_target_mask

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

def test_target_mask_encoding():
    board = Board(5, 5, num_candy_types=4)

    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(1), Candy(4), Candy(3), Candy(2)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(3)],
        [Candy(4), Candy(3), Candy(2), Candy(1), Candy(4)],
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(2)]
    ]

    game = Game(target_candy_type=1)
    move = ((0, 0), (0, 1))

    features = encode_state_move_with_target_mask(board, move, game)

    assert features.shape == (205,)

    expected_mask = np.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0]
    ])

    assert np.array_equal(features[-25:], expected_mask.flatten())

    features_no_game = encode_state_move_with_target_mask(board, move)

    assert features_no_game.shape == (205,)
    assert np.all(features_no_game[-25:] == 0)
