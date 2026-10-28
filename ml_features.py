import numpy as np

def encode_state_move(board, move, game=None):
    target_type = 0

    if game is not None and game.target_candy_type is not None:
        target_type = game.target_candy_type

    target_array = np.array([target_type])

    board_array = board.to_numpy()

    flattened_board = board_array.flatten()

    (row1, col1), (row2, col2) = move

    move_array = np.array([row1, col1, row2, col2])

    feature_vector = np.concatenate([flattened_board, move_array, target_array])

    return feature_vector