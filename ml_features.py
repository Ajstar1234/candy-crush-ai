import numpy as np

def encode_state_move(board, move):
    board_array = board.to_numpy()

    flattened_board = board_array.flatten()

    (row1, col1), (row2, col2) = move

    move_array = np.array([row1, col1, row2, col2])

    feature_vector = np.concatenate([flattened_board, move_array])

    return feature_vector