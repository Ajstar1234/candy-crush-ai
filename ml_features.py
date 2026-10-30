import numpy as np

def encode_state_move(board, move, game=None):
    target_type = 0

    if game is not None and game.target_candy_type is not None:
        target_type = game.target_candy_type

    target_array = np.array([target_type])

    board_array = board.to_numpy()

    colors = board_array[:, :, 0]
    powerups = board_array[:, :, 1]

    colors = colors + 1

    one_hot_colors = np.eye(6)[colors]

    flattened_one_hot_colors = one_hot_colors.flatten()
    flattened_powerups = powerups.flatten()

    print(one_hot_colors.shape)

    (row1, col1), (row2, col2) = move

    move_array = np.array([row1, col1, row2, col2])

    feature_vector = np.concatenate([
        flattened_one_hot_colors,
        flattened_powerups,
        move_array,
        target_array
    ])

    print(feature_vector.shape)

    return feature_vector