import numpy as np

def encode_state_move(board, move, game=None):
    target_type = 0

    if game is not None and game.target_candy_type is not None:
        target_type = game.target_candy_type

    target_array = np.array([target_type])

    num_categories = board.num_candy_types + 2
    board_array = board.to_numpy()

    colors = board_array[:, :, 0]
    powerups = board_array[:, :, 1]

    colors = colors + 1

    one_hot_colors = np.eye(num_categories)[colors]

    flattened_one_hot_colors = one_hot_colors.flatten()
    flattened_powerups = powerups.flatten()

    (row1, col1), (row2, col2) = move

    move_array = np.array([row1, col1, row2, col2])

    feature_vector = np.concatenate([
        flattened_one_hot_colors,
        flattened_powerups,
        move_array,
        target_array
    ])

    return feature_vector

def encode_state_move_with_target_mask(board, move, game=None):
    feature_vector = encode_state_move(board, move, game)

    target_mask = np.zeros((board.rows, board.cols), dtype=int)

    if game is not None and game.target_candy_type is not None:
        target_mask = (
            board.to_numpy()[:, :, 0] == game.target_candy_type
        ).astype(int)

    target_mask = target_mask.flatten()

    return np.concatenate([feature_vector, target_mask])

def encode_rl_state_move(board, move, game):
    feature_vector = encode_state_move_with_target_mask(board, move, game)

    remaining_turns = max(0, game.max_turns - game.turn)

    remaining_targets = max(
        0,
        game.target_candy_count - game.collect_candies
    )
    feature_vector = np.concatenate([
        feature_vector,
        np.array([remaining_turns, remaining_targets])
    ])

    return feature_vector

def encode_normalized_rl_state_move(board, move, game):
    features = encode_rl_state_move(
        board, move, game
    ).astype(np.float32)

    move_start = board.rows * board.cols * (board.num_candy_types + 3)

    # Normalize move coordinates
    features[move_start] /= max(1, board.rows - 1)
    features[move_start + 1] /= max(1, board.cols - 1)
    features[move_start + 2] /= max(1, board.rows - 1)
    features[move_start + 3] /= max(1, board.cols - 1)

    # Normalize target candy type
    features[move_start + 4] /= max(1, board.num_candy_types)

    # Normalize remaining turns and targets
    features[-2] /= max(1, game.max_turns)
    features[-1] /= max(1, game.target_candy_count)

    return features