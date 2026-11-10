import numpy as np
import random

from board import Board
from game import Game
from agent import Agent
from ml_features import encode_state_move, encode_state_move_with_target_mask


def generate_training_data(num_games=100, seed=None, rows=5, cols=5, num_candy_types=4, use_target_mask=False):

    X = []
    y = []

    for game_index in range(num_games):
        if seed is not None:
            game_seed = seed + game_index
            random.seed(game_seed)

        target_type = random.randint(1, num_candy_types)
        
        board = Board(rows, cols, num_candy_types=num_candy_types)
        board.create_board()
        board.resolve_board()

        game = Game(max_turns=8, target_candy_type=target_type, target_candy_count=10)

        agent = Agent()

        while not game.is_game_over():

            valid_moves = board.get_valid_moves()

            if not valid_moves:
                break

            for move in valid_moves:
                if use_target_mask:
                    features = encode_state_move_with_target_mask(board, move, game)
                else:
                    features = encode_state_move(board, move, game)
                score = agent.evaluate_move(board, move, game)

                X.append(features)
                y.append(score)

            chosen_move = agent.choose_move(board, game)

            total_cleared = board.make_move(chosen_move)

            if total_cleared is False:
                break

            points_earned = total_cleared * 10
            game.update_score(points_earned)

            target_candies_cleared = board.cleared_candies.get(game.target_candy_type, 0)
            game.update_collected_candies(target_candies_cleared)
            game.update_turn()

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = generate_training_data(
        num_games=10,
        seed=42,
        rows=9,
        cols=9,
        num_candy_types=6,
        use_target_mask=True
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target types:", np.unique(X[:, -82]))
    print("Minimum heuristic score:", np.min(y))
    print("Maximum heuristic score:", np.max(y))