import numpy as np
import random

from board import Board
from game import Game
from agent import Agent
from ml_features import encode_state_move


def generate_training_data(num_games=100, seed=None):
    if seed is not None:
        random.seed(seed)

    X = []
    y = []

    for game_index in range(num_games):
        target_type = random.randint(1, 4)
        board = Board(5, 5)
        board.create_board()
        board.resolve_board()

        game = Game(max_turns=8, target_candy_type=target_type, target_candy_count=10)

        agent = Agent()

        while not game.is_game_over():

            valid_moves = board.get_valid_moves()

            if not valid_moves:
                break

            for move in valid_moves:
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
    X, y = generate_training_data(num_games=10, seed=42)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target types:", np.unique(X[:, -1]))
    print("Minimum heuristic score:", np.min(y))
    print("Maximum heuristic score:", np.max(y))