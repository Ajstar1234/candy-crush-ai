import random

from board import Board
from game import Game


class RLEnvironment:
    def __init__(
        self,
        rows=9,
        cols=9,
        num_candy_types=6,
        max_turns=8,
        target_candy_type=1,
        target_candy_count=10
    ):
        self.rows = rows
        self.cols = cols
        self.num_candy_types = num_candy_types
        self.max_turns = max_turns
        self.target_candy_type = target_candy_type
        self.target_candy_count = target_candy_count

        self.board = None
        self.game = None
        self.done = False

    def reset(self, seed=None):
        self.done = False
        if seed is not None:
            random.seed(seed)

        self.board = Board(
            self.rows,
            self.cols,
            num_candy_types=self.num_candy_types
        )

        self.board.create_board()
        self.board.resolve_board()

        self.game = Game(
            max_turns=self.max_turns,
            target_candy_type=self.target_candy_type,
            target_candy_count=self.target_candy_count
        )

        if not self.board.get_valid_moves():
            if not self.board.reshuffle_board():
                self.done = True

        return self.board.to_numpy()

    def get_valid_moves(self):
        if self.board is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        return self.board.get_valid_moves()

    def step(self, move):
        if self.board is None or self.game is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        if self.done or self.game.is_game_over():
            raise ValueError("Cannot make a move after the game has ended.")

        if move not in self.get_valid_moves():
            raise ValueError("Invalid move.")

        total_cleared = self.board.make_move(move)

        if total_cleared is False:
            raise RuntimeError("A valid move failed to execute.")

        points_earned = total_cleared * 10
        self.game.update_score(points_earned)
        target_candies_cleared = self.board.cleared_candies.get(self.game.target_candy_type, 0)
        self.game.update_collected_candies(target_candies_cleared)
        self.game.update_turn()

        done = self.game.is_game_over()

        if not done and not self.board.get_valid_moves():
            if not self.board.reshuffle_board():
                done = True

        self.done = done

        observation = self.board.to_numpy()

        reward = float(target_candies_cleared)

        if self.game.is_won():
            reward += 10.0

        info = {
            "candies_cleared": total_cleared,
            "score": self.game.score,
            "won": self.game.is_won()
        }

        return observation, reward, done, info