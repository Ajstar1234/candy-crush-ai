import random


class Agent:

    def choose_move(self, board):
        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:
            return None

        return random.choice(valid_moves)