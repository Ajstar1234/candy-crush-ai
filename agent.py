import random


class Agent:

    def choose_move(self, board):
        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:
            return None

        best_move = None
        best_score = -1

        for move in valid_moves:
            score = self.evaluate_move(board, move)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def evaluate_move(self, board, move):
        (row1, col1), (row2, col2) = move

        # Temporarily swap the candies
        board.swap_candies(row1, col1, row2, col2)

        # Evaluate color bomb swaps
        first_candy = board.board[row1][col1]
        second_candy = board.board[row2][col2]

        target_color = None

        if first_candy.powerup == "Colorbomb" and second_candy.type in range(1, 5):
            target_color = second_candy.type

        elif second_candy.powerup == "Colorbomb" and first_candy.type in range(1, 5):
            target_color = first_candy.type

        if target_color is not None:
            score = 1  # Include the color bomb itself

            for row in board.board:
                for candy in row:
                    if candy.type == target_color:
                        score += 1

            # Undo the temporary swap before returning
            board.swap_candies(row1, col1, row2, col2)

            return score

        matched_positions = set()

        # Check both candies for matches
        for row, col in [(row1, col1), (row2, col2)]:

            left, right = board.check_for_horizontal_matches(row, col)

            if left is not None:
                for c in range(left, right + 1):
                    matched_positions.add((row, c))

            top, bottom = board.check_for_vertical_matches(row, col)

            if top is not None:
                for r in range(top, bottom + 1):
                    matched_positions.add((r, col))

        # Undo the temporary swap
        board.swap_candies(row1, col1, row2, col2)

        return len(matched_positions)