from candy import Candy

import numpy as np
import random

# Class that makes, displays, and updates the board of the game
# Takes the parameters rows and cols (columns) which are the demesions of the board
class Board:

    #==================================
    # Board initialization
    #==================================

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cleared_candies = {}

    # Creates the initial board filled with random types of candy
    def create_board(self):
        self.board = []
        # Create each row
        for i in range(0, self.rows):
            row = []
            # Create and fill each row in a with candy
            for j in range(0, self.cols):
                candy = Candy(random.randint(1, 4))
                row.append(candy)

            self.board.append(row)
    
    # Displays the boad of the game row by row 
    def display_board(self):
        for row in self.board:
            for candy in row:

                if candy.type is None:
                    symbol = "."

                elif candy.powerup == "Horizontal Striped":
                    symbol = f"{candy.type}H"

                elif candy.powerup == "Vertical Striped":
                    symbol = f"{candy.type}V"

                elif candy.powerup == "Wrapped":
                    symbol = f"{candy.type}W"

                elif candy.powerup == "Colorbomb":
                    symbol = "CB"

                else:
                    symbol = str(candy.type)

                print(f"{symbol:>3}", end=" ")

            print()

    def to_numpy(self):
        powerup_mapping = {
            None: 0,
            "Horizontal Striped": 1,
            "Vertical Striped": 2,
            "Wrapped": 3,
            "Colorbomb": 4
        }

        numeric_board = [
            [candy.type if candy.type is not None else -1 for candy in row]
            for row in self.board
        ]

        board_array = np.array(numeric_board, dtype=np.int64)

        powerup_board = [
            [powerup_mapping[candy.powerup] for candy in row]
            for row in self.board
        ]

        powerup_array = np.array(powerup_board, dtype=np.int64)

        combined_array = np.stack(
            [board_array, powerup_array],
            axis=-1
        )

        return combined_array

    #===============================
    # Move handling
    #===============================

    def swap_candies(self, row1, col1, row2, col2):
        row_difference = abs(row1 - row2)
        col_difference = abs(col1 - col2)

        # Only allow directly adjacent candies
        if row_difference + col_difference != 1:
            return False

        self.board[row1][col1], self.board[row2][col2] = (
            self.board[row2][col2],
            self.board[row1][col1]
        )

        return True

    def is_valid_swap(self, row1, col1, row2, col2):
        # First make sure the candies are adjacent
        if not self.swap_candies(row1, col1, row2, col2):
            return False

        # Color bombs can swap without making a normal match
        first_candy = self.board[row1][col1]
        second_candy = self.board[row2][col2]

        if first_candy.powerup == "Colorbomb" and second_candy.type in range(1, 5):
            return True

        if second_candy.powerup == "Colorbomb" and first_candy.type in range(1, 5):
            return True

        # Check whether either swapped candy is now part of a match
        first_horizontal = self.check_for_horizontal_matches(row1, col1)
        first_vertical = self.check_for_vertical_matches(row1, col1)

        second_horizontal = self.check_for_horizontal_matches(row2, col2)
        second_vertical = self.check_for_vertical_matches(row2, col2)

        valid = (
            first_horizontal != (None, None)
            or first_vertical != (None, None)
            or second_horizontal != (None, None)
            or second_vertical != (None, None)
        )

        # If invalid, undo the swap
        if not valid:
            self.swap_candies(row1, col1, row2, col2)

        return valid

    def get_valid_moves(self):
        valid_moves = []

        for row in range(self.rows):
            for col in range(self.cols):

                # Check swap with candy to the right
                if col < self.cols - 1:
                    if self.is_valid_swap(row, col, row, col + 1):
                        valid_moves.append(((row, col), (row, col + 1)))

                        # Undo valid swap so board stays unchanged
                        self.swap_candies(row, col, row, col + 1)

                # Check swap with candy below
                if row < self.rows - 1:
                    if self.is_valid_swap(row, col, row + 1, col):
                        valid_moves.append(((row, col), (row + 1, col)))

                        # Undo valid swap so board stays unchanged
                        self.swap_candies(row, col, row + 1, col)

        return valid_moves

    def make_move(self, move):
        if move is None:
            return False

        (row1, col1), (row2, col2) = move

        # Make sure the move is actually valid
        if not self.is_valid_swap(row1, col1, row2, col2):
            return False

        # Reset candy counts for this move
        self.cleared_candies = {}

        # Check for color bomb activation after the swap
        first_candy = self.board[row1][col1]
        second_candy = self.board[row2][col2]

        bomb_position = None
        target_color = None

        if first_candy.powerup == "Colorbomb" and second_candy.type in range(1, 5):
            bomb_position = (row1, col1)
            target_color = second_candy.type

        elif second_candy.powerup == "Colorbomb" and first_candy.type in range(1, 5):
            bomb_position = (row2, col2)
            target_color = first_candy.type

        if bomb_position is not None:
            cleared_positions = set()

            for row in range(self.rows):
                for col in range(self.cols):
                    candy = self.board[row][col]

                    if candy.type == target_color or (row, col) == bomb_position:
                        cleared_positions.add((row, col))

            # Clear all candies of the selected color and the bomb itself
            for row, col in cleared_positions:
                candy = self.board[row][col]
                candy_type = candy.type

                if candy_type is not None and candy_type != 0:
                    self.cleared_candies[candy_type] = self.cleared_candies.get(candy_type, 0) + 1



                self.board[row][col].type = None
                self.board[row][col].powerup = None

            total_cleared = len(cleared_positions)

            # Apply gravity and resolve any additional cascades
            self.update_board()
            total_cleared += self.resolve_board()

        else:
            total_cleared = self.resolve_board(
                swapped_positions=[(row1, col1), (row2, col2)]
            )

        return total_cleared

    #===========================
    # Match detection
    #===========================

    # Checks for if the candy at the candy_index contains any horizontal matches
    def check_for_horizontal_matches(self, row, col):
        if self.board[row][col].type is None or self.board[row][col].type == 0:
            return None, None
        
        match_count = 1

        # Checks the left side of the candy_index
        if col > 0:
            left_index = col
            for index in range(col -1, -1, -1):
                if self.board[row][col].type == self.board[row][index].type:
                    match_count += 1
                    left_index = index
                else:
                    break
        else:
            left_index = col

        # Checks the right side of the candy_index
        if col < len(self.board[row]) - 1:
            right_index = col
            for index in range(col + 1, len(self.board[row])):
                if self.board[row][col].type == self.board[row][index].type:
                    match_count += 1
                    right_index = index
                else:
                    break
        else:
            right_index = col

        # Matches need 3 or more of the same type candy in a row
        if match_count >= 3:
            # Returns the left and right index of the matched candies
            return left_index, right_index
        else:
            return None, None

    # Checks for if the candy at the candy_index contains any vertical matches
    def check_for_vertical_matches(self, row, col):
        if self.board[row][col].type is None or self.board[row][col].type == 0:
            return None, None
        match_count = 1

        # Checks the top side of the candy_index
        if row > 0:
            top_index = row
            for index in range(row - 1, -1, -1):
                if self.board[index][col].type == self.board[row][col].type:
                    match_count += 1
                    top_index = index
                else:
                    break
        else:
            top_index = row

        if row < len(self.board) - 1:
            bottom_index = row
            for index in range(row + 1, len(self.board)):
                if self.board[index][col].type == self.board[row][col].type:
                    match_count += 1
                    bottom_index = index
                else:
                    break 
        else:
            bottom_index = row

        if match_count >= 3:
            return top_index, bottom_index
        else:
            return None, None

    def find_matches(self):
        matched_positions = set()

        for row in range(self.rows):
            for col in range(self.cols):
                left, right = self.check_for_horizontal_matches(row, col)

                if left is not None:
                    for c in range(left, right + 1):
                        matched_positions.add((row, c))

                top, bottom = self.check_for_vertical_matches(row, col)

                if top is not None:
                    for r in range(top, bottom + 1):
                        matched_positions.add((r, col))

        return matched_positions
        

    # Will check the board for horizontal, vertical, or powerup matches
    # Matches will be destroyed and the index replaced 
    def check_for_matches(self, swapped_positions=None):
        matched_positions = self.find_matches()
        protected_positions = self.create_powerup(swapped_positions)
        matched_positions = self.activate_powerup(matched_positions, protected_positions)

        cleared_candies = {}

        for row, col in matched_positions:

            if (row, col) not in protected_positions:
                candy_type = self.board[row][col].type

                if candy_type is not None and candy_type != 0:
                    cleared_candies[candy_type] = cleared_candies.get(candy_type, 0) + 1

                self.board[row][col].type = None

        for candy_type, count in cleared_candies.items():
            self.cleared_candies[candy_type] = self.cleared_candies.get(candy_type, 0) + count

        return len(matched_positions - protected_positions)

    def create_powerup(self, swapped_positions=None):
        protected_positions = set()
        powerups_to_create = set()
        processed_horizontal = set()
        processed_vertical = set()
        overlapping_positions = set()

        for row in range(len(self.board)):

            for col in range(len(self.board[row])):
                left_index, right_index = self.check_for_horizontal_matches(row, col)
                top_index, bottom_index = self.check_for_vertical_matches(row, col)

                horizontal_length = right_index - left_index + 1 if left_index is not None else 0
                vertical_length = bottom_index - top_index + 1 if top_index is not None else 0

                if horizontal_length >= 3 and vertical_length >= 3:
                    overlapping_positions.add((row, col))

                if horizontal_length >= 5:
                    match_key = (row, left_index, right_index)

                    if match_key not in processed_horizontal:
                        processed_horizontal.add(match_key)

                        powerup_position = (row, left_index)

                        if swapped_positions is not None:
                            for swap_row, swap_col in swapped_positions:
                                if swap_row == row and left_index <= swap_col <= right_index:
                                    powerup_position = (swap_row, swap_col)
                                    break

                        protected_positions.add(powerup_position)
                        powerups_to_create.add(
                            (*powerup_position, "Colorbomb")
                        )

                elif horizontal_length == 4:
                    match_key = (row, left_index, right_index)

                    if match_key not in processed_horizontal:
                        processed_horizontal.add(match_key)

                        powerup_position = (row, left_index)

                        if swapped_positions is not None:
                            for swap_row, swap_col in swapped_positions:
                                if swap_row == row and left_index <= swap_col <= right_index:
                                    powerup_position = (swap_row, swap_col)
                                    break

                        protected_positions.add(powerup_position)
                        powerups_to_create.add(
                            (*powerup_position, "Horizontal Striped")
                        )
                
                if vertical_length >= 5:
                    match_key = (col, top_index, bottom_index)

                    if match_key not in processed_vertical:
                        processed_vertical.add(match_key)

                        powerup_position = (top_index, col)

                        if swapped_positions is not None:
                            for swap_row, swap_col in swapped_positions:
                                if swap_col == col and top_index <= swap_row <= bottom_index:
                                    powerup_position = (swap_row, swap_col)
                                    break

                        protected_positions.add(powerup_position)
                        powerups_to_create.add(
                            (*powerup_position, "Colorbomb")
                        )

                elif vertical_length == 4:
                    match_key = (col, top_index, bottom_index)

                    if match_key not in processed_vertical:
                        processed_vertical.add(match_key)

                        powerup_position = (top_index, col)

                        if swapped_positions is not None:
                            for swap_row, swap_col in swapped_positions:
                                if swap_col == col and top_index <= swap_row <= bottom_index:
                                    powerup_position = (swap_row, swap_col)
                                    break

                        protected_positions.add(powerup_position)
                        powerups_to_create.add(
                            (*powerup_position, "Vertical Striped")
                        )

        # Handle overlapping matches
        for row, col in overlapping_positions:
            powerups_to_create = {
                (r, c, powerup_type)
                for r, c, powerup_type in powerups_to_create
                if (r, c) != (row, col)
            }

            protected_positions.add((row, col))
            powerups_to_create.add((row, col, "Wrapped"))

        # Apply powerups to candies
        for row, col, powerup_type in powerups_to_create:
            self.board[row][col].powerup = powerup_type

            if powerup_type == "Colorbomb":
                self.board[row][col].type = 0

        return protected_positions


    def activate_powerup(self, matched_positions, protected_positions):
        activated_powerups = set()
        positions_to_check = list(matched_positions)

        while positions_to_check:
            row, col = positions_to_check.pop()

            if (row, col) in protected_positions:
                continue

            candy = self.board[row][col]

            if (row, col) in activated_powerups:
                continue

            if candy.powerup == "Horizontal Striped":
                activated_powerups.add((row, col))

                for c in range(self.cols):
                    if (row, c) not in matched_positions:
                        matched_positions.add((row, c))
                        positions_to_check.append((row, c))

            elif candy.powerup == "Vertical Striped":
                activated_powerups.add((row, col))

                for r in range(self.rows):
                    if (r, col) not in matched_positions:
                        matched_positions.add((r, col))
                        positions_to_check.append((r, col))

            elif candy.powerup == "Wrapped":
                activated_powerups.add((row, col))

                # Clear the 3x3 area surrounding the wrapped candy
                for r in range(row - 1, row + 2):
                    for c in range(col - 1, col + 2):

                        # Make sure the position is inside the board
                        if 0 <= r < self.rows and 0 <= c < self.cols:

                            if (r, c) not in matched_positions:
                                matched_positions.add((r, c))
                                positions_to_check.append((r, c))

        return matched_positions



    #============================
    # Board updates
    #============================

    # (In progress) Update the board when candy is matched and new candy replacec it by falling from a higher up column 
    def update_board(self):
        for col in range(self.cols):

            remaining_candies = []

            # Collect all non-destroyed candies in this column
            for row in range(self.rows):
                if self.board[row][col].type is not None:
                    remaining_candies.append(self.board[row][col])

            # Number of new candies needed
            missing = self.rows - len(remaining_candies)

            # Create new candies for the top
            new_candies = []

            for i in range(missing):
                new_candies.append(Candy(random.randint(1, 4)))

            # New column = new candies on top + existing candies below
            new_column = new_candies + remaining_candies

            # Put the column back into the board
            for row in range(self.rows):
                self.board[row][col] = new_column[row]



    def resolve_board(self, swapped_positions=None):
        total_cleared = 0

        while True:
            matches = self.check_for_matches(swapped_positions)

            if matches == 0:
                break

            total_cleared += matches
            self.update_board()

            # Cascades are not directly caused by a swap
            swapped_positions = None

        return total_cleared

    def reshuffle_board(self):
        candies = []

        for row in self.board:
            for candy in row:
                candies.append(candy)

        for attempt in range(100):
            random.shuffle(candies)

            self.board = []

            for row in range(self.rows):
                start = row * self.cols
                end = start + self.cols

                new_row = candies[start:end]
                self.board.append(new_row)

            if len(self.find_matches()) == 0 and len(self.get_valid_moves()) > 0:
                return True
            
        # If reshuffling fails, gerneate a fresh board
        self.create_board()
        self.resolve_board()

        return len(self.get_valid_moves()) > 0