from candy import Candy

import random

# Class that makes, displays, and updates the board of the game
# Takes the parameters rows and cols (columns) which are the demesions of the board
class Board:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

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
                print(candy.type, end=" ")
            print()

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

    # (In progress) Checks a candy's index for specific powerup combinations
    def check_for_powerup_creation(self, row, col):
        left_index, right_index = self.check_for_horizontal_matches(row, col)
        top_index, bottom_index = self.check_for_vertical_matches(row, col)
        if right_index is None or left_index is None or bottom_index is None or top_index is None:
            return False
        if right_index - left_index >= 4 or bottom_index - top_index >= 4:
            self.board[row][col].create_colorbomb_powerup()
            if right_index - left_index >= 4:
                self.destroy_horizontal_match(row, left_index, col - 1)
                self.destroy_horizontal_match(row, col + 1, right_index)
            if bottom_index - top_index >= 4:
                self.destroy_vertical_match(col, top_index, row - 1)
                self.destroy_vertical_match(col, row + 1, bottom_index )
            return True
        elif right_index - left_index >= 3 and bottom_index - top_index >= 3:
            self.board[row][col].create_wrapped_powerup()
            self.destroy_horizontal_match(row, left_index, col - 1)
            self.destroy_horizontal_match(row, col + 1, right_index)
            self.destroy_vertical_match(col, top_index, row - 1)
            self.destroy_vertical_match(col, row + 1, bottom_index)
            return True
        elif right_index - left_index >= 3 or bottom_index - top_index >= 3:
            if right_index - left_index >= 3:
                self.board[row][col].create_horizontal_striped_powerup()
                self.destroy_horizontal_match(row, left_index, col - 1)
                self.destroy_horizontal_match(row, col + 1, right_index)
            if bottom_index - top_index >= 3:
                self.board[row][col].create_vertical_striped_powerup()
                self.destroy_vertical_match(col, top_index, row - 1)
                self.destroy_vertical_match(col, row + 1, bottom_index )
            return True
        elif right_index - left_index == 1 and bottom_index - top_index == 1:
            self.board[row][col].create_fish_powerup()
            self.destroy_horizontal_match(row, left_index, col - 1)
            self.destroy_horizontal_match(row, col + 1, right_index)
            self.destroy_vertical_match(col, top_index, row - 1)
            self.destroy_vertical_match(col, row + 1, bottom_index )
            return True
        else:
            return False
        
    def horizontal_striped_powerup(self, row, col):
        for candy in self.board[row]:
            if candy is not None:
                candy.type = None

    def vertical_striped_powerup(self, row, col):
        for candy in self.board:
            if candy[col] is not None:
                candy[col].type = None

    def colorbomb_powerup(self, row, col):
        for candies in self.board:
            for candy in candies:
                if candy.type == self.board[row][col].type:
                    candy.type = None

    def wrapped_powerup(self, row, col):
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if i >= 0 and i < len(self.board) and j >= 0 and j < len(self.board[i]):
                    if self.board[i][j] is not None:
                        self.board[i][j].type = None

    def fish_powerup(self, row, col):
        pass

    # Checks for if the candy at the candy_index contains any horizontal matches
    def check_for_horizontal_matches(self, row, col):
        if self.board[row][col].type is None:
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
        

    # Will check the board for horizontal, vertical, or powerup matches
    # Matches will be destroyed and the index replaced 
    def check_for_matches(self):
        col = 0 
        match_count = 0
        for row in range(0, len(self.board)):
            for col in range(0, len(self.board[row])):
                powerup_creation_check = self.check_for_powerup_creation(row, col)
                if powerup_creation_check == True:
                    continue
                left_index, right_index = self.check_for_horizontal_matches(row, col)
                top_index, bottom_index = self.check_for_vertical_matches(row, col)
                if left_index is not None and right_index is not None:
                    match_count += 1
                    self.check_for_powerup_match(row, col)
                    self.destroy_horizontal_match(row, left_index, right_index)
                if top_index is not None and bottom_index is not None:
                    match_count += 1
                    self.destroy_vertical_match(col, top_index, bottom_index)
        return match_count

    def check_for_powerup_match(self, row, col,):
        candy = self.board[row][col] 
        if candy.powerup == "Horizontal Striped":
            self.horizontal_striped_powerup(row, col)

        pass

    # Destroys the candies in a horizontal match based on it's row from the left index to the right index
    def destroy_horizontal_match(self, row, left_index, right_index):
        for candy_index in range(left_index, right_index + 1):
            self.board[row][candy_index].type = None 

    def destroy_vertical_match(self, col, top_index, bottom_index):
        for candy_index in range(top_index, bottom_index + 1):
            self.board[candy_index][col].type = None

    def resolve_board(self):
        while True:
            matches = self.check_for_matches()

            if matches == 0:
                break

            self.update_board()
            
