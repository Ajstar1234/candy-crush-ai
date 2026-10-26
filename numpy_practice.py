import numpy as np

from board import Board

board = Board(5, 5)
board.create_board()
board.display_board()

numeric_board = [
    [candy.type for candy in row]
    for row in board.board
]

board_array = np.array(numeric_board)

flat_board = board_array.flatten()

print(board_array)
print(board_array.shape)
print(flat_board)

board = [
    [1, 2, 3, 4, 1],
    [4, 1, 2, 3, 4],
    [1, 2, 3, 4, 1],
    [2, 3, 4, 1, 2],
    [3, 4, 1, 2, 3]
]

board_array = np.array(board)

red_count = np.sum(board_array == 1)
green_count = np.sum(board_array == 2)
blue_count = np.sum(board_array == 3)

all_count = np.sum(board_array.size)

flat_board = board_array.flatten()

print(flat_board)
print(flat_board.shape)

print(all_count)
print(red_count)
print(board_array)
print(board_array.shape)
print(board_array[2, 3])
print(board_array[0])

