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


board = Board(5, 5)
board.create_board()

board.board[0][0].powerup = "Horizontal Striped"
board.board[1][2].powerup = "Vertical Striped"
board.board[3][4].powerup = "Wrapped"

board.board[2][2].type = 0
board.board[2][2].powerup = "Colorbomb"

powerup_mapping = {
    None: 0,
    "Horizontal Striped": 1,
    "Vertical Striped": 2,
    "Wrapped": 3,
    "Colorbomb": 4
}


powerup_board = [
    [powerup_mapping[candy.powerup] for candy in row]
    for row in board.board
]

powerup_array = np.array(powerup_board)

print(powerup_array)
print(powerup_array.shape)


print(powerup_mapping[None])
print(powerup_mapping["Horizontal Striped"])
print(powerup_mapping["Colorbomb"])

combined_array = np.stack(
    [board_array, powerup_array],
    axis=-1
)

print(combined_array.shape)

board = Board(5, 5)
board.create_board()

board.board[2][2].type = 0
board.board[2][2].powerup = "Colorbomb"

board_array = board.to_numpy()

print(board_array.shape)
print(board_array[2, 2])

move = ((1, 2), (1, 3))

cords = []
for cord in move:
    for num in cord:
        cords.append(num)
cord_array = np.array(cords)
print(cord_array)
print(cord_array.shape)

flat_board = board_array.flatten()

state_move_features = np.concatenate([flat_board, cord_array])

print(state_move_features.shape)

from ml_features import encode_state_move

board = Board(5, 5)
board.create_board()

move = ((1, 2), (1, 3))

features = encode_state_move(board, move)

print(features.shape)
print(features[-4:])