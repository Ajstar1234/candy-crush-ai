from board import Board
from candy import Candy
from unittest.mock import patch

import numpy as np

def test_create_board():
    board = Board(5, 5)
    board.create_board()

    assert len(board.board) == 5
    assert len(board.board[0]) == 5

def test_horizontal_match():
    board = Board(5, 5)
    board.create_board()

    # Manually set a horizontal match
    board.board[0][0] = Candy(1)
    board.board[0][1] = Candy(1)
    board.board[0][2] = Candy(1)
    board.board[0][3] = Candy(2)

    left, right = board.check_for_horizontal_matches(0, 1)

    assert left == 0
    assert right == 2

def test_vertical_match():
    board = Board(5, 5)
    board.create_board()

    # Manually set a vertical match
    board.board[0][0] = Candy(1)
    board.board[1][0] = Candy(1)
    board.board[2][0] = Candy(1)
    board.board[3][0] = Candy(2)

    top, bottom = board.check_for_vertical_matches(1, 0)

    assert top == 0
    assert bottom == 2

def test_find_matches():
    board = Board(5, 5)

    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(2), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(4), Candy(1), Candy(3), Candy(4), Candy(1)]
    ]

    matches = board.find_matches()

    assert matches == {(2, 1), (2, 2), (2, 3)}

def test_multiple_matches():
    board = Board(5, 5)

    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(1), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(1), Candy(2), Candy(1), Candy(1), Candy(1)]
    ]

    matches = board.find_matches()

    assert matches == {(2, 0), (3, 0), (4, 0), (4, 2), (4, 3), (4, 4)}

def test_horizontal_striped_creation():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(1), Candy(1), Candy(1), Candy(4)],
        [Candy(1), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(2), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(1), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    protected_positions = board.create_powerup()

    assert board.board[0][0].powerup == 'Horizontal Striped'
    assert (0, 0) in protected_positions

def test_vertical_striped_creation():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(1), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(1), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    protected_positions = board.create_powerup()

    assert board.board[0][0].powerup == 'Vertical Striped'
    assert (0, 0) in protected_positions

def test_wrapped_candy_creation():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(1), Candy(1), Candy(3), Candy(4)],
        [Candy(1), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    protected_positions = board.create_powerup()

    assert board.board[0][0].powerup == 'Wrapped'
    assert (0, 0) in protected_positions

def test_colorbomb_creation():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(1), Candy(1), Candy(1), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(2), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    protected_positions = board.create_powerup()

    assert board.board[0][0].powerup == 'Colorbomb'
    assert board.board[0][0].type == 0
    assert (0, 0) in protected_positions

def test_horizontal_striped_activation():
    board = Board(5, 5)
    board.board = [
        [Candy(1, 'Horizontal Striped'), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    matched_positions = {(0, 0)}
    protected_positions = set()

    result = board.activate_powerup(matched_positions, protected_positions)

    expected_positions = {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}

    assert result == expected_positions

def test_vertical_striped_activation():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2, 'Vertical Striped'), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    matched_positions = {(1, 0)}
    protected_positions = set()

    result = board.activate_powerup(matched_positions, protected_positions)

    expected_positions = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}

    assert result == expected_positions

def test_wrapped_activation():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3, 'Wrapped'), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    matched_positions = {(1, 1)}
    protected_positions = set()

    result = board.activate_powerup(matched_positions, protected_positions)

    expected_positions = {(0, 0), (0, 1), (0, 2),
                          (1, 0), (1, 1), (1, 2),
                          (2, 0), (2, 1), (2, 2)}

    assert result == expected_positions

def test_powerup_chain_reaction():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2, 'Horizontal Striped'), Candy(3), Candy(4, 'Vertical Striped'), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(3), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(2), Candy(2), Candy(1), Candy(3), Candy(1)]
    ]

    matched_positions = {(1, 0)}
    protected_positions = set()

    result = board.activate_powerup(matched_positions, protected_positions)

    expected_positions = {(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                          (0, 2), (2, 2), (3, 2), (4, 2)}

    assert result == expected_positions

def test_colorbomb_activation():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)],
        [Candy(3), Candy(1), Candy(0, 'Colorbomb'), Candy(2), Candy(4)],
        [Candy(4), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(1), Candy(4), Candy(2), Candy(3), Candy(2)]
    ]

    move = ((2,2), (2, 3))
    total_cleared = board.make_move(move)

    print("\nBoard after color bomb activation:")
    board.display_board()

    for row in range(board.rows):
        for col in range(board.cols):
            candy = board.board[row][col]

            if candy.powerup == "Colorbomb":
                print(f"Color bomb found at ({row}, {col})")

    assert total_cleared >= 8
    assert all(
        candy.powerup != "Colorbomb"
        for row in board.board
        for candy in row
    )
    assert board.cleared_candies.get(2, 0) >= 7

def test_invalid_swap():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(2), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(4), Candy(1), Candy(3), Candy(4), Candy(1)]
    ]

    original_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    move_validation = board.is_valid_swap(0, 0, 1, 1)

    assert move_validation is False

    current_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    assert current_board == original_board

def test_invalid_adjacent_swap():
    board = Board(5, 5)
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)], 
        [Candy(1), Candy(2), Candy(2), Candy(2), Candy(3)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(4)],
        [Candy(4), Candy(1), Candy(3), Candy(4), Candy(1)]
    ]

    original_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    move_validation = board.is_valid_swap(0, 0, 0, 1)

    assert move_validation is False

    current_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    assert current_board == original_board

def test_reshuffle_board():
    board = Board(5, 5)

    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(3)],
        [Candy(4), Candy(1), Candy(2), Candy(3), Candy(4)],
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)]
    ]

    original_candies = sorted(
        candy.type for row in board.board for candy in row
    )

    result = board.reshuffle_board()

    reshuffled_candies = sorted(
        candy.type for row in board.board for candy in row
    )

    assert result is True
    assert len(board.board) == 5
    assert all(len(row) == 5 for row in board.board)
    assert original_candies == reshuffled_candies
    assert len(board.find_matches()) == 0
    assert len(board.get_valid_moves()) > 0

def test_colorbomb_objective_tracking():
    board = Board(5, 5)

    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)],
        [Candy(3), Candy(1), Candy(0, "Colorbomb"), Candy(2), Candy(4)],
        [Candy(4), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(1), Candy(4), Candy(2), Candy(3), Candy(2)]
    ]

    # Swap the color bomb with a red candy
    move = ((2, 2), (2, 1))

    total_cleared = board.make_move(move)

    assert total_cleared >= 6
    assert board.cleared_candies.get(1, 0) >= 6

def test_cascade_objective_tracking():
    board = Board(5, 5)

    board.board = [
        [Candy(1), Candy(1), Candy(1), Candy(4), Candy(1)],
        [Candy(2), Candy(3), Candy(4), Candy(1), Candy(2)],
        [Candy(3), Candy(4), Candy(1), Candy(2), Candy(3)],
        [Candy(4), Candy(1), Candy(2), Candy(3), Candy(4)],
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)]
    ]

    with patch("board.random.randint", side_effect=[1, 1, 1, 2, 3, 4]):
        board.resolve_board()

    assert board.cleared_candies.get(1, 0) >= 6
    assert len(board.find_matches()) == 0

def test_to_numpy():
    board = Board(5, 5)
    board.create_board()

    board_array = board.to_numpy()

    assert board_array.shape == (5, 5, 2)
    assert board_array.size == 50
    assert np.all((board_array[:, :, 0] >= 1) & (board_array[:, :, 0] <= 4))
    assert np.all(board_array[:, :, 1] == 0) 

def test_to_numpy_powerups():
    board = Board(5, 5)
    board.create_board()

    board.board[0][0].powerup = "Horizontal Striped"

    board.board[2][2].type = 0
    board.board[2][2].powerup = "Colorbomb"

    board_array = board.to_numpy()

    assert board_array[0, 0, 1] == 1
    assert np.array_equal(board_array[2, 2], [0, 4])

def test_to_numpy_empty_candy():
    board = Board(5, 5)
    board.create_board()

    board.board[1][3].type = None

    board_array = board.to_numpy()

    assert board_array[1, 3, 0] == -1

    assert board_array.dtype == np.int64

    assert np.array_equal(board_array[1, 3], [-1, 0])
    