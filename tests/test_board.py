from board import Board
from candy import Candy

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

    assert total_cleared >= 8
    assert all(
        candy.powerup != "Colorbomb"
        for row in board.board
        for candy in row
    )

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