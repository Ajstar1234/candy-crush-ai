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