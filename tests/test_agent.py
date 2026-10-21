from board import Board
from candy import Candy
from agent import Agent

def test_agent_evaluate_move():
    board = Board(5, 5)
    agent = Agent()
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(1), Candy(3), Candy(4), Candy(1), Candy(2)],
        [Candy(3), Candy(1), Candy(2), Candy(2), Candy(4)],
        [Candy(4), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(1), Candy(4), Candy(2), Candy(3), Candy(2)]
    ]

    original_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    move = ((2, 0), (2, 1))
    score = agent.evaluate_move(board, move)

    current_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    assert score == 3
    assert current_board == original_board

def test_agent_choose_move():
    board = Board(5, 5)
    agent = Agent()
    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(1), Candy(3), Candy(4), Candy(1), Candy(2)],
        [Candy(3), Candy(1), Candy(2), Candy(2), Candy(4)],
        [Candy(4), Candy(2), Candy(3), Candy(4), Candy(1)],
        [Candy(1), Candy(4), Candy(2), Candy(3), Candy(2)]
    ]

    original_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    valid_moves = board.get_valid_moves()
    assert len(valid_moves) > 0

    move = agent.choose_move(board)
    assert move in valid_moves

    score = agent.evaluate_move(board, move)

    for moves in valid_moves:
        assert agent.evaluate_move(board, moves) <= score

    current_board = [
        [candy.type for candy in row]
        for row in board.board
    ]

    assert current_board == original_board