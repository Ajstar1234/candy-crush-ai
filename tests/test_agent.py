from board import Board
from candy import Candy
from agent import Agent
from game import Game

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

def test_objective_move_bonus():
    board = Board(5, 5)
    agent = Agent()
    game = Game(target_candy_type=1, target_candy_count=10)

    board.board = [
        [Candy(1), Candy(2), Candy(3), Candy(4), Candy(2)],
        [Candy(1), Candy(3), Candy(4), Candy(2), Candy(1)],
        [Candy(2), Candy(1), Candy(3), Candy(4), Candy(2)],
        [Candy(4), Candy(2), Candy(1), Candy(3), Candy(4)],
        [Candy(3), Candy(4), Candy(2), Candy(1), Candy(3)]
    ]

    move = ((2, 0), (2, 1))

    normal_score = agent.evaluate_move(board, move)
    objective_score = agent.evaluate_move(board, move, game)

    assert normal_score == 3
    assert objective_score == 9
    assert objective_score > normal_score

def test_choose_move_prioritizes_objective():
    board = Board(5, 5)
    agent = Agent()
    game = Game(target_candy_type=1, target_candy_count=10)

    board.board = [
        [Candy(1), Candy(3), Candy(3), Candy(2), Candy(1)],
        [Candy(2), Candy(1), Candy(1), Candy(4), Candy(1)],
        [Candy(2), Candy(2), Candy(4), Candy(2), Candy(3)],
        [Candy(4), Candy(2), Candy(2), Candy(3), Candy(4)],
        [Candy(3), Candy(4), Candy(4), Candy(1), Candy(2)]
    ]

    normal_move = agent.choose_move(board)
    objective_move = agent.choose_move(board, game)

    # Without an objective, prefer clearing 4 candies
    assert normal_move == ((2, 2), (3, 2))

    # With a red objective, prefer clearing 3 red candies
    assert objective_move == ((0, 0), (1, 0))

    # Verify the objective move receives its expected bonus
    assert agent.evaluate_move(board, objective_move, game) == 9