from board import Board
from game import Game
from ml_agent import MLAgent

def test_ml_agent_choose_move():
    board = Board(5, 5)
    board.create_board()
    board.resolve_board()

    agent = MLAgent()
    game = Game(target_candy_type=1, target_candy_count=10)

    valid_moves = board.get_valid_moves()

    if len(valid_moves) == 0:
        board.reshuffle_board()
        valid_moves = board.get_valid_moves()

    move = agent.choose_move(board, game)

    assert move in valid_moves

def test_ml_agent_9x9():
    board = Board(9, 9, num_candy_types=6)
    board.create_board()
    board.resolve_board()

    agent = MLAgent(input_size=734, checkpoint_path="models/best_move_model_9x9.pth")
    game = Game(max_turns=8, target_candy_type=1, target_candy_count=10)

    valid_moves = board.get_valid_moves()

    if len(valid_moves) == 0:
        board.reshuffle_board()
        valid_moves = board.get_valid_moves()

    assert len(valid_moves) > 0

    move = agent.choose_move(board, game)

    assert move in valid_moves