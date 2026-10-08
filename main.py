from board import Board
from agent import Agent

def main():
    board = Board(5, 5)
    agent = Agent()

    board.create_board()

    print("Initial board:")
    board.display_board()

    board.resolve_board()

    print("\nStarting board:")
    board.display_board()

    for turn in range(10):
        print(f"\nTurn {turn + 1}")

        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:
            print("No valid moves available")
            break

        move = agent.choose_move(board)

        print("Agent chose:")
        print(move)

        board.make_move(move)

        print("Board after move:")
        board.display_board()


main()