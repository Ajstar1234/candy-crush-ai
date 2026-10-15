from board import Board
from agent import Agent
from game import Game
from candy import Candy 

def main():
    board = Board(5, 5)
    agent = Agent()
    game = Game(max_turns=5, target_score=1000)

    board.create_board()

    print("Initial board:")
    board.display_board()

    board.resolve_board()

    print("\nStarting board:")
    board.display_board()

    for turn in range(game.max_turns):
        print(f"\nTurn {turn + 1}")

        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:
            print("No valid moves available")
            break

        move = agent.choose_move(board)

        print("Agent chose:")
        print(move)

        total_cleared = board.make_move(move)

        if total_cleared is not False:
            points_earned = total_cleared * 10
            game.update_score(points_earned)
            game.update_turn()

            print(f"Candies cleared: {total_cleared}")
            print(f"Points earned: {points_earned}")
            print(f"Total score: {game.score}")         

        else:
            print("Invalid move")
            break

        print("Board after move:")
        board.display_board()

        if game.is_won():
            print("You Win!")
            print(f"Final score: {game.score}")
            print(f"Number of turns used: {game.turn}")
            break

    if game.is_lost():
        print("You Lose")
        print(f"Final score: {game.score}")
        print(f"Number of turns used: {game.turn}")

main()