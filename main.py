from board import Board


def main():
    board = Board(5, 5)

    board.create_board()

    print("Initial board:")
    board.display_board()

    while True:
        match_count = board.check_for_matches()

        if match_count == 0:
            break

        board.update_board()

        print("\nUpdated board:")
        board.display_board()


main()