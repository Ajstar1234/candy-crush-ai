from board import Board

def main():
    board = Board(5, 5)
    board.create_board()
    board.display_board()
    print("\n")
    match_count = board.check_for_matches()
    while match_count > 0:
        board.check_for_matches()
        match_count = board.check_for_matches()
        board.update_board()
        print("\n")
        board.display_board()

    pass



main()