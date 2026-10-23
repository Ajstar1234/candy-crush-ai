from game import Game

def test_candy_objective_win():
    game = Game(target_candy_type=1, target_candy_count=10)

    game.update_collected_candies(6)

    assert game.is_won() is False

    game.update_collected_candies(4)

    assert game.is_won() is True

    assert game.is_game_over() is True

def test_candy_objective_loss():
    game = Game(max_turns=5, target_candy_type=1, target_candy_count=10)

    game.update_collected_candies(6)

    for _ in range(5):
        game.update_turn()

    assert game.is_lost() is True

    assert game.is_game_over() is True

def test_candy_objective_win_on_last_turn():
    game = Game(max_turns=5, target_candy_type=1, target_candy_count=10)

    for _ in range(4):
        game.update_turn()

    game.update_collected_candies(10)
    game.update_turn()

    assert game.turn == 5
    assert game.is_won() is True
    assert game.is_lost() is False
    assert game.is_game_over() is True
