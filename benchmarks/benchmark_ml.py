from board import Board
from game import Game
from ml_agent import MLAgent
import random

def run_ml_game(agent, seed=None, max_turns=8, target_candy_count=10):
    if seed is not None:
        random.seed(seed)

    board = Board(5,5)
    board.create_board()
    board.resolve_board()
    game = Game(max_turns=max_turns, target_candy_type=1, target_candy_count=target_candy_count)

    while not game.is_game_over():
        move = agent.choose_move(board, game)

        if move is None:
            reshuffle = board.reshuffle_board()

            if not reshuffle:
                break

            continue

        total_cleared = board.make_move(move)

        if total_cleared is False:
            break

        points_earned = total_cleared * 10
        game.update_score(points_earned)

        target_candies_cleared = board.cleared_candies.get(game.target_candy_type, 0)
        game.update_collected_candies(target_candies_cleared)

        game.update_turn()

    results = {
        "won": game.is_won(),
        "score": game.score,
        "turns": game.turn,
        "candies_collected": game.collect_candies,
    }

    return results

def benchmark_ml(num_games=100, max_turns=8, target_candy_count=10):
    agent = MLAgent()
    results = []

    for i in range(num_games):
        game_result = run_ml_game(agent, seed=i, max_turns=max_turns, target_candy_count=target_candy_count)
        results.append(game_result)

    return results

if __name__ == "__main__":
    difficulties = {
        "Easy": {"max_turns": 8, "target_candy_count": 10},
        "Medium": {"max_turns": 6, "target_candy_count": 12},
        "Hard": {"max_turns": 5, "target_candy_count": 15}
    }

    for difficulty, settings in difficulties.items():
        print(f"Running benchmark for {difficulty} difficulty...")
        results = benchmark_ml(num_games=100, max_turns=settings["max_turns"], target_candy_count=settings["target_candy_count"])

        total_wins = 0
        total_score = 0
        total_turns = 0
        total_target_candies_collected = 0
        for result in results:
            if result["won"]:
                total_wins += 1

            total_score += result["score"]
            total_turns += result["turns"]
            total_target_candies_collected += result["candies_collected"]

        win_rate = total_wins / len(results) * 100
        average_score = total_score / len(results)
        average_turns = total_turns / len(results)
        average_target_candies_collected = total_target_candies_collected / len(results)

        print(f"Win rate: {win_rate:.2f}%")
        print(f"Average score: {average_score:.2f}")
        print(f"Average turns: {average_turns:.2f}")
        print(f"Average target candies collected: {average_target_candies_collected:.2f}")
