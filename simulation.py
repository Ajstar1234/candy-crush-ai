from board import Board
from agent import Agent
from game import Game

import random


def simulate_game(objective_aware=True, seed=None, max_turns=8, target_candy_count=10):
    if seed is not None:
        random.seed(seed)

    board = Board(5, 5)
    agent = Agent()
    game = Game(
        max_turns=max_turns,
        target_candy_type=1,
        target_candy_count=target_candy_count
    )

    board.create_board()
    board.resolve_board()

    while not game.is_game_over():

        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:
            if not board.reshuffle_board():
                break

        if objective_aware:
            move = agent.choose_move(board, game)
        else:
            move = agent.choose_move(board)

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
        "turns_used": game.turn,
        "candies_collected": game.collect_candies
    }

    return results

def run_simulations(num_games=100, objective_aware=True, max_turns=8, target_candy_count=10):
    if num_games <= 0:
        raise ValueError("num_games must be greater than zero")
    
    results = []

    for i in range(num_games):
        game_result = simulate_game(objective_aware=objective_aware, seed=i, max_turns=max_turns, target_candy_count=target_candy_count)
        results.append(game_result)

    games_won = sum(result["won"] for result in results)

    win_rate = (games_won / num_games) * 100

    average_score = sum(result["score"] for result in results) / num_games

    average_turns_used = sum(result["turns_used"] for result in results) / num_games

    average_candies_collected = sum(result["candies_collected"] for result in results) / num_games

    return {
        "num_games": num_games,
        "games_won": games_won,
        "win_rate": win_rate,
        "average_score": average_score,
        "average_turns_used": average_turns_used,
        "average_candies_collected": average_candies_collected,
        "results": results
    }

if __name__ == "__main__":
    greedy_stats = run_simulations(100, objective_aware=False)
    objective_stats = run_simulations(100, objective_aware=True)

    print("Greedy agent:")
    print(f"Win rate: {greedy_stats['win_rate']:.2f}%")
    print(f"Average turns: {greedy_stats['average_turns_used']:.2f}")

    print("\nObjective-aware agent:")
    print(f"Win rate: {objective_stats['win_rate']:.2f}%")
    print(f"Average turns: {objective_stats['average_turns_used']:.2f}")