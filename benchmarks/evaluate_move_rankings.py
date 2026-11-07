import random

from board import Board
from game import Game
from agent import Agent
from ml_agent import MLAgent

def evaluate_move_ranking(num_boards=100, seed=1000):
    heuristic_agent = Agent()

    ml_agent = MLAgent(
        input_size=734,
        checkpoint_path="models/best_move_model_9x9.pth"
    )

    correct_predictions = 0
    total_evaluations = 0
    total_score_gap = 0
    total_random_accuracy = 0
    total_random_score_gap = 0

    for i in range(num_boards):
        random.seed(seed + i)

        board = Board(9, 9, num_candy_types=6)

        board.create_board()
        board.resolve_board()
        
        game = Game(max_turns=8, target_candy_type=1, target_candy_count=10)

        valid_moves = board.get_valid_moves()

        if not valid_moves:
            continue

        heuristic_scores = []
        for move in valid_moves:
            heuristic_scores.append(heuristic_agent.evaluate_move(board, move, game))

        best_heuristic_score = max(heuristic_scores)

        optimal_moves = sum(
            score == best_heuristic_score
            for score in heuristic_scores
        )

        random_accuracy = optimal_moves / len(valid_moves)

        average_random_move_score = sum(heuristic_scores) / len(heuristic_scores)
        random_score_gap = best_heuristic_score - average_random_move_score

        total_random_accuracy += random_accuracy
        total_random_score_gap += random_score_gap

        ml_move = ml_agent.choose_move(board, game)
        ml_heuristic_score = heuristic_agent.evaluate_move(board, ml_move, game)
        score_gap = best_heuristic_score - ml_heuristic_score
        total_evaluations += 1
        total_score_gap += score_gap

        if score_gap == 0:
            correct_predictions += 1

    if total_evaluations == 0:
        raise ValueError("No valid board states were evaluated")

    move_ranking_accuracy = correct_predictions / total_evaluations * 100
    average_heuristic_score_gap = total_score_gap / total_evaluations
    average_random_accuracy = total_random_accuracy / total_evaluations * 100
    average_random_score_gap = total_random_score_gap / total_evaluations


    print(f"Boards evaluated: {total_evaluations}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Move-ranking accuracy: {move_ranking_accuracy:.2f}%")
    print(f"Average heuristic score gap: {average_heuristic_score_gap:.2f}")
    print(f"Random optimal-move accuracy: {average_random_accuracy:.2f}%")
    print(f"Random average heuristic score gap: {average_random_score_gap:.2f}")

if __name__ == "__main__":
    evaluate_move_ranking(num_boards=100, seed=1000)