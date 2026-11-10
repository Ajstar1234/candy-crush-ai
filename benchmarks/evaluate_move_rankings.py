import random

from board import Board
from game import Game
from agent import Agent
from ml_agent import MLAgent

def evaluate_move_ranking(num_boards=100, seed=1000):
    heuristic_agent = Agent()

    original_ml_agent = MLAgent(
        input_size=734,
        checkpoint_path="models/best_move_model_9x9.pth"
    )

    target_mask_ml_agent = MLAgent(
        input_size=815,
        checkpoint_path="models/best_move_model_9x9_target_mask.pth",
        use_target_mask=True
    )

    original_correct_predictions = 0
    total_evaluations = 0
    original_total_score_gap = 0
    total_random_accuracy = 0
    total_random_score_gap = 0
    target_mask_correct_predictions = 0
    target_mask_total_score_gap = 0

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

        original_ml_move = original_ml_agent.choose_move(board, game)

        original_ml_score = heuristic_agent.evaluate_move(
            board, original_ml_move, game
        )

        original_score_gap = best_heuristic_score - original_ml_score

        original_total_score_gap += original_score_gap

        if original_score_gap == 0:
            original_correct_predictions += 1

        target_mask_ml_move = target_mask_ml_agent.choose_move(board, game)

        target_mask_ml_score = heuristic_agent.evaluate_move(
            board, target_mask_ml_move, game
        )

        target_mask_score_gap = best_heuristic_score - target_mask_ml_score

        target_mask_total_score_gap += target_mask_score_gap

        if target_mask_score_gap == 0:
            target_mask_correct_predictions += 1

        total_evaluations += 1

    if total_evaluations == 0:
        raise ValueError("No valid board states were evaluated")

    original_move_ranking_accuracy = original_correct_predictions / total_evaluations * 100
    original_average_heuristic_score_gap = original_total_score_gap / total_evaluations
    target_mask_move_ranking_accuracy = target_mask_correct_predictions / total_evaluations * 100
    target_mask_average_heuristic_score_gap = target_mask_total_score_gap / total_evaluations
    average_random_accuracy = total_random_accuracy / total_evaluations * 100
    average_random_score_gap = total_random_score_gap / total_evaluations


    print(f"Boards evaluated: {total_evaluations}")
    print(f"Original correct predictions: {original_correct_predictions}")
    print(f"Original move-ranking accuracy: {original_move_ranking_accuracy:.2f}%")
    print(f"Original average heuristic score gap: {original_average_heuristic_score_gap:.2f}")
    print(f"Target mask correct predictions: {target_mask_correct_predictions}")
    print(f"Target mask move-ranking accuracy: {target_mask_move_ranking_accuracy:.2f}%")
    print(f"Target mask average heuristic score gap: {target_mask_average_heuristic_score_gap:.2f}")
    print(f"Random optimal-move accuracy: {average_random_accuracy:.2f}%")
    print(f"Random average heuristic score gap: {average_random_score_gap:.2f}")

if __name__ == "__main__":
    evaluate_move_ranking(num_boards=100, seed=1000)