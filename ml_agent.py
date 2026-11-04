import torch
import numpy as np

from ml_model import MoveEvaluationModel
from ml_features import encode_state_move


class MLAgent:
    def __init__(self):
        self.model = MoveEvaluationModel()

        self.model.load_state_dict(
            torch.load("best_move_model.pth", weights_only=True, map_location="cpu")
        )

        self.model.eval()

    def choose_move(self, board, game=None):
        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:
            return None

        move_features = []

        for move in valid_moves:
            move_features.append(encode_state_move(board, move, game))

        features_array = np.array(move_features)
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        with torch.no_grad():
                predicted_scores = self.model(features_tensor).squeeze(-1)

                highest_score_index = torch.argmax(predicted_scores)

                best_move = valid_moves[highest_score_index.item()]

                return best_move
