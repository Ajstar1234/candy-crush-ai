import torch
import numpy as np

from ml_model import MoveEvaluationModel
from ml_features import encode_state_move, encode_state_move_with_target_mask


class MLAgent:
    def __init__(self, input_size=180, checkpoint_path="models/best_move_model.pth", use_target_mask=False):
        self.model = MoveEvaluationModel(input_size=input_size)

        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        )

        self.model.eval()
        self.use_target_mask = use_target_mask

    def choose_move(self, board, game=None):
        valid_moves = board.get_valid_moves()

        if len(valid_moves) == 0:
            return None

        move_features = []

        for move in valid_moves:
            if self.use_target_mask:
                move_features.append(encode_state_move_with_target_mask(board, move, game))
            else:
                move_features.append(encode_state_move(board, move, game))

        features_array = np.array(move_features)
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        with torch.no_grad():
            predicted_scores = self.model(features_tensor).squeeze(-1)

        highest_score_index = torch.argmax(predicted_scores)

        best_move = valid_moves[highest_score_index.item()]

        return best_move
