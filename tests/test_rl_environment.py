import numpy as np
import pytest

from rl_environment import RLEnvironment
from ml_features import encode_rl_state_move
from board import Board
from game import Game


def test_rl_environment_reset():
    env = RLEnvironment()

    observation = env.reset(seed=42)

    assert observation.shape == (9, 9, 2)
    assert env.board is not None
    assert env.game is not None
    assert len(env.get_valid_moves()) > 0


def test_rl_environment_reproducibility():
    env = RLEnvironment()

    first_observation = env.reset(seed=42)
    second_observation = env.reset(seed=42)

    assert np.array_equal(first_observation, second_observation)


def test_rl_environment_requires_reset():
    env = RLEnvironment()

    with pytest.raises(ValueError):
        env.get_valid_moves()

def test_rl_environment_step():
    env = RLEnvironment()
    env.reset(seed=42)

    move = env.get_valid_moves()[0]

    observation, reward, done, info = env.step(move)

    assert observation.shape == (9, 9, 2)
    assert env.game.turn == 1
    assert info["score"] == info["candies_cleared"] * 10
    assert env.game.collect_candies == env.board.cleared_candies.get(
        env.game.target_candy_type, 0
    )
    assert reward >= 0.0
    assert done == env.game.is_game_over()

def test_rl_environment_game_over():
    env = RLEnvironment(
        max_turns=1,
        target_candy_count=1000
    )

    env.reset(seed=42)

    move = env.get_valid_moves()[0]

    observation, reward, done, info = env.step(move)

    assert done is True
    assert env.game.turn == 1
    assert info["won"] is False

    with pytest.raises(ValueError):
        env.step(move)

def test_rl_environment_reward():
    env = RLEnvironment()
    env.reset(seed=42)

    move = env.get_valid_moves()[0]

    observation, reward, done, info = env.step(move)

    expected_reward = env.board.cleared_candies.get(
        env.game.target_candy_type, 0
    )

    if env.game.is_won():
        expected_reward += 10.0

    assert reward == expected_reward

def test_rl_environment_win_bonus(monkeypatch):
    env = RLEnvironment(target_candy_count=1)
    env.reset(seed=42)

    move = env.get_valid_moves()[0]

    def fake_make_move(selected_move):
        env.board.cleared_candies = {
            env.game.target_candy_type: 1
        }
        return 3

    monkeypatch.setattr(env.board, "make_move", fake_make_move)

    observation, reward, done, info = env.step(move)

    assert reward == 11.0
    assert done is True
    assert info["won"] is True
    assert env.game.collect_candies == 1

def test_rl_environment_reshuffle(monkeypatch):
    env = RLEnvironment(target_candy_count=1000)
    env.reset(seed=42)

    move = env.get_valid_moves()[0]

    state = {
        "moved": False,
        "reshuffled": False
    }

    def fake_get_valid_moves():
        if state["moved"] and not state["reshuffled"]:
            return []
        return [move]

    def fake_make_move(selected_move):
        state["moved"] = True
        env.board.cleared_candies = {}
        return 3

    def fake_reshuffle():
        state["reshuffled"] = True
        return True

    monkeypatch.setattr(env.board, "get_valid_moves", fake_get_valid_moves)
    monkeypatch.setattr(env.board, "make_move", fake_make_move)
    monkeypatch.setattr(env.board, "reshuffle_board", fake_reshuffle)

    observation, reward, done, info = env.step(move)

    assert state["reshuffled"] is True
    assert done is False
    assert env.done is False
    assert env.game.turn == 1
    assert reward == 0.0

def test_rl_environment_reshuffle_failure(monkeypatch):
    env = RLEnvironment(target_candy_count=1000)
    env.reset(seed=42)

    move = env.get_valid_moves()[0]

    state = {"moved": False}

    def fake_get_valid_moves():
        if state["moved"]:
            return []
        return [move]

    def fake_make_move(selected_move):
        state["moved"] = True
        env.board.cleared_candies = {}
        return 3

    def fake_reshuffle():
        return False

    monkeypatch.setattr(env.board, "get_valid_moves", fake_get_valid_moves)
    monkeypatch.setattr(env.board, "make_move", fake_make_move)
    monkeypatch.setattr(env.board, "reshuffle_board", fake_reshuffle)

    observation, reward, done, info = env.step(move)

    assert done is True
    assert env.done is True
    assert env.game.turn == 1
    assert info["won"] is False

    with pytest.raises(ValueError):
        env.step(move)

def test_rl_environment_reset_reshuffle_failure(monkeypatch):
    env = RLEnvironment()

    monkeypatch.setattr(
        "board.Board.get_valid_moves",
        lambda self: []
    )

    monkeypatch.setattr(
        "board.Board.reshuffle_board",
        lambda self: False
    )

    observation = env.reset(seed=42)

    assert observation.shape == (9, 9, 2)
    assert env.done is True
    assert env.game.turn == 0
    assert env.get_valid_moves() == []

def test_rl_environment_reset_reshuffle_success(monkeypatch):
    env = RLEnvironment()

    state = {"reshuffled": False}

    def fake_get_valid_moves(self):
        if not state["reshuffled"]:
            return []
        return [((0, 0), (0, 1))]

    def fake_reshuffle(self):
        state["reshuffled"] = True
        return True

    monkeypatch.setattr(
        "board.Board.get_valid_moves",
        fake_get_valid_moves
    )

    monkeypatch.setattr(
        "board.Board.reshuffle_board",
        fake_reshuffle
    )

    observation = env.reset(seed=42)

    assert state["reshuffled"] is True
    assert env.done is False
    assert env.game.turn == 0
    assert len(env.get_valid_moves()) > 0
    assert observation.shape == (9, 9, 2)


def test_encode_rl_state_move():
    board = Board(9, 9, num_candy_types=6)
    board.create_board()
    board.resolve_board()

    game = Game(
        max_turns=8,
        target_candy_type=1,
        target_candy_count=10
    )

    move = board.get_valid_moves()[0]

    features = encode_rl_state_move(board, move, game)

    assert features.shape == (817,)
    assert features[-2] == 8
    assert features[-1] == 10

    game.update_collected_candies(3)
    game.update_turn()

    updated_features = encode_rl_state_move(board, move, game)

    assert updated_features[-2] == 7
    assert updated_features[-1] == 7