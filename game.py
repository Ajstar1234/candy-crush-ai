# Stores and updates game logic and scoring
class Game:

    def __init__(self, score = 0, turn = 0, target_score = 500):
        self.score = score
        self.turn = turn
        self.target_score = target_score
    
    # Updates the score of the game
    def update_score(self, points):
        self.score += points

    # Updates the turn of the game by 1
    def update_turn(self):
        self.turn += 1

    #Check if the player won
    def is_won(self):
        return self.score >= self.target_score