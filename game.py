# Stores and updates game logic and scoring
class Game:

    def __init__(self, score = 0, turn = 0, max_turns = 10, target_score = 500, target_candy_type = None, target_candy_count = 0):
        self.score = score
        self.turn = turn
        self.max_turns = max_turns
        self.target_score = target_score
        self.target_candy_type = target_candy_type
        self.target_candy_count = target_candy_count
        self.collect_candies = 0
    
    # Updates the score of the game
    def update_score(self, points):
        self.score += points

    # Updates the turn of the game by 1
    def update_turn(self):
        self.turn += 1

    #Check if the player won
    def is_won(self):
        if self.target_candy_type is not None:
            return self.collect_candies >= self.target_candy_count
        else:
            return self.score >= self.target_score

    #Check if the player lost
    def is_lost(self):
        return self.turn >= self.max_turns and not self.is_won()

    #Check if the game is over
    def is_game_over(self):
        return self.is_won() or self.is_lost()

    def update_collected_candies(self, amount):
        self.collect_candies += amount