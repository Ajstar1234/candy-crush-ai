# Creates random candy to fill the game board
class Candy:

    def __init__(self, type, powerup = None):
        self.type = type
        self.powerup = powerup

    def __str__(self):
        return self.create_candy()

    # Randomly create a candy type based on the type it was given 
    def create_candy(self):
        if self.type == 1:
            return "Red"
        elif self.type == 2:
            return "Green"
        elif self.type == 3:
            return "Blue"
        elif self.type == 4:
            return "Purple"

    def create_vertical_striped_powerup(self):
        self.powerup = "Vertical Striped"

    def create_horizontal_striped_powerup(self):
        self.powerup = "Horizontal Striped"

    def create_colorbomb_powerup(self):
        self.powerup = "Colorbomb"

    def create_wrapped_powerup(self):
        self.powerup = "Wrapped"

    def create_fish_powerup(self):
        self.powerup = "Fish"
    