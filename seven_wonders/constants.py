from enum import Enum

class CardColor(Enum):
    BROWN = "brown"
    GREY = "grey"
    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"
    GREEN = "green"
    PURPLE = "purple"

class ScienceSymbol(Enum):
    COMPASS = "compass"
    GEAR = "gear"
    TABLET = "tablet"

class GameState(Enum):
    SETUP = "setup"
    AGE_ACTIVE = "age_active"
    CARD_SELECTION = "card_selection"
    CARD_ACTION = "card_action"
    MILITARY_RESOLUTION = "military_resolution"
    GAME_OVER = "game_over"