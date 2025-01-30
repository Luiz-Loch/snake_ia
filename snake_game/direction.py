from enum import Enum

class Direction(Enum):
    LEFT: tuple[int, int] = (-1, 0)
    RIGHT: tuple[int, int] = (1, 0)
    UP: tuple[int, int] = (0, -1)
    DOWN: tuple[int, int] = (0, 1)
