from snake_game.direction import Direction
import numpy as np
import pygame
import random

# Configurações do jogo
WIDTH: int = 400
HEIGHT: int = 400
BLOCK_SIZE: int = 20


class SnakeGame:
    def __init__(self) -> None:
        self.score: int = 0
        self.width: int = WIDTH
        self.height: int = HEIGHT
        self.block_size: int = BLOCK_SIZE
        self.direction: Direction = random.choice(list(Direction))
        self.food: list[int] = []
        self.snake: list[list[int]] = []

        self.reset()

    def _get_random_start_position(self) -> list[int]:
        margin: float = 0.3
        x_min = int(self.width * margin // self.block_size) * self.block_size
        x_max = int(self.width * (1 - margin) // self.block_size) * self.block_size
        y_min = int(self.height * margin // self.block_size) * self.block_size
        y_max = int(self.height * (1 - margin) // self.block_size) * self.block_size

        x = random.randrange(x_min, x_max + self.block_size, self.block_size)
        y = random.randrange(y_min, y_max + self.block_size, self.block_size)
        return [x, y]

    def _get_food_position(self) -> list[int]:
        return self.food

    def _get_head_position(self) -> list[int]:
        return self.snake[0]

    def distance_head_to_food(self) -> int:
        head_x, head_y = self._get_head_position()
        food_x, food_y = self._get_food_position()
        return abs(food_x - head_x) + abs(food_y - head_y)

    def reset(self) -> np.ndarray:
        self.snake = [self._get_random_start_position()]
        self.direction = random.choice(list(Direction))  # Direção inicial (para direita)
        self.food = self.spawn_food()
        self.score = 0
        return self.get_state()

    def spawn_food(self) -> list[int]:
        while True:
            food_position = [random.randrange(0, self.width, self.block_size),
                             random.randrange(0, self.height, self.block_size)]
            if food_position not in self.snake:
                return food_position

    def step(self, action: Direction) -> tuple[np.ndarray, int, bool]:
        self.direction = action
        reward: int = 0

        # Verifica distância inicial da cabeça até a comida
        distance_food_before: int = self.distance_head_to_food()

        # Movimentar a cobra
        dx, dy = self.direction.value
        head: list[int] = [self.snake[0][0] + dx * self.block_size,
                           self.snake[0][1] + dy * self.block_size]
        self.snake.insert(0, head)

        # Checar se comeu a comida
        if head == self.food:
            self.food = self.spawn_food()
            self.score += 1
        else:
            self.snake.pop()

        # Checar colisões
        done: bool = (head[0] < 0 or head[0] >= self.width or
                      head[1] < 0 or head[1] >= self.height or
                      head in self.snake[1:])

        # Verifica distância final da cabeça até a comida
        distance_food_after: int = self.distance_head_to_food()

        # Calcula a recompensa
        if head == self.food:
            # Se comeu a comida: +10 pontos
            reward: int = 10
        elif done:
            # Se perdeu o jogo: -10 pontos
            reward: int = -10
        elif distance_food_after < distance_food_before:
            # Se ficou mais próximo da comida: +1 ponto
            reward: int = 1
        # Se ficou na mesma distância ou se afastou: 0 pontos

        return self.get_state(), reward, done

    def get_state(self) -> np.ndarray:
        # Retorna o estado do jogo como uma matriz 2D ou uma lista
        grid = np.zeros((self.width // self.block_size, self.height // self.block_size))

        for x, y in self.snake:
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y // self.block_size][x // self.block_size] = 1

        food_x, food_y = self.food
        grid[food_y // self.block_size][food_x // self.block_size] = 2
        return grid.flatten()

    def render(self, screen: pygame.Surface) -> None:
        screen.fill((0, 0, 0))  # Limpa a tela

        # Desenha a cobra
        for x, y in self.snake:
            pygame.draw.rect(screen, (0, 255, 0), (x, y, self.block_size, self.block_size))

        # Desenha a comida
        food_x, food_y = self.food
        pygame.draw.rect(screen, (255, 0, 0), (food_x, food_y, self.block_size, self.block_size))

        pygame.display.flip()  # Atualiza a tela
