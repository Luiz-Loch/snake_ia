FOR_NOTEBOOK: bool = True
FPS: int = 5

import pygame
import pickle

if not FOR_NOTEBOOK:
    from snake_game import SnakeGame, Direction
    from training import ReplayMemory

# #############################
if FOR_NOTEBOOK:
    from collections import deque
    from enum import Enum
    import random
    import numpy as np


    class Direction(Enum):
        LEFT: tuple[int, int] = (-1, 0)
        RIGHT: tuple[int, int] = (1, 0)
        UP: tuple[int, int] = (0, -1)
        DOWN: tuple[int, int] = (0, 1)


    class SnakeGame:
        def __init__(self) -> None:
            self.score: int = 0
            self.width: int = 400
            self.height: int = 400
            self.block_size: int = 20
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

            # Movimentar a cobra
            dx, dy = self.direction.value
            head = [self.snake[0][0] + dx * self.block_size,
                    self.snake[0][1] + dy * self.block_size]
            self.snake.insert(0, head)

            # Checar se comeu a comida
            if head == self.food:
                self.food = self.spawn_food()
                self.score += 1
            else:
                self.snake.pop()

            # Checar colisões
            done = (head[0] < 0 or head[0] >= self.width or
                    head[1] < 0 or head[1] >= self.height or
                    head in self.snake[1:])
            reward = 10 if head == self.food else -1 if done else 0

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


    class ReplayMemory:
        def __init__(self, capacity: int) -> None:
            self.capacity: int = capacity
            self.memory: deque = deque(maxlen=capacity)

        def __len__(self) -> int:
            return len(self.memory)

        def add(self, state: np.ndarray, action: Direction, reward: int, next_state: np.ndarray, done: bool) -> None:
            self.memory.append((state, action, reward, next_state, done))

        def sample(self, batch_size: int) -> list:
            return random.sample(self.memory, batch_size)

        def clear(self) -> None:
            self.memory.clear()

        def is_full(self) -> bool:
            return len(self.memory) == self.capacity


############################

def play() -> None:
    pygame.init()
    game = SnakeGame()
    screen = pygame.display.set_mode((game.width, game.height))
    clock = pygame.time.Clock()
    running: bool = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Controles manuais
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and game.direction != Direction.RIGHT:
                    game.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and game.direction != Direction.LEFT:
                    game.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and game.direction != Direction.DOWN:
                    game.direction = Direction.UP
                elif event.key == pygame.K_DOWN and game.direction != Direction.UP:
                    game.direction = Direction.DOWN

        state, reward, done = game.step(game.direction)

        if done:
            print(f"Game Over! Score: {game.score}")
            # print(f"{game.get_state().reshape((game.width // game.block_size), (game.height // game.block_size))}")
            game.reset()

        game.render(screen)
        clock.tick(FPS)

    pygame.quit()


def collect_human_play(memory: ReplayMemory) -> None:
    pygame.init()
    game: SnakeGame = SnakeGame()
    screen = pygame.display.set_mode((game.width, game.height))
    clock = pygame.time.Clock()
    running: bool = True

    state = game.get_state()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Controles manuais
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and game.direction != Direction.RIGHT:
                    game.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and game.direction != Direction.LEFT:
                    game.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and game.direction != Direction.DOWN:
                    game.direction = Direction.UP
                elif event.key == pygame.K_DOWN and game.direction != Direction.UP:
                    game.direction = Direction.DOWN

        # Realiza a ação e coleta os resultados
        next_state, reward, done = game.step(game.direction)
        memory.add(state, game.direction, reward, next_state, done)
        state = next_state

        if done:
            print(f"Game Over! Score: {game.score}")
            print(f"Partida finalizada! Experiências coletadas: {len(memory)}")
            game.reset()
            pygame.quit()
            return

        game.render(screen)
        clock.tick(FPS)


if __name__ == "__main__":
    # play()
    replay_memory: ReplayMemory = ReplayMemory(100_000)
    collect_human_play(replay_memory)

    # Salvar memória em um arquivo
    with open(f"replay_memory_{len(replay_memory):03}.pkl", "wb") as f:
        pickle.dump(replay_memory, f)
