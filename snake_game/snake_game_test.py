from snake_game import SnakeGame
import numpy as np

def test_food_spawn_location() -> None:
    print("Testando se a comida não aparece no lugar da cobra...")
    game: SnakeGame = SnakeGame()
    for _ in range(100):  # Testa várias iterações
        game.snake = [[100, 100], [80, 100], [60, 100]]  # Define um corpo da cobra
        food = game.spawn_food()
        if food in game.snake:
            print("Erro: Comida apareceu dentro da cobra!")
            return
    print("Teste de comida passou!")

def test_snake_initial_location() -> None:
    print("Testando os locais iniciais da cobra...")
    game: SnakeGame = SnakeGame()
    possible_positions = set()  # Armazena os locais iniciais únicos
    for _ in range(100):  # Testa várias vezes
        game.reset()
        possible_positions.add(tuple(game.snake[0]))
    print(f"Locais iniciais possíveis: {possible_positions}")
    print("Teste de locais iniciais passou!")

def test_get_state_output() -> None:
    print("Testando a saída da função get_state...")
    game: SnakeGame = SnakeGame()
    state: np.ndarray = game.get_state()
    expected_size = (game.width // game.block_size) * (game.height // game.block_size)
    if len(state) != expected_size:
        print(f"Erro: Tamanho do estado ({len(state)}) não corresponde ao esperado ({expected_size})!")
        return
    unique_values = np.unique(state)
    print(f"Valores únicos no estado: {unique_values} (esperado: 0, 1, 2)")
    if not all(value in [0, 1, 2] for value in unique_values):
        print(f"Erro: Valores inesperados no estado: {unique_values}")
        return
    print("Teste da função get_state passou!")

def test_get_state_output_2() -> None:
    print("Testando a saída da função get_state...")
    game: SnakeGame = SnakeGame()
    print(f"{game.get_state().reshape((game.width // game.block_size), (game.height // game.block_size))}")

def test_get_state_output_3() -> None:
    print("Testando a saída da função get_state...")
    game: SnakeGame = SnakeGame()
    state: np.ndarray = game.get_state()
    print(f"{state}")
    print()
    print(f"{state[np.newaxis]}")

def test_get_state_output_4() -> None:
    print("Testando a saída da função get_state...")
    game: SnakeGame = SnakeGame()
    print(f"{game._get_food_position()}")
    print(f"{game._get_food_position()}")

if __name__ == "__main__":
    print("Iniciando testes manuais...")
    # test_food_spawn_location()
    # print()
    # test_snake_initial_location()
    # print()
    # test_get_state_output()
    # print()
    # test_get_state_output_2()
    # print()
    # test_get_state_output_3()
    print("Testes finalizados!")
