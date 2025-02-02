from snake_game import SnakeGame
from training import DQNAgent
from tensorflow.keras.models import load_model
import pygame
import numpy as np

# Configurações
MODEL_PATH: str = "./models/snake_model_150.keras"  # Caminho do modelo salvo

def play():
    # Inicializar o jogo
    pygame.init()
    game: SnakeGame = SnakeGame()
    screen = pygame.display.set_mode((game.width, game.height))
    clock = pygame.time.Clock()


    # Carregar o modelo treinado
    model = load_model(MODEL_PATH)
    state_size: int = game.get_state().shape[0]
    agent: DQNAgent = DQNAgent(state_size)  # Para usar as funções do agente
    agent.model = model  # Substituir o modelo do agente pelo modelo carregado

    # Resetar o jogo
    state: np.ndarray = game.reset()
    done: bool = False

    # Loop do jogo
    while not done:
        # Fazer a previsão da ação com base no estado atual
        action = agent.act(state, epsilon=0.0)  # epsilon=0 para exploração 100% pelo modelo

        # Executar a ação no ambiente
        next_state, reward, done = game.step(action)

        game.render(screen)
        clock.tick(10) # vezes mais rápido que o jogo normal

        # Atualizar o estado atual
        state = next_state

    pygame.quit()
    print("Fim do jogo!")


if __name__ == "__main__":
    play()
