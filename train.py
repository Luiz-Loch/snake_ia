import matplotlib.pyplot as plt
import numpy as np
import pickle
from snake_game import SnakeGame
from training import DQNAgent
from training import ReplayMemory

# Configurações do treinamento
EPISODES: int = 10  # Número de episódios de treinamento
BATCH_SIZE: int = 16  # Tamanho do lote para atualização
GAMMA: float = 0.95  # Fator de desconto para o Q-learning
EPSILON_START: float = 1.0  # Taxa de exploração inicial
EPSILON_MIN: float = 0.5  # Taxa de exploração mínima
EPSILON_DECAY: float = 0.995  # Taxa de decaimento da exploração
LEARNING_RATE: float = 0.001  # Taxa de aprendizado


def train():
    game: SnakeGame = SnakeGame()
    state_size: int = game.get_state().shape[0]  # Tamanho do estado

    agent: DQNAgent = DQNAgent(state_size)

    # Tente carregar a memória salva, ou crie uma nova se não existir
    try:
        with open("./replay_memory.pkl", "rb") as f:
            replay_memory = pickle.load(f)
        print(f"Replay Memory carregada com {len(replay_memory)} experiências.")
    except FileNotFoundError:
        replay_memory: ReplayMemory = ReplayMemory(100_000)
        print("Replay Memory não encontrada, inicializando vazia.")

    epsilon: float = EPSILON_START  # Inicia com exploração máxima
    rewards: list = []  # Para armazenar as recompensas por episódio

    for episode in range(EPISODES):
        state: np.ndarray = game.reset()
        done: bool = False
        total_reward: int = 0
        step_count: int = 0

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = game.step(action)
            total_reward += reward

            # Armazena a experiência
            replay_memory.add(state, action, reward, next_state, done)

            # Treina o agente
            agent.train(replay_memory, BATCH_SIZE, GAMMA)

            state = next_state

            # Limite de 150 jogadas sem pontuar
            if reward != 10:
                step_count += 1
            if step_count == 150:
                done = True

        # Decai a taxa de exploração
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

        # Salvar o modelo periodicamente
        # if (episode + 1) % 100 == 0:
        #     agent.model.save(f"./models/snake_model_{episode + 1:03}.h5")

        agent.model.save(f"./models/snake_model_{episode + 1:03}.keras")

    # Visualizar o progresso após o treinamento
    visualize(rewards)


def visualize(rewards):
    # Visualizar a recompensa por episódio
    plt.plot(rewards)
    plt.title("Recompensa por Episódio")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Total")
    plt.show()
    plt.savefig("./figures/recompensa.png", dpi=300)


if __name__ == "__main__":
    train()
