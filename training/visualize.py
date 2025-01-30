import matplotlib.pyplot as plt
import numpy as np
import random

def plot_rewards(rewards):
    """
    Plota a recompensa total por episódio para acompanhar o progresso do treinamento.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Recompensa Total", color='blue')
    plt.title('Recompensa Total por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_moving_average(rewards, window=50):
    """
    Plota a média móvel das recompensas para suavizar o gráfico.
    """
    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg, label=f'Média Móvel (janela {window})', color='red')
    plt.title(f'Média Móvel das Recompensas por Episódio (janela {window})')
    plt.xlabel('Episódio')
    plt.ylabel('Média Móvel de Recompensa')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Exemplo de visualização
    rewards = [random.randint(0, 10) for _ in range(1000)]  # Substitua por suas recompensas
    plot_rewards(rewards)
    plot_moving_average(rewards)
