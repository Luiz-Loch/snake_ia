from tensorflow.keras import layers, Sequential
from training.replay_memory import ReplayMemory
import snake_game
import numpy as np
import random
import tensorflow as tf


class DQNAgent:
    def __init__(self, state_size: int) -> None:
        self.state_size: int = state_size
        self.action_size: int = len(snake_game.Direction)
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state: np.ndarray, epsilon: float) -> snake_game.Direction:
        if tf.random.uniform((1,)) < epsilon:
            # Ação aleatória, convertendo para a direção correta
            action_idx = random.choice(range(self.action_size))  # Índice de ação (inteiro)
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            action_idx = tf.argmax(q_values[0]).numpy()  # Índice da ação com o maior valor Q

        # Acessando a direção com o índice
        action = list(snake_game.Direction)[action_idx]  # Mapeando o índice para a direção correspondente
        return action

    def train(self, replay_memory: ReplayMemory, batch_size: int, gamma: float = 0.99) -> None:
        if len(replay_memory) < batch_size:
            return

        batch = random.sample(replay_memory.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += gamma * tf.reduce_max(self.model.predict(next_state[np.newaxis]))

            target_q = self.model.predict(state[np.newaxis], verbose=0)
            target_q[0] = target
            self.model.fit(state[np.newaxis], target_q, epochs=1, verbose=0)
