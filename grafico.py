import ast
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Configuração do LaTeX no matplotlib
plt.rcParams.update({
    # "text.usetex": True,  # Ativa suporte ao LaTeX
    "font.family": "serif",  # Usa fonte serifada (estilo acadêmico)
    "font.serif": "Computer Modern Roman",
    "axes.labelsize": 14,  # Tamanho da fonte dos eixos
    "axes.titlesize": 14,  # Tamanho do título
    "legend.fontsize": 10,  # Tamanho da legenda
    "xtick.labelsize": 10,  # Tamanho dos rótulos do eixo X
    "ytick.labelsize": 10,  # Tamanho dos rótulos do eixo Y
})

# Configuration of the plots
sns.set_theme(context="paper",
              style="whitegrid",
              palette="viridis")
BASE_PATH: str = "./models/"
DPI: int = 300
FORMAT: str = "pdf"


def grafico_reward(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))  # Define o tamanho da figura
    plt.plot(df.index,
             df["Reward"],
             marker="o",
             linestyle="-",
             color="b",
             label="Recompensa")  # Linha azul com marcadores

    plt.xlabel("Episódio")  # Nome do eixo X
    plt.ylabel("Recompensa")  # Nome do eixo Y
    plt.title("Recompensa por episódio")  # Título do gráfico

    plt.legend()  # Exibir legenda
    plt.grid(True)  # Adicionar grade para melhor visualização
    plt.tight_layout()  # Ajusta o espaçamento para evitar margens extras
    plt.savefig(f"{BASE_PATH}/reward.{FORMAT}", dpi=DPI, bbox_inches="tight")


def grafico_time(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))  # Define o tamanho da figura
    plt.plot(df.index,
             df["Time"],
             marker="o",
             linestyle="-",
             color="b",
             label="Tempo")  # Linha azul com marcadores

    plt.xlabel("Episódio")  # Nome do eixo X
    plt.ylabel("Tempo [s]")  # Nome do eixo Y
    plt.title("Tempo de treinamento por episódio")  # Título do gráfico

    plt.legend()  # Exibir legenda
    plt.grid(True)  # Adicionar grade para melhor visualização
    plt.tight_layout()  # Ajusta o espaçamento para evitar margens extras
    plt.savefig(f"{BASE_PATH}/time.{FORMAT}", dpi=DPI, bbox_inches="tight")


if __name__ == '__main__':
    data: list[list]
    with open(f"{BASE_PATH}/rewards.txt", "r", encoding="UTF-8") as file:
        data = ast.literal_eval(file.read())

    data: pd.DataFrame = pd.DataFrame(data[1:], columns=data[0]).set_index("Episode")

    # print(data.dtypes)
    # print(data.head())
    print(f'{data["Time"].cumsum().iloc[-1]} segundos')
    print(f'{data["Time"].cumsum().iloc[-1] / 60} minutos')
    print(f'{data["Time"].cumsum().iloc[-1] / 60 / 60} horas')
    print(f'{data["Time"].mean()} segundos por época')

    grafico_reward(data)
    grafico_time(data)
