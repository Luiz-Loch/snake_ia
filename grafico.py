import ast
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Configuration of the plots
sns.set_theme(context="paper",
              style="whitegrid",
              palette="viridis")
DPI: int = 300
FORMAT: str = "svg"


def grafico(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))  # Define o tamanho da figura
    plt.plot(df.index, df["Reward"], marker="o", linestyle="-", color="b", label="Reward")  # Linha azul com marcadores
    plt.xlabel("Episode")  # Nome do eixo X
    plt.ylabel("Reward")  # Nome do eixo Y
    plt.title("Reward por Episode")  # Título do gráfico
    plt.legend()  # Exibir legenda
    plt.grid(True)  # Adicionar grade para melhor visualização
    # plt.show()  # Mostrar o gráfico
    plt.savefig(f"./models/reward.{FORMAT}", dpi=DPI)


if __name__ == '__main__':
    data: list[list]
    with open("./models/rewards.txt", "r", encoding="UTF-8") as file:
        data = ast.literal_eval(file.read())

    data: pd.DataFrame = pd.DataFrame(data[1:], columns=data[0]).set_index("Episode")

    print(data.dtypes)
    print(data.head())
    print(data["Time"].cumsum()) # 6,5 horas de treinamento

    grafico(data)
