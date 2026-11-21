import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("comparacao_final.csv")

x = np.arange(len(df))
largura = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

barras1 = ax.bar(
    x - largura / 2,
    df["Acuracia"] * 100,
    largura,
    label="Acurácia"
)

barras2 = ax.bar(
    x + largura / 2,
    df["F1_macro"] * 100,
    largura,
    label="F1-score macro"
)

ax.bar_label(barras1, fmt="%.1f%%", padding=3)
ax.bar_label(barras2, fmt="%.1f%%", padding=3)

ax.set_title("Comparação de desempenho - Trânsito IA")
ax.set_ylabel("Desempenho (%)")
ax.set_xlabel("Versão do modelo")
ax.set_xticks(x)
ax.set_xticklabels(df["Modelo"])
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("grafico_comparacao_final.png", dpi=300)
plt.close()

print("Gráfico comparativo atualizado com sucesso!")