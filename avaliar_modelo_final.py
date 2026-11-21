
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# 1. Carregar o teste independente
df = pd.read_csv("teste_independente.csv")

texto = "reclamacao_labeled"

assert len(df) == 100
assert df["categoria"].notna().all()
assert df[texto].notna().all()
assert not df[texto].duplicated().any()

df["categoria"] = df["categoria"].astype(int)

categorias = {
    0: "Não relacionado ao trânsito",
    1: "Transferência de veículos",
    2: "Licenciamento, IPVA e DPVAT",
    3: "Multas de trânsito",
    4: "CNH e habilitação",
    5: "Financiamento e gravame",
    6: "Outros serviços de trânsito"
}

assert df["categoria"].isin(categorias).all()

# 2. Verificar sobreposição de dados
base = pd.read_csv(
    "versao_2/reclamacoes_direcionadas.csv"
)

assert not df[texto].isin(base[texto]).any()

X_teste = df[texto]
y_teste = df["categoria"]

# 3. Carregar os modelos já treinados
modelos = {
    "V1": joblib.load(
        "versao_1/modelo_transito.joblib"
    ),
    "V2": joblib.load(
        "versao_2/modelo_transito_v2.joblib"
    )
}

resultados = []

# 4. Avaliar cada modelo
for nome, modelo in modelos.items():

    previsoes = modelo.predict(X_teste)

    acuracia = accuracy_score(
        y_teste, previsoes
    )

    f1 = f1_score(
        y_teste,
        previsoes,
        average="macro",
        zero_division=0
    )

    acertos = int((y_teste == previsoes).sum())
    erros = len(df) - acertos

    relatorio = classification_report(
        y_teste,
        previsoes,
        labels=list(categorias.keys()),
        target_names=list(categorias.values()),
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(f"AVALIAÇÃO INDEPENDENTE - {nome}")
    print(f"Acurácia: {acuracia:.2%}")
    print(f"F1-score macro: {f1:.4f}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")
    print(relatorio)

    resultados.append({
        "Modelo": nome,
        "Acuracia": acuracia,
        "F1_macro": f1,
        "Acertos": acertos,
        "Erros": erros
    })

    df[f"previsao_{nome}"] = previsoes

    # Salvar relatório individual
    with open(
        f"avaliacao_final_{nome}.txt",
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(f"Modelo: {nome}\n")
        arquivo.write(f"Total de testes: {len(df)}\n")
        arquivo.write(f"Acurácia: {acuracia:.2%}\n")
        arquivo.write(f"F1-score macro: {f1:.4f}\n")
        arquivo.write(f"Acertos: {acertos}\n")
        arquivo.write(f"Erros: {erros}\n\n")
        arquivo.write(relatorio)

    # Gerar matriz de confusão
    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=list(categorias.keys())
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(categorias.values()),
        yticklabels=list(categorias.values())
    )

    plt.title(f"Matriz de Confusão - Teste Final {nome}")
    plt.xlabel("Categoria prevista")
    plt.ylabel("Categoria real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        f"matriz_final_{nome}.png",
        dpi=300
    )

    plt.close()

# 5. Comparação final
comparacao = pd.DataFrame(resultados)

print("\nCOMPARAÇÃO FINAL:")
print(comparacao.to_string(index=False))

comparacao.to_csv(
    "comparacao_final.csv",
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    "previsoes_teste_final.csv",
    index=False,
    encoding="utf-8-sig"
)

# 6. Gráfico comparativo
grafico = comparacao.set_index("Modelo")[
    ["Acuracia", "F1_macro"]
] * 100

grafico.plot(
    kind="bar",
    figsize=(9, 6),
    color=["steelblue", "seagreen"],
    rot=0
)

plt.title("Comparação dos Modelos - Teste Independente")
plt.ylabel("Desempenho (%)")
plt.ylim(0, 100)
plt.legend(["Acurácia", "F1-score macro"])
plt.tight_layout()
plt.savefig("grafico_comparacao_final.png", dpi=300)
plt.close()

print("\nAvaliação final concluída com sucesso!")
