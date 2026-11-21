
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# 1. Carregar os dados
df = pd.read_csv("reclamacoes_direcionadas.csv")

df = df.dropna(
    subset=["reclamacao_labeled", "categoria"]
)

df = df.drop_duplicates(
    subset=["reclamacao_labeled"]
)

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

print("Total de registros:", len(df))
print("\nDistribuição das categorias:")
print(df["categoria"].value_counts().sort_index())

# 2. Definir entradas e saídas
X = df["reclamacao_labeled"]
y = df["categoria"]

# 3. Separar treinamento e validação
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nRegistros de treinamento:", len(X_treino))
print("Registros de validação:", len(X_teste))

# 4. Criar o modelo de PLN
modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True
    )),
    ("classificador", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# 5. Treinar o modelo
print("\nTreinando o modelo...")
modelo.fit(X_treino, y_treino)

# 6. Realizar previsões
previsoes = modelo.predict(X_teste)

# 7. Avaliar os resultados
acuracia = accuracy_score(y_teste, previsoes)

f1 = f1_score(
    y_teste,
    previsoes,
    average="macro",
    zero_division=0
)

print(f"\nAcurácia: {acuracia:.2%}")
print(f"F1-score macro: {f1:.4f}")

relatorio = classification_report(
    y_teste,
    previsoes,
    labels=list(categorias.keys()),
    target_names=list(categorias.values()),
    zero_division=0
)

print("\nRelatório de classificação:")
print(relatorio)

# 8. Salvar relatório
with open(
    "resultados_experimentais.txt",
    "w",
    encoding="utf-8"
) as arquivo:
    arquivo.write(f"Total de registros: {len(df)}\n")
    arquivo.write(f"Treinamento: {len(X_treino)}\n")
    arquivo.write(f"Validação: {len(X_teste)}\n")
    arquivo.write(f"Acurácia: {acuracia:.2%}\n")
    arquivo.write(f"F1-score macro: {f1:.4f}\n\n")
    arquivo.write(relatorio)

# 9. Gerar matriz de confusão
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

plt.title("Matriz de Confusão - Modelo Experimental")
plt.xlabel("Categoria prevista")
plt.ylabel("Categoria real")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("matriz_confusao.png", dpi=300)
plt.close()

# 10. Salvar modelo experimental
joblib.dump(modelo, "modelo_transito.joblib")

print("\nTreinamento concluído!")
print("Modelo salvo: modelo_transito.joblib")
print("Matriz salva: matriz_confusao.png")
print("Relatório salvo: resultados_experimentais.txt")
