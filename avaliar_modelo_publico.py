import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Carregar modelo publico
modelo = joblib.load("modelo_transito_publico.joblib")

# 2. Carregar teste independente
teste = pd.read_csv("teste_independente.csv")
treino = pd.read_csv("dados_treinamento_publico.csv")

texto = "reclamacao_labeled"

# 3. Verificar integridade do teste
assert len(teste) == 100
assert not teste[[texto, "categoria"]].isna().any().any()
assert not teste[texto].isin(treino[texto]).any()

X = teste[texto]
y = teste["categoria"]

# 4. Realizar previsoes
previsoes = modelo.predict(X)

# 5. Calcular metricas
acuracia = accuracy_score(y, previsoes)

f1 = f1_score(
    y,
    previsoes,
    average="macro",
    zero_division=0
)

acertos = int((y == previsoes).sum())
erros = len(y) - acertos

relatorio = classification_report(
    y,
    previsoes,
    zero_division=0
)

# 6. Exibir resultados
resultado = (
    f"MODELO PUBLICO - TESTE INDEPENDENTE\n"
    f"Total de registros: {len(teste)}\n"
    f"Acuracia: {acuracia:.2%}\n"
    f"F1-score macro: {f1:.4f}\n"
    f"Acertos: {acertos}\n"
    f"Erros: {erros}\n\n"
    f"{relatorio}"
)

print(resultado)

# 7. Salvar relatorio
with open(
    "avaliacao_final_publico.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(resultado)

# 8. Gerar matriz de confusao
categorias = modelo.classes_

matriz = confusion_matrix(
    y,
    previsoes,
    labels=categorias
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categorias,
    yticklabels=categorias
)

plt.title("Matriz de Confusao - Modelo Publico")
plt.xlabel("Categoria prevista")
plt.ylabel("Categoria real")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "matriz_final_publico.png",
    dpi=300
)

plt.close()

print("\nAvaliacao final concluida com sucesso!")