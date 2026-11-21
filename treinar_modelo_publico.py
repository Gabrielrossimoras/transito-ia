
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

# 1. Carregar as duas versões da base
v1 = pd.read_csv("versao_1/reclamacoes_direcionadas.csv")
v2 = pd.read_csv("reclamacoes_direcionadas.csv")

texto = "reclamacao_labeled"

v1["categoria"] = v1["categoria"].astype(int)
v2["categoria"] = v2["categoria"].astype(int)

categorias = {
    0: "Não relacionado ao trânsito",
    1: "Transferência de veículos",
    2: "Licenciamento, IPVA e DPVAT",
    3: "Multas de trânsito",
    4: "CNH e habilitação",
    5: "Financiamento e gravame",
    6: "Outros serviços de trânsito"
}

# 2. Recuperar a validação original da V1
treino_v1, teste = train_test_split(
    v1,
    test_size=0.20,
    random_state=42,
    stratify=v1["categoria"]
)

# 3. Preparar o treinamento da V2
treino = v2[
    ~v2[texto].isin(teste[texto])
].copy()

# Garantir que não existe sobreposição
assert not treino[texto].isin(teste[texto]).any()
assert len(teste) == 39
assert len(treino) == 267

X_treino = treino[texto]
y_treino = treino["categoria"]

X_teste = teste[texto]
y_teste = teste["categoria"]

print("Total de registros:", len(v2))
print("Treinamento:", len(treino))
print("Validação:", len(teste))

# 4. Arquitetura do modelo para publicação
modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
        min_df=2
    )),
    ("classificador", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])
       
# 5. Treinar a versão 2
print("\nTreinando modelo V2...")
modelo.fit(X_treino, y_treino)

# 6. Avaliar a versão 2
previsoes = modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, previsoes)

f1 = f1_score(
    y_teste,
    previsoes,
    average="macro",
    zero_division=0
)

# 7. Avaliar a V1 na mesma amostra
modelo_v1 = joblib.load(
    "versao_1/modelo_transito.joblib"
)

previsoes_v1 = modelo_v1.predict(X_teste)

acuracia_v1 = accuracy_score(
    y_teste, previsoes_v1
)

f1_v1 = f1_score(
    y_teste,
    previsoes_v1,
    average="macro",
    zero_division=0
)

# 8. Comparar os resultados
comparacao = (
    "\nCOMPARAÇÃO ENTRE MODELOS\n"
    f"Acurácia V1: {acuracia_v1:.2%}\n"
    f"Acurácia V2: {acuracia:.2%}\n"
    f"F1-score macro V1: {f1_v1:.4f}\n"
    f"F1-score macro V2: {f1:.4f}\n"
    f"Variação da acurácia: "
    f"{(acuracia-acuracia_v1)*100:+.2f} p.p.\n"
)

print(comparacao)

relatorio = classification_report(
    y_teste,
    previsoes,
    labels=list(categorias.keys()),
    target_names=list(categorias.values()),
    zero_division=0
)

print(relatorio)

# 9. Salvar relatório
with open(
   "resultados_publico.txt",
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(f"Total rotulado: {len(v2)}\n")
    arquivo.write(f"Treinamento: {len(treino)}\n")
    arquivo.write(f"Validação: {len(teste)}\n")
    arquivo.write(comparacao)
    arquivo.write("\n")
    arquivo.write(relatorio)

# 10. Gerar matriz de confusão
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

plt.title("Matriz de Confusão - Modelo V2")
plt.xlabel("Categoria prevista")
plt.ylabel("Categoria real")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("matriz_confusao_publico.png", dpi=300)
plt.close()

# 11. Salvar modelo e divisão dos dados
joblib.dump(modelo, "modelo_transito_publico.joblib")

teste.to_csv(
   "validacao_publico.csv",
    index=False,
    encoding="utf-8-sig"
)

treino.to_csv(
   "dados_treinamento_publico.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTreinamento V2 concluído!")
print("Arquivos gerados com sucesso.")
