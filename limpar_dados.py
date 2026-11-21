import pandas as pd

df = pd.read_csv("reclamacoes_transito.csv")

total_original = len(df)

# Selecionar as colunas necessárias
df = df[["empresa", "reclamacao_labeled"]].copy()

# Remover registros sem texto
df = df.dropna(subset=["reclamacao_labeled"])

# Remover espaços desnecessários
df["reclamacao_labeled"] = (
    df["reclamacao_labeled"]
    .str.strip()
)

# Remover textos vazios
df = df[df["reclamacao_labeled"] != ""]

# Remover reclamações duplicadas
df = df.drop_duplicates(
    subset=["reclamacao_labeled"]
)

# Criar coluna para futura classificação
df["categoria"] = ""

# Salvar a base limpa
df.to_csv(
    "reclamacoes_transito_limpo.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Limpeza concluída!")
print("Registros originais:", total_original)
print("Registros removidos:", total_original - len(df))
print("Registros restantes:", len(df))