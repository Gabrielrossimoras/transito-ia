import pandas as pd

df = pd.read_csv(
    "ConsumerBR.csv",
    nrows=10000
)

df.to_csv(
    "amostra_consumerbr.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Amostra criada com sucesso!")
print("Total de registros:", len(df))
print("Colunas:", df.columns.tolist())

