
import pandas as pd
import os

arquivo = "teste_independente.csv"
texto = "reclamacao_labeled"

categorias = {
    "0": "Não relacionado ao trânsito",
    "1": "Transferência de veículos",
    "2": "Licenciamento, IPVA e DPVAT",
    "3": "Multas de trânsito",
    "4": "CNH e habilitação",
    "5": "Financiamento e gravame",
    "6": "Outros serviços de trânsito"
}

# Carregar registros utilizados no desenvolvimento
dados = pd.read_csv(
    "versao_2/reclamacoes_direcionadas.csv"
)

# Recuperar progresso ou criar amostra independente
if os.path.exists(arquivo):

    df = pd.read_csv(
        arquivo,
        dtype={"categoria": str}
    )

else:

    base = pd.read_csv(
        "reclamacoes_rotuladas.csv"
    )

    # Selecionar somente registros ainda não rotulados
    candidatos = base[
        base["categoria"].isna()
        & ~base[texto].isin(dados[texto])
    ].copy()

    candidatos = candidatos.drop_duplicates(
        subset=[texto]
    )

    df = candidatos.sample(
        n=min(100, len(candidatos)),
        random_state=123
    ).reset_index(drop=True)

    df["categoria"] = ""

df["categoria"] = df["categoria"].fillna("")

# Garantir que os textos não foram usados anteriormente
assert not df[texto].isin(dados[texto]).any()

df.to_csv(
    arquivo,
    index=False,
    encoding="utf-8-sig"
)

# Rotulagem independente
for i in df.index:

    if df.at[i, "categoria"] != "":
        continue

    concluidos = (df["categoria"] != "").sum()

    print("\n" + "=" * 70)
    print("TESTE INDEPENDENTE")
    print(f"Progresso: {concluidos}/{len(df)}")

    print("\nEmpresa:", df.at[i, "empresa"])
    print("\nRECLAMAÇÃO:")
    print(df.at[i, texto])

    print("\nCATEGORIAS:")

    for codigo, nome in categorias.items():
        print(f"{codigo} - {nome}")

    print("S - Salvar e sair")

    while True:

        escolha = input(
            "\nCategoria correta: "
        ).strip().lower()

        if escolha == "s":
            print("Progresso salvo!")
            raise SystemExit

        if escolha in categorias:
            df.at[i, "categoria"] = escolha
            break

        print("Opção inválida.")

    df.to_csv(
        arquivo,
        index=False,
        encoding="utf-8-sig"
    )

    print("Classificação salva!")

print("\nRotulagem do teste concluída!")
print("\nDistribuição das categorias:")
print(df["categoria"].value_counts().sort_index())
