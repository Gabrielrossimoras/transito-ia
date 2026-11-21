
import pandas as pd
import os

arquivo_base = "reclamacoes_transito_limpo.csv"
arquivo_rotulado = "reclamacoes_rotuladas.csv"

categorias = {
    "0": "Não relacionado ao trânsito",
    "1": "Transferência de veículos",
    "2": "Licenciamento, IPVA e DPVAT",
    "3": "Multas de trânsito",
    "4": "CNH e habilitação",
    "5": "Financiamento e gravame",
    "6": "Outros serviços de trânsito"
}

# Recuperar progresso ou criar uma amostra
if os.path.exists(arquivo_rotulado):
    df = pd.read_csv(
        arquivo_rotulado,
        dtype={"categoria": str}
    )
else:
    base = pd.read_csv(arquivo_base)

    df = base.sample(
        n=min(500, len(base)),
        random_state=42
    ).reset_index(drop=True)

    df["categoria"] = ""

df["categoria"] = df["categoria"].fillna("")

# Rotulagem manual
for i in df.index:

    if df.at[i, "categoria"] != "":
        continue

    concluidos = (df["categoria"] != "").sum()

    print("\n" + "=" * 70)
    print(f"Progresso: {concluidos}/{len(df)}")
    print("=" * 70)

    print("\nEmpresa:", df.at[i, "empresa"])
    print("\nReclamação:")
    print(df.at[i, "reclamacao_labeled"])

    print("\nCATEGORIAS:")

    for codigo, nome in categorias.items():
        print(f"{codigo} - {nome}")

    print("S - Salvar e sair")

    while True:
        escolha = input("\nDigite a categoria: ").strip()

        if escolha.lower() == "s":
            df.to_csv(
                arquivo_rotulado,
                index=False,
                encoding="utf-8-sig"
            )

            print("Progresso salvo!")
            raise SystemExit

        if escolha in categorias:
            df.at[i, "categoria"] = escolha
            break

        print("Opção inválida. Tente novamente.")

    # Salvar após cada classificação
    df.to_csv(
        arquivo_rotulado,
        index=False,
        encoding="utf-8-sig"
    )

print("\nRotulagem concluída!")

print("\nDistribuição das categorias:")
print(df["categoria"].value_counts())
