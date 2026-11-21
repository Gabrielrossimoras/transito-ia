
import pandas as pd
import os

arquivo_base = "reclamacoes_transito_limpo.csv"
arquivo_rotulado = "reclamacoes_rotuladas.csv"
arquivo_saida = "reclamacoes_direcionadas.csv"

categorias = {
    "0": "Não relacionado ao trânsito",
    "1": "Transferência de veículos",
    "2": "Licenciamento, IPVA e DPVAT",
    "3": "Multas de trânsito",
    "4": "CNH e habilitação",
    "5": "Financiamento e gravame",
    "6": "Outros serviços de trânsito"
}

termos = {
    "1": r"transferência|transferir|atpv|comunicação de venda",
    "2": r"ipva|licenciamento|detran|dpvat|crlv",
    "3": r"multa de trânsito|multas de trânsito|infração de trânsito",
    "4": r"renovação da cnh|renovar a cnh|autoescola|exame psicotécnico|primeira habilitação|suspensão da cnh|cassação da cnh|curso de reciclagem|segunda via da cnh|exame de direção|carteira de habilitação",
    "5": r"gravame|financiamento|alienação fiduciária",
    "6": r"vistoria|emplacamento|placa mercosul"
}

meta = 30

base = pd.read_csv(arquivo_base)

# Preservar a amostra aleatória original
amostra = pd.read_csv(
    arquivo_rotulado,
    dtype={"categoria": str}
)

amostra["categoria"] = amostra["categoria"].fillna("")

# Recuperar classificações direcionadas anteriores
if os.path.exists(arquivo_saida):
    dados = pd.read_csv(
        arquivo_saida,
        dtype={"categoria": str}
    )
else:
    dados = amostra[
        amostra["categoria"].isin(categorias)
    ].copy()

# Excluir a amostra aleatória da busca direcionada
base = base[
    ~base["reclamacao_labeled"].isin(
        amostra["reclamacao_labeled"]
    )
]

while True:

    contagem = dados["categoria"].value_counts()

    print("\n" + "=" * 70)
    print("DISTRIBUIÇÃO ATUAL")

    for codigo, nome in categorias.items():
        quantidade = contagem.get(codigo, 0)
        print(f"{codigo} - {nome}: {quantidade}/{meta}")

    print("\nS - Salvar e sair")

    alvo = input(
        "\nQual categoria deseja buscar? "
    ).strip()

    if alvo.lower() == "s":
        break

    if alvo not in termos:
        print("Escolha uma categoria de 1 a 6.")
        continue

    candidatos = base[
        base["reclamacao_labeled"].str.contains(
            termos[alvo],
            case=False,
            na=False,
            regex=True
        )
        & ~base["reclamacao_labeled"].isin(
            dados["reclamacao_labeled"]
        )
    ]

    if candidatos.empty:
        print("Nenhum candidato disponível.")
        continue

    registro = candidatos.sample(1).copy()

    print("\n" + "=" * 70)
    print("Empresa:", registro["empresa"].iloc[0])
    print("\nRECLAMAÇÃO:")
    print(registro["reclamacao_labeled"].iloc[0])

    print("\nCATEGORIAS:")

    for codigo, nome in categorias.items():
        print(f"{codigo} - {nome}")

    print("S - Salvar e sair")

    escolha = input(
        "\nClassificação correta: "
    ).strip()

    if escolha.lower() == "s":
        break

    if escolha not in categorias:
        print("Categoria inválida.")
        continue

    registro["categoria"] = escolha

    dados = pd.concat(
        [dados, registro],
        ignore_index=True
    )

    dados.to_csv(
        arquivo_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nClassificação salva!")

dados.to_csv(
    arquivo_saida,
    index=False,
    encoding="utf-8-sig"
)

print("\nProgresso salvo!")
print("Total classificado:", len(dados))
