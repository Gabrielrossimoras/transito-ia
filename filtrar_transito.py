
import pandas as pd
import re

arquivo_entrada = "ConsumerBR.csv"
arquivo_saida = "reclamacoes_transito.csv"

# Palavras-chave relacionadas aos serviços de trânsito
termos = [
    r"\bdetran\b",
    r"\bciretran\b",
    r"\bsenatran\b",
    r"\bdenatran\b",
    r"\bcnh\b",
    r"\bcrlv\b",
    r"\bcrv\b",
    r"\batpv\b",
    r"\bipva\b",
    r"\blicenciamento\b",
    r"\bemplacamento\b",
    r"\bgravame\b",
    r"\bdpvat\b",
    r"multa[s]? de trânsito",
    r"transferência de veículo",
    r"documentação veicular",
    r"carteira de habilitação",
    r"vistoria veicular",
    r"renovação da habilitação"
]

padrao = "|".join(termos)

total_analisado = 0
total_filtrado = 0
primeiro_bloco = True

print("Iniciando a filtragem do ConsumerBR...")

# Leitura em blocos de 50 mil registros
for bloco in pd.read_csv(
    arquivo_entrada,
    chunksize=50000,
    low_memory=False
):
    filtro = bloco["reclamacao_labeled"].str.contains(
        padrao,
        case=False,
        na=False,
        regex=True
    )

    selecionados = bloco[filtro]

    selecionados.to_csv(
        arquivo_saida,
        mode="w" if primeiro_bloco else "a",
        header=primeiro_bloco,
        index=False,
        encoding="utf-8"
    )

    primeiro_bloco = False

    total_analisado += len(bloco)
    total_filtrado += len(selecionados)

    print(
        f"Analisados: {total_analisado:,} | "
        f"Selecionados: {total_filtrado:,}"
    )

print("\nFiltragem concluída!")
print("Total analisado:", total_analisado)
print("Total selecionado:", total_filtrado)
print("Arquivo gerado:", arquivo_saida)
