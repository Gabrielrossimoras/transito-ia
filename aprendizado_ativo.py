
import pandas as pd
import numpy as np
import joblib

# Carregar o modelo experimental
modelo = joblib.load("modelo_transito.joblib")

# Carregar as bases de dados
base = pd.read_csv("reclamacoes_transito_limpo.csv")
dados = pd.read_csv("reclamacoes_direcionadas.csv")
amostra = pd.read_csv("reclamacoes_rotuladas.csv")

dados["categoria"] = dados["categoria"].astype(int)

categorias = {
    0: "Não relacionado ao trânsito",
    1: "Transferência de veículos",
    2: "Licenciamento, IPVA e DPVAT",
    3: "Multas de trânsito",
    4: "CNH e habilitação",
    5: "Financiamento e gravame",
    6: "Outros serviços de trânsito"
}

# Excluir reclamações já rotuladas
# e preservar a amostra aleatória original
candidatos = base[
    ~base["reclamacao_labeled"].isin(
        dados["reclamacao_labeled"]
    )
    & ~base["reclamacao_labeled"].isin(
        amostra["reclamacao_labeled"]
    )
].copy().reset_index(drop=True)

print("Analisando reclamações disponíveis...")

# Estimar probabilidades de classificação
probabilidades = modelo.predict_proba(
    candidatos["reclamacao_labeled"]
)

classes = list(modelo.classes_)

# Calcular a incerteza do modelo
ordenadas = np.sort(probabilidades, axis=1)

candidatos["incerteza"] = (
    1 - (ordenadas[:, -1] - ordenadas[:, -2])
)

# Armazenar probabilidades de cada categoria
for i, classe in enumerate(classes):
    candidatos[f"prob_{classe}"] = probabilidades[:, i]

candidatos["previsao"] = modelo.predict(
    candidatos["reclamacao_labeled"]
)

print("Reclamações disponíveis:", len(candidatos))

while not candidatos.empty:

    print("\n" + "=" * 70)
    print("APRENDIZADO ATIVO")
    print("Total rotulado:", len(dados))

    contagem = dados["categoria"].value_counts()

    for codigo, nome in categorias.items():
        total = contagem.get(codigo, 0)
        print(f"{codigo} - {nome}: {total}")

    print("\nI - Buscar reclamação incerta")
    print("S - Salvar e sair")

    alvo = input(
        "\nCategoria que deseja melhorar: "
    ).strip().lower()

    if alvo == "s":
        break

    if alvo == "i":
        indice = candidatos["incerteza"].idxmax()

    elif alvo in ["0", "1", "2", "3", "4", "5", "6"]:

        # Priorizar candidatos da categoria desejada
        pontuacao = (
            0.7 * candidatos[f"prob_{alvo}"]
            + 0.3 * candidatos["incerteza"]
        )

        indice = pontuacao.idxmax()

    else:
        print("Opção inválida.")
        continue

    registro = candidatos.loc[indice]

    print("\n" + "=" * 70)
    print("Empresa:", registro["empresa"])

    print("\nRECLAMAÇÃO:")
    print(registro["reclamacao_labeled"])

    print("\nCLASSIFICAÇÃO MANUAL:")

    for codigo, nome in categorias.items():
        print(f"{codigo} - {nome}")

    print("P - Pular reclamação")
    print("S - Salvar e sair")

    escolha = input(
        "\nCategoria correta: "
    ).strip().lower()

    if escolha == "s":
        break

    if escolha == "p":
        candidatos = candidatos.drop(indice)
        continue

    if escolha not in ["0", "1", "2", "3", "4", "5", "6"]:
        print("Opção inválida.")
        continue

    # Revelar a previsão após a classificação humana
    previsto = int(registro["previsao"])

    print("\nPrevisão do modelo:", categorias[previsto])
    print("Sua classificação:", categorias[int(escolha)])

    if previsto == int(escolha):
        print("O modelo acertou!")
    else:
        print("Classificação corrigida pelo usuário.")

    novo = pd.DataFrame([{
        "empresa": registro["empresa"],
        "reclamacao_labeled": registro["reclamacao_labeled"],
        "categoria": int(escolha)
    }])

    dados = pd.concat(
        [dados, novo],
        ignore_index=True
    )

    dados.to_csv(
        "reclamacoes_direcionadas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    candidatos = candidatos.drop(indice)

    print("\nClassificação salva!")

print("\nAprendizado ativo encerrado.")
print("Total de registros rotulados:", len(dados))
