
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Trânsito IA",
    page_icon="🚘",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

# Categorias do modelo
categorias = {
    0: "Não relacionado ao trânsito",
    1: "Transferência de veículos",
    2: "Licenciamento, IPVA e DPVAT",
    3: "Multas de trânsito",
    4: "CNH e habilitação",
    5: "Financiamento e gravame",
    6: "Outros serviços de trânsito"
}

# Carregar o modelo treinado
@st.cache_resource
def carregar_modelo():
    return joblib.load(
        BASE / "modelo_transito_publico.joblib"
    )

modelo = carregar_modelo()

# Barra lateral
with st.sidebar:

    st.title("🚘 Trânsito IA")

    st.write(
        "Sistema inteligente de classificação "
        "de reclamações sobre serviços de trânsito."
    )

    st.divider()

    st.subheader("Informações acadêmicas")

    st.write("**Instituição:** UNOESC")
    st.write("**Curso:** Ciência de Dados e IA")
    st.write("**Disciplina:** Projeto de IA")
    st.write("**Acadêmico:** Gabriel Rossi Moras")

    st.divider()

    st.write("**Modelo:** Regressão Logística")
    st.write("**Vetorização:** TF-IDF")
    st.write("**Versão:** 3.0")

# Cabeçalho
st.title("🚘 Trânsito IA")

st.write(
    "Análise Inteligente de Reclamações "
    "sobre Serviços de Trânsito"
)

st.divider()

# Indicadores
col1, col2, col3, col4 = st.columns(4)

col1.metric("Acurácia final", "81%", "+14 p.p.")
col2.metric("F1-score macro", "0,7888")
col3.metric("Registros rotulados", "406")
col4.metric("Categorias", "7")

st.divider()

# Abas
aba1, aba2, aba3 = st.tabs([
    "🔍 Classificação",
    "📊 Resultados",
    "📚 Sobre o projeto"
])

# ABA 1 - CLASSIFICAÇÃO
with aba1:

    st.header("Classificar reclamação")

    reclamacao = st.text_area(
        "Digite o texto da reclamação:",
        height=180,
        placeholder="Descreva a situação..."
    )

    if st.button(
        "Classificar reclamação",
        type="primary"
    ):

        if not reclamacao.strip():

            st.warning(
                "Digite uma reclamação para continuar."
            )

        else:

            # Realizar previsão
            previsao = int(
                modelo.predict([reclamacao])[0]
            )

            probabilidades = modelo.predict_proba(
                [reclamacao]
            )[0]

            classe_idx = list(
                modelo.classes_
            ).index(previsao)

            pontuacao = probabilidades[classe_idx]

            st.success(
                f"Categoria identificada: "
                f"{categorias[previsao]}"
            )

            st.metric(
                "Probabilidade estimada pelo modelo",
                f"{pontuacao:.2%}"
            )

            st.progress(float(pontuacao))
            if pontuacao < 0.40:
                st.warning(
                    "Classificação com baixa confiança estimada. "
                    "Recomenda-se uma análise manual da reclamação."
                )

            st.subheader(
                "Probabilidades por categoria"
            )

            df_prob = pd.DataFrame({
                "Categoria": [
                    categorias[int(c)]
                    for c in modelo.classes_
                ],
                "Probabilidade": probabilidades
            })

            df_prob = df_prob.sort_values(
                "Probabilidade",
                ascending=False
            )

            st.bar_chart(
                df_prob.set_index("Categoria")
            )

            st.caption(
                "As probabilidades são estimativas "
                "do modelo e não garantem que "
                "a classificação esteja correta."
            )

    st.info(
        "Este sistema é um protótipo acadêmico. "
        "Não substitui uma análise profissional. "
        "Evite inserir dados pessoais."
    )

# ABA 2 - RESULTADOS
with aba2:

    st.header("Desempenho da inteligência artificial")

        st.write(
        "Comparação entre as versões V1, V2 e V3 "
        "utilizando 100 reclamações de teste independente."
    )

    comparacao = pd.read_csv(
        BASE / "comparacao_final.csv"
    )

    st.dataframe(
        comparacao,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Comparação dos modelos")

    st.image(
        str(BASE / "grafico_comparacao_final.png")
    )

    st.subheader("Matriz de confusão - Modelo V3")

    st.image(
        str(BASE / "matriz_final_publico.png")
    )

    st.subheader("Principais resultados")

    c1, c2, c3 = st.columns(3)

    c1.metric("Acertos", "80")
    c2.metric("Erros", "20")
    c3.metric("Redução de erros", "39,39%")

    st.warning(
        "O teste foi realizado sobre uma amostra "
        "da base previamente filtrada. "
        "Algumas categorias possuem poucos exemplos."
    )

# ABA 3 - SOBRE O PROJETO
with aba3:

    st.header("Sobre o projeto")

    st.subheader("Objetivo")

    st.write(
        "Utilizar técnicas de Processamento de "
        "Linguagem Natural e aprendizado de máquina "
        "para classificar reclamações relacionadas "
        "a serviços de trânsito."
    )

    st.subheader("Base de dados")

    st.markdown(
        "[ConsumerBR - Dataset público]"
        "(https://zenodo.org/records/18022568)"
    )

    dados = pd.DataFrame({
        "Etapa": [
            "Base original",
            "Registros filtrados",
            "Registros após limpeza",
            "Registros rotulados",
            "Treinamento V2",
            "Validação experimental",
            "Teste independente"
        ],
        "Quantidade": [
            3150552,
            11085,
            11001,
            406,
            267,
            39,
            100
        ]
    })

    st.dataframe(
        dados,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Metodologia")

    st.markdown("""
    1. Coleta da base pública ConsumerBR.
    2. Filtragem por palavras-chave.
    3. Limpeza e remoção de duplicidades.
    4. Classificação manual das reclamações.
    5. Vetorização dos textos com TF-IDF.
    6. Treinamento com Regressão Logística.
    7. Ampliação da base por aprendizado ativo.
    8. Treinamento da versão V2.
    9. Avaliação em uma amostra independente.
    10. Desenvolvimento da interface Streamlit.
    """)

    st.subheader("Tecnologias utilizadas")

    st.write(
        "Python, Pandas, Scikit-learn, "
        "NumPy, Matplotlib, Seaborn, "
        "Joblib e Streamlit."
    )

# Rodapé
st.divider()

st.caption(
    "Trânsito IA | Projeto de Inteligência Artificial "
    "| UNOESC | 2026"
)
