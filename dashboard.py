import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Configuração inicial
st.set_page_config(page_title="Análise de Vendas", layout="wide")

# Carregar dados e modelos
@st.cache_data
def load_data():
    vendas_futuras = pd.read_csv('vendas_futuras.csv')
    le = joblib.load('label_encoder.pkl')
    return vendas_futuras, le

vendas_futuras, le = load_data()

# Sidebar
st.sidebar.header("Navegação")
view_option = st.sidebar.radio(
    "Selecione a Visualização",
    options=["Dashboard Analítico", "Sobre o Projeto"],
    index=0
)

if view_option == "Sobre o Projeto":
    st.markdown("""
    # 📊 Análise de Vendas - Dashboard Streamlit

    ## 📌 Sobre o Projeto
    Este projeto tem como objetivo analisar e prever vendas com base em dados históricos. Utilizamos **Python, Pandas, Scikit-learn e Streamlit** para criar um dashboard interativo que permite visualizar tendências de vendas, identificar categorias mais lucrativas e prever receitas futuras.

    ## 🔍 Storytelling dos Dados

    1. **Coleta e Processamento dos Dados**  
       - Os dados foram extraídos de um arquivo CSV contendo informações de vendas por categoria, mês e tipo de atendimento.
       - Foram aplicadas técnicas de **limpeza e transformação** para garantir consistência.
       - Para variáveis categóricas, utilizamos **label encoding** e, posteriormente, optamos por **one-hot encoding** para otimizar a precisão do modelo.

    2. **Modelo Preditivo**  
       - Implementamos um **Random Forest Regressor**, ajustado com **RandomizedSearchCV** para otimização de hiperparâmetros.
       - O modelo foi treinado com os dados filtrados e validados, atingindo um bom **R² Score**, indicando previsões confiáveis.
       
    3. **Construção do Dashboard**  
       - O **Streamlit** foi utilizado para criar um painel interativo com visualizações em **Plotly**.
       - Foram adicionadas métricas gerais e filtros para permitir uma análise específica por categoria.
       - Gráficos destacados:
         - **Top Categorias por Vendas** (Gráfico de barras)
         - **Previsão Mensal** (Gráfico de linha por categoria)
         - **Heatmap de Vendas** (Distribuição por mês e categoria)

    ## 📈 Conclusão
    Com base na análise dos dados e previsões:
    - **As 5 principais categorias** concentram a maior parte das vendas, sugerindo que promoções ou estratégias devem ser focadas nelas.
    - **A sazonalidade das vendas** é evidente: algumas categorias têm picos específicos em certos meses, o que pode ser aproveitado para campanhas estratégicas.
    - **A previsão geral de vendas** indica um crescimento sustentável, reforçando que as tendências históricas são bons indicadores do futuro.

    Este dashboard pode ser utilizado para **tomada de decisão estratégica**, ajudando empresas a **maximizar seus lucros e otimizar seus investimentos**. 🚀

    ---

    ## 🛠 Tecnologias Utilizadas
    - Python (Pandas, Scikit-learn)
    - Streamlit
    - Plotly
    - Random Forest Regressor

    ## 📥 Como Executar o Projeto
    1. Clone o repositório:
       ```sh
       git clone https://github.com/seu-usuario/seu-repositorio.git
       ```
    2. Instale as dependências:
       ```sh
       pip install -r requirements.txt
       ```
    3. Execute o dashboard:
       ```sh
       streamlit run dash.py
       ```
    """)

elif view_option == "Dashboard Analítico":
    st.sidebar.header("Filtros")
    selected_category = st.sidebar.selectbox(
        "Selecione a Categoria",
        options=vendas_futuras['Category'].unique(),
        format_func=lambda x: le.inverse_transform([x])[0]
    )

    # Métricas principais
    st.header("📊 Painel de Análise de Vendas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Categorias", vendas_futuras['Category'].nunique())
    with col2:
        st.metric("Previsão Total de Vendas", f"R$ {vendas_futuras['predicted_sales'].sum():,.2f}")
    with col3:
        avg_sales = vendas_futuras['predicted_sales'].mean()
        st.metric("Média por Categoria/Mês", f"R$ {avg_sales:,.2f}")

    # Visualização 1: Top categorias
    st.subheader("Top 5 Categorias por Vendas")
    top_categories = vendas_futuras.groupby('Category')['predicted_sales'].sum().nlargest(5)
    top_categories.index = top_categories.index.map(lambda x: le.inverse_transform([x])[0])
    fig1 = px.bar(top_categories, orientation='h')
    st.plotly_chart(fig1, use_container_width=True)

    # Visualização 2: Análise temporal
    st.subheader("Previsão Mensal por Categoria")
    filtered_data = vendas_futuras[vendas_futuras['Category'] == selected_category]
    fig2 = px.line(
        filtered_data,
        x='Month',
        y='predicted_sales',
        title=f"Vendas Previstas - {le.inverse_transform([selected_category])[0]}"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Visualização 3: Heatmap de Vendas
    st.subheader("Distribuição de Vendas por Categoria e Mês")
    heatmap_data = vendas_futuras.pivot_table(
        index='Category',
        columns='Month',
        values='predicted_sales',
        aggfunc='sum'
    )
    heatmap_data.index = heatmap_data.index.map(lambda x: le.inverse_transform([x])[0])
    fig3 = px.imshow(
        heatmap_data,
        labels=dict(x="Mês", y="Categoria", color="Vendas"),
        aspect="auto"
    )
    st.plotly_chart(fig3, use_container_width=True)