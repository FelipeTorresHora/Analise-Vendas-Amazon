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