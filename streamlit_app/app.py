import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise de Vendas de Jogos",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurações de paths
BASE_DIR = Path(__file__).parent.parent.resolve()  # sobe 1 pasta: de streamlit_app para raiz do projeto

DB_PATH = BASE_DIR / "data" / "processed" / "games.db"
MODEL_PATH = BASE_DIR / "ml"
# Cache para carregar dados
@st.cache_data
def load_data():
    """Carrega dados do banco SQLite"""
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        query = """
        SELECT 
            title, rating, platform, genres, developer, publisher, year,
            global_sales, na_sales, pal_sales, jp_sales, other_sales,
            metascore, slug
        FROM games_dim
        WHERE year IS NOT NULL
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Cache para carregar modelos
@st.cache_resource
def load_models():
    """Carrega todos os modelos treinados"""
    models = {}
    model_files = {
        "Global Sales": "model_sales_global.pkl",
        "NA Sales": "model_sales_na.pkl",
        "PAL Sales": "model_sales_pal.pkl",
        "JP Sales": "model_sales_jp.pkl",
        "Other Sales": "model_sales_other.pkl",
        "Metascore": "model_metascore.pkl"
    }
    
    for name, filename in model_files.items():
        try:
            models[name] = joblib.load(MODEL_PATH / filename)
        except Exception as e:
            st.warning(f"Modelo {name} não encontrado: {e}")
    
    return models

# Função para fazer previsões
def make_predictions(model, features_df):
    """Faz previsões usando o modelo"""
    try:
        prediction = model.predict(features_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Erro na previsão: {e}")
        return None

# Interface principal
def main():
    st.title("📊Análise de Vendas de Jogos")
    st.markdown("---")
    
    # Carregar dados e modelos
    df = load_data()
    models = load_models()
    
    if df is None:
        st.error("Não foi possível carregar os dados. Verifique se o banco de dados existe.")
        return
    
    if not models:
        st.error("Nenhum modelo foi carregado. Verifique se os modelos foram treinados.")
        return
    
    # Sidebar para navegação
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Escolha uma opção:", [
        "Dashboard",
        "Previsões",
        "Análise Exploratória",
        "Insights"
    ])
    
    if page == "Dashboard":
        dashboard_page(df)
    elif page == "Previsões":
        predictions_page(df, models)
    elif page == "Análise Exploratória":
        analysis_page(df)
    elif page == "Insights":
        insights_page(df)

def dashboard_page(df):
    st.header("Dashboard Geral")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Jogos", f"{len(df):,}")
    
    with col2:
        avg_global_sales = df['global_sales'].mean()
        st.metric("Vendas Globais Médias", f"{avg_global_sales:.2f}M")
    
    with col3:
        avg_metascore = df['metascore'].mean()
        st.metric("Metascore Médio", f"{avg_metascore:.1f}")
    
    with col4:
        latest_year = df['year'].max()
        st.metric("Ano Mais Recente", f"{int(latest_year)}")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vendas por Plataforma")
        platform_sales = df.groupby('platform')['global_sales'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=platform_sales.index,
            y=platform_sales.values,
            title="Top 10 Plataformas por Vendas",
            labels={'x': 'Plataforma', 'y': 'Vendas Globais (M)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Vendas por Ano")
        yearly_sales = df.groupby('year')['global_sales'].sum()
        fig = px.line(
            x=yearly_sales.index,
            y=yearly_sales.values,
            title="Evolução das Vendas Anuais",
            labels={'x': 'Ano', 'y': 'Vendas Globais (M)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabela dos top jogos
    st.subheader("Top 10 Jogos por Vendas Globais")
    top_games = df.nlargest(10, 'global_sales')[['title', 'platform', 'publisher', 'year', 'global_sales', 'metascore']]
    st.dataframe(top_games, use_container_width=True)

def predictions_page(df, models):
    st.header("Previsões de Vendas")
    
    st.markdown("Configure as características do jogo para obter previsões:")
    
    # Formulário para entrada de dados
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            rating = st.selectbox("Rating ESRB", sorted(df['rating'].dropna().unique()))
            platform = st.selectbox("Plataforma", sorted(df['platform'].dropna().unique()))
            genres = st.selectbox("Gênero", sorted(df['genres'].dropna().unique()))
        
        with col2:
            developer = st.selectbox("Desenvolvedor", sorted(df['developer'].dropna().unique()))
            publisher = st.selectbox("Publisher", sorted(df['publisher'].dropna().unique()))
            year = st.slider("Ano", int(df['year'].min()), int(df['year'].max()), int(df['year'].median()))
        
        submitted = st.form_submit_button("Fazer Previsões")
    
    if submitted:
        # Preparar dados para previsão
        features_df = pd.DataFrame({
            'rating': [rating],
            'platform': [platform],
            'genres': [genres],
            'developer': [developer],
            'publisher': [publisher],
            'year': [year]
        })
        
        st.subheader("Resultados das Previsões")
        
        # Fazer previsões para todos os modelos
        predictions = {}
        for model_name, model in models.items():
            prediction = make_predictions(model, features_df)
            if prediction is not None:
                predictions[model_name] = prediction
        
        # Exibir resultados
        if predictions:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Previsões de Vendas (Milhões)")
                for name, value in predictions.items():
                    if "Sales" in name:
                        st.metric(name, f"{value:.2f}M")
            
            with col2:
                st.subheader("⭐ Previsão de Qualidade")
                if "Metascore" in predictions:
                    metascore_pred = predictions["Metascore"]
                    st.metric("Metascore", f"{metascore_pred:.1f}")
                    
                    # Interpretação do Metascore
                    if metascore_pred >= 90:
                        st.success("Excelente! Jogo de alta qualidade.")
                    elif metascore_pred >= 75:
                        st.info("Bom jogo com boa recepção.")
                    elif metascore_pred >= 60:
                        st.warning("Jogo mediano.")
                    else:
                        st.error("Jogo com baixa qualidade.")
            
            # Gráfico de vendas regionais
            if any("Sales" in name for name in predictions.keys()):
                st.subheader("Distribuição Regional de Vendas")
                
                sales_data = {
                    'Região': ['América do Norte', 'Europa/Austrália', 'Japão', 'Outros'],
                    'Vendas': [
                        predictions.get('NA Sales', 0),
                        predictions.get('PAL Sales', 0),
                        predictions.get('JP Sales', 0),
                        predictions.get('Other Sales', 0)
                    ]
                }
                
                fig = px.pie(
                    values=sales_data['Vendas'],
                    names=sales_data['Região'],
                    title="Distribuição de Vendas por Região"
                )
                st.plotly_chart(fig, use_container_width=True)

def analysis_page(df):
    st.header("Análise Exploratória")
    
    # Filtros
    st.subheader("🔍 Filtros")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_platforms = st.multiselect("Plataformas", df['platform'].unique(), default=df['platform'].unique()[:5])
    
    with col2:
        year_range = st.slider("Período", int(df['year'].min()), int(df['year'].max()), 
                              (int(df['year'].min()), int(df['year'].max())))
    
    with col3:
        selected_genres = st.multiselect("Gêneros", df['genres'].unique(), default=df['genres'].unique()[:5])
    
    # Filtrar dados
    filtered_df = df[
        (df['platform'].isin(selected_platforms)) &
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1]) &
        (df['genres'].isin(selected_genres))
    ]
    
    st.markdown("---")
    
    # Análises
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlação: Metascore vs Vendas")
        fig = px.scatter(
            filtered_df.dropna(subset=['metascore', 'global_sales']),
            x='metascore',
            y='global_sales',
            color='platform',
            title="Relação entre Crítica e Vendas",
            labels={'metascore': 'Metascore', 'global_sales': 'Vendas Globais (M)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Vendas por Gênero")
        genre_sales = filtered_df.groupby('genres')['global_sales'].sum().sort_values(ascending=True)
        fig = px.bar(
            x=genre_sales.values,
            y=genre_sales.index,
            orientation='h',
            title="Total de Vendas por Gênero",
            labels={'x': 'Vendas Globais (M)', 'y': 'Gênero'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap de vendas regionais
    st.subheader("Heatmap de Vendas Regionais")
    regional_cols = ['na_sales', 'pal_sales', 'jp_sales', 'other_sales']
    region_data = filtered_df[regional_cols].corr()
    
    fig = px.imshow(
        region_data,
        text_auto=True,
        title="Correlação entre Vendas Regionais",
        color_continuous_scale='RdYlBu_r'
    )
    st.plotly_chart(fig, use_container_width=True)

def insights_page(df):
    st.header("Insights dos Dados")
    
    # Insights automáticos
    st.subheader("Principais Descobertas")
    
    # Top publisher
    top_publisher = df.groupby('publisher')['global_sales'].sum().sort_values(ascending=False).iloc[0]
    top_publisher_name = df.groupby('publisher')['global_sales'].sum().sort_values(ascending=False).index[0]
    
    st.success(f"**Top Publisher**: {top_publisher_name} com {top_publisher:.1f}M em vendas totais")
    
    # Melhor plataforma
    best_platform = df.groupby('platform')['global_sales'].mean().sort_values(ascending=False).iloc[0]
    best_platform_name = df.groupby('platform')['global_sales'].mean().sort_values(ascending=False).index[0]
    
    st.info(f"**Melhor Plataforma** (vendas médias): {best_platform_name} com {best_platform:.2f}M por jogo")
    
    # Gênero mais lucrativo
    top_genre = df.groupby('genres')['global_sales'].sum().sort_values(ascending=False).iloc[0]
    top_genre_name = df.groupby('genres')['global_sales'].sum().sort_values(ascending=False).index[0]
    
    st.warning(f" **Gênero Mais Lucrativo**: {top_genre_name} com {top_genre:.1f}M em vendas totais")
    
    # Análise temporal
    st.subheader("Análise Temporal")
    
    # Década mais produtiva
    df['decade'] = (df['year'] // 10) * 10
    decade_stats = df.groupby('decade').agg({
        'title': 'count',
        'global_sales': 'sum'
    }).rename(columns={'title': 'games_count'})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Jogos Lançados por Década")
        fig = px.bar(
            x=decade_stats.index,
            y=decade_stats['games_count'],
            title="Número de Jogos por Década",
            labels={'x': 'Década', 'y': 'Número de Jogos'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Vendas por Década")
        fig = px.bar(
            x=decade_stats.index,
            y=decade_stats['global_sales'],
            title="Vendas Totais por Década",
            labels={'x': 'Década', 'y': 'Vendas Globais (M)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribuição regional
    st.subheader("Distribuição Regional de Vendas")
    
    regional_totals = {
        'América do Norte': df['na_sales'].sum(),
        'Europa/Austrália': df['pal_sales'].sum(),
        'Japão': df['jp_sales'].sum(),
        'Outros': df['other_sales'].sum()
    }
    
    fig = px.pie(
        values=list(regional_totals.values()),
        names=list(regional_totals.keys()),
        title="Distribuição Global de Vendas por Região"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas avançadas
    st.subheader("Estatísticas Avançadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vendas Globais**")
        st.write(f"Média: {df['global_sales'].mean():.2f}M")
        st.write(f"Mediana: {df['global_sales'].median():.2f}M")
        st.write(f"Desvio Padrão: {df['global_sales'].std():.2f}M")
        st.write(f"Máximo: {df['global_sales'].max():.2f}M")
    
    with col2:
        st.markdown("**Metascore**")
        metascore_clean = df['metascore'].dropna()
        st.write(f"Média: {metascore_clean.mean():.1f}")
        st.write(f"Mediana: {metascore_clean.median():.1f}")
        st.write(f"Desvio Padrão: {metascore_clean.std():.1f}")
        st.write(f"Máximo: {metascore_clean.max():.1f}")

if __name__ == "__main__":
    main()