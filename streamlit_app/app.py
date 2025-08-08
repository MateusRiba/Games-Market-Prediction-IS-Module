import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, warnings
from pathlib import Path
from sqlalchemy import create_engine
warnings.filterwarnings("ignore")


#   Configurações gerais
st.set_page_config(
    page_title="Análise de Vendas de Jogos",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR   = Path(__file__).parent.parent.resolve()
DB_PATH    = BASE_DIR / "data" / "processed" / "games.db"
MODEL_PATH = BASE_DIR / "ml"

# utilidades de carregamento

@st.cache_data
def load_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query  = """
        SELECT title, rating, platform, genres, developer, publisher, year,
               global_sales, na_sales, pal_sales, jp_sales, other_sales,
               metascore, slug
        FROM games_dim WHERE year IS NOT NULL;
    """
    return pd.read_sql(query, engine)

@st.cache_resource
def load_models():
    files = {
        "Global Sales": "model_sales_global.pkl",
        "NA Sales"   : "model_sales_na.pkl",
        "PAL Sales"  : "model_sales_pal.pkl",
        "JP Sales"   : "model_sales_jp.pkl",
        "Other Sales": "model_sales_other.pkl",
        "Metascore"  : "model_metascore.pkl",
    }
    models = {}
    for name, fname in files.items():
        try:
            models[name] = joblib.load(MODEL_PATH / fname)
        except Exception as e:
            st.warning(f"Modelo {name} indisponível: {e}")
    return models

def predict(model, df_row):
    try:
        return model.predict(df_row)[0]
    except Exception as e:
        st.error(f"Erro na previsão: {e}")
        return np.nan


# funções auxiliares de cenário

ELASTICITY = 1.2            # |ε| preço–demanda
BASE_PRICE = 60

def price_factor(price):
    ratio = price / BASE_PRICE
    f = ratio ** (-ELASTICITY)
    return np.clip(f, 0.5, 1.6)   # segurança

marketing_factor_map = {
    "Baixo" : 1.05,   # +5 %
    "Médio" : 1.25,   # +25 %
    "Alto"  : 1.50,   # +50 %
}

def score_to_success(ms):
    """Converte metascore → chance de sucesso (%) e risco (%)"""
    success = np.clip((ms - 50) * 2, 0, 95)  # 50 →0 %, 95→90-95 %
    risk    = 100 - success
    return success, risk

#   Streamlit 
def main():
    st.title("📊 Análise de Vendas de Jogos")
    st.markdown("---")

    data   = load_data()
    models = load_models()
    if data is None or not len(models):
        st.stop()

    page = st.sidebar.radio("Navegação", ["Dashboard", "Previsões",
                                          "Análise Exploratória", "Insights"])
    if page == "Dashboard":
        dashboard_page(data)
    elif page == "Previsões":
        predictions_page(data, models)
    elif page == "Análise Exploratória":
        analysis_page(data)
    else:
        insights_page(data)

# ---------- DASHBOARD --------------------------------------------------------
def dashboard_page(df):
    st.header("Dashboard Geral")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Jogos", f"{len(df):,}")
    c2.metric("Vendas Globais Médias", f"{df['global_sales'].mean():.2f} M")
    c3.metric("Metascore Médio", f"{df['metascore'].mean():.1f}")
    c4.metric("Ano Mais Recente", int(df['year'].max()))
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        top_platform = df.groupby("platform")["global_sales"].sum().nlargest(10)
        fig = px.bar(top_platform, title="Top 10 Plataformas por Vendas",
                     labels={"value":"Vendas Globais (M)", "platform":"Plataforma"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        yearly = df.groupby("year")["global_sales"].sum()
        fig = px.line(yearly, title="Evolução das Vendas Anuais",
                      labels={"value":"Vendas Globais (M)", "year":"Ano"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Jogos por Vendas Globais")
    st.dataframe(df.nlargest(10, "global_sales")[["title","platform","publisher",
                                                  "year","global_sales","metascore"]],
                 use_container_width=True)

# ---------- PREDICTIONS ------------------------------------------------------
def predictions_page(df, models):
    st.header("Previsões de Vendas")
    st.markdown("Configure as características do jogo e **simule cenários**:")

    # 🎯  Parâmetros de cenário (fora do form, pois afetam só pós-previsão)
    price            = st.slider("Preço de Venda (USD)", 1, 100, 60)
    marketing_level  = st.selectbox("Nível de Investimento em Marketing",
                                    ["Baixo", "Médio", "Alto"])

    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            rating   = st.selectbox("Rating ESRB", sorted(df['rating'].dropna().unique()))
            platform = st.selectbox("Plataforma", sorted(df['platform'].dropna().unique()))
            # até 2 gêneros
            genres   = st.multiselect("Gêneros (máx. 2)", sorted(df['genres'].dropna().unique()),
                                      max_selections=2)
            if not genres:
                st.warning("Escolha pelo menos 1 gênero.")
        with c2:
            developer = st.selectbox("Desenvolvedor", sorted(df['developer'].dropna().unique()))
            publisher = st.selectbox("Publisher", sorted(df['publisher'].dropna().unique()))
            year      = st.slider("Ano", int(df['year'].min()), int(df['year'].max()),
                                  int(df['year'].median()))
        submitted = st.form_submit_button("Fazer Previsões")

    if not submitted:
        return

    # ----- prepara linha de features
    feat = pd.DataFrame([{
        "rating": rating,
        "platform": platform,
        "genres": "|".join(genres),           # concatenação para o modelo
        "developer": developer,
        "publisher": publisher,
        "year": year
    }])

    # ----- obtém previsões cruas
    preds = {name: predict(m, feat) for name, m in models.items()}
    original_sales = preds.get("Global Sales", 0)

    # ----- aplica fatores de cenário
    pf = price_factor(price)
    mf = marketing_factor_map[marketing_level]
    for k in preds:
        if "Sales" in k:
            preds[k] *= pf * mf

    metascore_pred = preds["Metascore"]
    success_perc, risk_perc = score_to_success(metascore_pred)

    # ========== exibição ==========
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Vendas Previstas (Milhões)")
        for k, v in preds.items():
            if "Sales" in k:
                st.metric(k, f"{v:.2f} M")

        # comparação gráfico
        fig = go.Figure([
            go.Bar(name="Original", x=["Global"], y=[original_sales]),
            go.Bar(name="C/ Cenário", x=["Global"], y=[preds["Global Sales"]])
        ])
        fig.update_layout(barmode="group", title="Impacto de Preço + Marketing")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Qualidade & Risco")
        st.metric("Metascore Previsto", f"{metascore_pred:.1f}")

        st.text("Chance de Sucesso Comercial")
        st.progress(int(success_perc))
        st.text(f"{success_perc:.1f}%")

        st.text("Risco de Falha")
        st.progress(int(risk_perc))
        st.text(f"{risk_perc:.1f}%")

    # gráfico pizza regional
    st.subheader("Distribuição Regional de Vendas")
    pie = px.pie(
        names=["NA", "PAL", "JP", "Other"],
        values=[preds.get('NA Sales',0), preds.get('PAL Sales',0),
                preds.get('JP Sales',0), preds.get('Other Sales',0)],
        title="Vendas por Região"
    )
    st.plotly_chart(pie, use_container_width=True)

# ---------- ANALYSIS ---------------------------------------------------------
def analysis_page(df):
    st.header("Análise Exploratória")
    c1,c2,c3 = st.columns(3)
    with c1:
        plats = st.multiselect("Plataformas", df["platform"].unique(), df["platform"].unique()[:5])
    with c2:
        y0,y1 = st.slider("Período", int(df.year.min()), int(df.year.max()),
                          (int(df.year.min()), int(df.year.max())))
    with c3:
        gens = st.multiselect("Gêneros", df["genres"].unique(), df["genres"].unique()[:5])

    filt = (df["platform"].isin(plats)) & (df["year"].between(y0,y1)) & (df["genres"].isin(gens))
    dfx = df[filt]

    sc = px.scatter(
        dfx.dropna(subset=["metascore","global_sales"]),
        x="metascore", y="global_sales", color="platform",
        labels={"metascore":"Metascore","global_sales":"Vendas Globais (M)"},
        title="Metascore × Vendas"
    )
    st.plotly_chart(sc, use_container_width=True)

    gs = dfx.groupby("genres")["global_sales"].sum().sort_values()
    bar = px.bar(gs, orientation='h', title="Vendas por Gênero",
                 labels={"value":"Vendas Globais (M)", "genres":"Gênero"})
    st.plotly_chart(bar, use_container_width=True)

# ---------- INSIGHTS ---------------------------------------------------------
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


# --------------------------------------------------------
if __name__ == "__main__":
    main()
