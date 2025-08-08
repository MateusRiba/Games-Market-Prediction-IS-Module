#streamlit_app/app.py

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

def reconcile_global(preds: dict) -> dict:
    """Se Global < soma regionais, substitui Global pela soma (consistência)."""
    def _safe(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    reg_sum = (
        _safe(preds.get("NA Sales")) +
        _safe(preds.get("PAL Sales")) +
        _safe(preds.get("JP Sales")) +
        _safe(preds.get("Other Sales"))
    )
    if "Global Sales" in preds and _safe(preds.get("Global Sales")) < reg_sum:
        preds["Global Sales"] = reg_sum
    return preds


def sales_percentile(sales: float, df: pd.DataFrame) -> float:
    """Percentil (0-100) da venda prevista comparada ao histórico do dataset."""
    arr = df["global_sales"].dropna().to_numpy()
    if arr.size == 0:
        return 50.0
    sales = max(0.0, float(sales))
    pct = (arr <= sales).sum() / arr.size * 100.0
    return float(np.clip(pct, 1, 99))


def success_from_ms_and_sales(ms: float, sales: float, df: pd.DataFrame, w_ms: float = 0.65):
    """
    Combina Metascore e percentil de vendas para estimar chance de sucesso.
    - Metascore -> curva logística (centrada ~70) com piso 25% e teto 99%.
    - Vendas -> percentil histórico (0-100).
    - Peso padrão: 65% metascore, 35% vendas (ajuste em w_ms).
    """
    # componente metascore (S-curve)
    raw = 1.0 / (1.0 + np.exp(-0.3 * (ms - 70)))
    succ_ms = 100.0 * (0.25 + 0.75 * raw)          # ~60 → ~35-40%, 75 → ~86%, 90 → ~96%
    succ_ms = float(np.clip(succ_ms, 5, 99))

    # componente vendas (percentil)
    succ_sales = sales_percentile(sales, df)        # 0..100

    # blend ponderado
    w_sales = 1.0 - w_ms
    success = w_ms * succ_ms + w_sales * succ_sales
    success = float(np.clip(success, 1, 99))
    risk = 100.0 - success
    return success, risk, succ_ms, succ_sales


def _weighted_choice(series: pd.Series):
    """Escolhe 1 valor ponderado pela frequência observada no dataset."""
    vc = series.dropna().value_counts(normalize=True)
    return np.random.choice(vc.index, p=vc.values)

def sample_random_game(df: pd.DataFrame):
    """Gera um 'jogo' plausível com base nas distribuições do dataset."""
    rating    = _weighted_choice(df["rating"])
    platform  = _weighted_choice(df["platform"])
    dev       = _weighted_choice(df["developer"])
    pub       = _weighted_choice(df["publisher"])
    year      = int(_weighted_choice(df["year"]))

    # gêneros: 1 ou 2, ponderado por frequência
    vc_gen = df["genres"].dropna().value_counts(normalize=True)
    k = np.random.choice([1, 2], p=[0.6, 0.4])
    genres = list(np.random.choice(vc_gen.index, size=k, replace=False, p=vc_gen.values))

    # preço ~ triangular (pico em 60), marketing nível ponderado
    price = int(round(np.random.triangular(1, 60, 100)))
    marketing_level = np.random.choice(["Baixo", "Médio", "Alto"], p=[0.4, 0.4, 0.2])

    return {
        "rating": rating,
        "platform": platform,
        "developer": dev,
        "publisher": pub,
        "year": year,
        "genres": genres,
        "price": price,
        "marketing_level": marketing_level,
    }


def price_factor(price):
    ratio = price / BASE_PRICE
    f = ratio ** (-ELASTICITY)
    return np.clip(f, 0.5, 1.6)   # segurança

marketing_factor_map = {
    "Baixo" : 1.05,   # +5 %
    "Médio" : 1.25,   # +25 %
    "Alto"  : 1.50,   # +50 %
}

def score_to_success(ms, center=70, steepness=0.18, min_s=35, max_s=98):
    ms = float(np.clip(ms, 0, 100))
    sigma = 1.0 / (1.0 + np.exp(-steepness * (ms - center)))
    success = min_s + (max_s - min_s) * sigma
    success = float(np.clip(success, 0, 100))
    risk = float(100 - success)
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

    # Parâmetros do cenário (pós-modelo)
    price           = st.slider("Preço de Venda (USD)", 1, 100, 60)
    marketing_level = st.selectbox("Nível de Investimento em Marketing", ["Baixo", "Médio", "Alto"])

    # 🔀 Randomizar
    rand_col1, rand_col2 = st.columns([1, 3])
    with rand_col1:
        do_random = st.button("🔀 Randomizar e Prever")

    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            rating   = st.selectbox("Rating ESRB", sorted(df['rating'].dropna().unique()))
            platform = st.selectbox("Plataforma", sorted(df['platform'].dropna().unique()))
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

    def run_prediction(_rating, _platform, _genres, _developer, _publisher, _year, _price, _marketing_level):
        # ----- features
        feat = pd.DataFrame([{
            "rating": _rating,
            "platform": _platform,
            "genres": "|".join(_genres),
            "developer": _developer,
            "publisher": _publisher,
            "year": _year
        }])

        # ----- previsões “cruas”
        preds = {name: predict(m, feat) for name, m in models.items()}
        preds = reconcile_global(preds)
        original_sales = preds.get("Global Sales", 0.0)

        # ----- aplica fatores de preço + marketing
        pf = price_factor(_price)
        mf = marketing_factor_map[_marketing_level]
        for k in preds:
            if "Sales" in k:
                preds[k] *= pf * mf

        preds = reconcile_global(preds)

        metascore_pred = preds.get("Metascore", np.nan)
        global_pred    = preds.get("Global Sales", 0.0)

        success_perc, risk_perc, succ_ms_comp, succ_sales_comp = success_from_ms_and_sales(
            metascore_pred if not np.isnan(metascore_pred) else 50.0,
            global_pred,
            df,
            w_ms=0.5  # ajuste é possivel para o peso do metascore 
)

        # ---------- Exibição ----------
        st.subheader("Parâmetros usados")
        st.write({
            "rating": _rating, "platform": _platform, "genres": _genres,
            "developer": _developer, "publisher": _publisher, "year": _year,
            "preço(USD)": _price, "marketing": _marketing_level
        })

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Vendas Previstas (Milhões)")
            for k, v in preds.items():
                if "Sales" in k:
                    st.metric(k, f"{v:.2f} M")

            fig = go.Figure([
                go.Bar(name="Original", x=["Global"], y=[original_sales]),
                go.Bar(name="C/ Cenário", x=["Global"], y=[preds.get("Global Sales", 0)])
            ])
            fig.update_layout(barmode="group", title="Impacto de Preço + Marketing")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Qualidade & Risco")
            if not np.isnan(metascore_pred):
                st.metric("Metascore Previsto", f"{metascore_pred:.1f}")
            else:
                st.metric("Metascore Previsto", "—")

            st.text("Chance de Sucesso Comercial")
            st.progress(int(success_perc))
            st.text(f"{success_perc:.1f}%")

            st.text("Risco de Falha")
            st.progress(int(risk_perc))
            st.text(f"{risk_perc:.1f}%")

            st.caption(f"Contribuições — Qualidade: {succ_ms_comp:.1f}% · Vendas (percentil): {succ_sales_comp:.1f}%")

        st.subheader("Distribuição Regional de Vendas")
        pie = px.pie(
            names=["NA", "PAL", "JP", "Other"],
            values=[preds.get('NA Sales',0), preds.get('PAL Sales',0),
                    preds.get('JP Sales',0), preds.get('Other Sales',0)],
            title="Vendas por Região"
        )
        st.plotly_chart(pie, use_container_width=True)

    # 1) Caminho do botão “Randomizar e Prever”
    if do_random:
        params = sample_random_game(df)
        run_prediction(
            params["rating"], params["platform"], params["genres"],
            params["developer"], params["publisher"], params["year"],
            params["price"], params["marketing_level"]
        )
        return

    # 2) Caminho do submit normal
    if submitted and genres:
        run_prediction(rating, platform, genres, developer, publisher, year, price, marketing_level)


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
