import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "metascore.parquet"
MODEL = Path(__file__).resolve().parents[2] / "models" / "metascore.pkl"

st.set_page_config(page_title="metascore Demo", layout="wide")

df = pd.read_parquet(DATA)

st.title("Distribuição de metascore")
st.plotly_chart(px.histogram(df, x="metascore", nbins=30), use_container_width=True)

with st.expander("Scatter Genre × metascore"):
    st.plotly_chart(px.strip(df, x="genres", y="metascore"), use_container_width=True)

st.header("Prever nota para um novo jogo")
genres = st.selectbox("Gênero", df["genres"].unique())
rating = st.selectbox("Rating", df["rating"].dropna().unique())
developer = st.selectbox("Developer", df["developer"].dropna().unique())

if st.button("Prever"):
    pipe = joblib.load(MODEL)
    sample = {"genres": [genres], "rating": [rating], "developer": [developer]}
    pred = pipe.predict(pd.DataFrame(sample))[0]
    st.success(f"metascore previsto: **{pred:.1f}**")
