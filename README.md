# 🎮 Games Market Predictor

Previsão de **vendas globais e regionais** (NA, PAL, JP, Other) e **Metascore** para novos jogos a partir de atributos como plataforma, gêneros, publisher, etc.

Inclui:

* **ETL** → cria `data/processed/games.db` (SQLite).
* **Treino** → treina 6 modelos (Random Forest) e salva em `ml/*.pkl`.
* **API** (**FastAPI**) → endpoint de predição por alvo + endpoint de filtros.
* **App** (**Streamlit**) → UI para análise, previsões e simulação de cenários (preço e marketing).

---

## 📁 Estrutura

```
.
├── backend/
│   └── app/
│       ├── main.py               # FastAPI (predict, filters)
│       ├── models.py             # ORM (Game -> games_dim)
│       └── api/
│           ├── filters.py
│           └── debug.py
├── data/
│   ├── raw/                      # CSVs de origem
│   └── processed/
│       └── games.db              # (gerado pelo ETL)
├── ml/
│   ├── train_models.py           # treina e salva modelos
│   └── model_*.pkl               # (gerados localmente; ignorados no git)
└── streamlit_app/
    └── app.py                    # frontend Streamlit
```
---

## Começando

### 1) Pré-requisitos

* **Python 3.10 – 3.12** (recomendado 3.11/3.12).
* Git instalado.

> **Importante (scikit-learn & pickles):** modelos salvos com uma versão do `scikit-learn` podem dar erro se carregados com outra. Se ocorrer, **re-treine** os modelos (Seção *Treinar modelos*).

---

### 2) Clonar e criar ambiente

```bash
git clone <URL-DO-REPO>
cd Games-Market-Prediction-IS-Module

# criar venv (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

# ou no Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate
```

---

### 3) Instalar dependências

Crie (ou apenas use) o arquivo `requirements.txt` com o conteúdo abaixo e instale:

```bash
pip install -r requirements.txt
```

`requirements.txt` sugerido:

```txt
pandas>=2.2.2
numpy>=1.26.4
scikit-learn==1.5.2
joblib>=1.4.2

SQLAlchemy>=2.0.30
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0

streamlit>=1.33.0
plotly>=5.22.0

python-slugify>=8.0.4
rapidfuzz>=3.9.1
```

---

### 4) Preparar os dados (ETL) 

Coloque os CSVs originais em **`data/raw/`** com os nomes esperados (ajuste no script se necessário):

* `games (1).csv` (Metacritic)
* `games_sales.csv` (VGChartz)

Execute:

```bash
python ETL_build_dataset.py
```

Saída esperada:
`ETL concluído ... linhas gravadas em data/processed/games.db`

OBS: ambos os datasets já estão disponiveis, então esse passo é opcional.

---

### 5) Treinar modelos

```bash
python ml/train_models.py
```

Isso gera:

```
ml/model_sales_global.pkl
ml/model_sales_na.pkl
ml/model_sales_pal.pkl
ml/model_sales_jp.pkl
ml/model_sales_other.pkl
ml/model_metascore.pkl
```

Cada treino imprime o **MAE** de validação.

> Se aparecer erro de versão do `scikit-learn` ao carregar pickles, **re-treine** com a versão instalada (rodando o passo acima) ou alinhe a versão do `scikit-learn` ao criar a venv.

---

### 6) Rodar o **backend** (FastAPI)

```bash
uvicorn backend.app.main:app --reload --port 8000
```

* **Docs interativas:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Exemplos:

```bash
# Filtros (para popular selects do frontend)
curl http://127.0.0.1:8000/filters

# Predição de um alvo (ex.: vendas globais)
curl -X POST http://127.0.0.1:8000/predict/global_sales \
  -H "Content-Type: application/json" \
  -d '{"rating":"E","platform":["PS4"],"genres":["Action","Adventure"],
       "developer":"Ubisoft","publisher":"Ubisoft","year":2020}'
```

---

### 7) Rodar o **frontend** (Streamlit)

Em outro terminal (mesma venv):

```bash
python -m streamlit run streamlit_app/app.py --server.port 8501
```

* App local: [http://localhost:8501](http://localhost:8501)

No app você pode:

* Ver **Dashboard** com KPIs e gráficos.
* Fazer **Previsões** e **simular cenários**:

  * **Preço** (neutro = USD 60; fator baseado em **elasticidade preço–demanda**).
  * **Marketing** (níveis Baixo/Médio/Alto com impacto moderado).
* Ver **chance de sucesso** e **risco** a partir do **Metascore** previsto.
* Explorar dados históricos e **insights**.

---

## Como funciona (resumo)

1. **ETL**: junta Metacritic + VGChartz via *slug* + *fuzzy matching*, normaliza colunas e grava em SQLite (`games_dim`).
2. **Treino**: `RandomForestRegressor` em *pipeline* com `OneHotEncoder(handle_unknown='ignore')`.
   Um modelo por alvo: `global_sales`, `na_sales`, `pal_sales`, `jp_sales`, `other_sales`, `metascore`.
3. **API**:

   * `POST /predict/{target}` → recebe features (gêneros até 2, plataforma, etc.), normaliza listas (concatena com `|`) e retorna a previsão do alvo.
   * `GET /filters` → valores distintos de gênero, rating e plataforma (para o frontend).
4. **App**:

   * Carrega dados e modelos locais.
   * Aplica **fatores de cenário** após a previsão:

     * **Preço**: `factor = (price/60) ** (-ε)`, com `ε ≈ 1.2` (limites de segurança).
     * **Marketing**: `Baixo=+5%`, `Médio=+25%`, `Alto=+50%` (efeito moderado com base em estudos de MMM/indústria).
   * **Chance de sucesso** e **risco**: função monotônica do Metascore previsto, exibida com barras de progresso.
   * Gráficos: comparação “original vs cenário” e distribuição regional.

---

## Dicas & Troubleshooting

* **Porta em uso**

  * Backend: `--port 8001`
  * Frontend: `--server.port 8502`

* **ModuleNotFoundError**

  * Garanta que a *venv* está ativa e rode `pip install -r requirements.txt`.

* **Erro ao carregar .pkl / `_RemainderColsList`**

  * Re-treine com `python ml/train_models.py`.

* **`games.db` não encontrado**

  * Rode o **ETL** (Seção 4).

* **Evitar commitar `.pkl`**

  * `.gitignore`:

    ```
    ml/*.pkl
    ml/*.joblib
    !ml/.gitkeep
    ```
  * Se já foram adicionados: `git rm --cached ml/*.pkl`

---

## 📚 Bibliotecas (o que cada uma faz)
Caso o requirements não funcione, é tambem viavel baixar as bibliotecas individualmente.

* **pandas** – manipulação de dados tabulares (ETL, consultas, groupby).
* **numpy** – operações numéricas/arrays.
* **scikit-learn** – ML (pipelines, OneHotEncoder, RandomForestRegressor).
* **joblib** – salvar/carregar modelos (`.pkl`).
* **SQLAlchemy** – acesso ao **SQLite** (`data/processed/games.db`).
* **FastAPI** – API REST (endpoints de predição e filtros).
* **uvicorn\[standard]** – servidor ASGI para rodar a API.
* **pydantic** – validação/serialização dos esquemas de entrada.
* **streamlit** – frontend web para dashboards, exploração e previsões.
* **plotly** – gráficos interativos no Streamlit.
* **python-slugify** – gera slugs (IDs amigáveis) para casar registros.
* **rapidfuzz** – *fuzzy matching* veloz e preciso para unir fontes.

---

## ✅ Roadmap curto (ideias)

* Cache de predições no backend.
* Validação de combinações plataforma×ano.
* Feature store para reuso de encoders entre treino/produção.
* Testes unitários para ETL e API (`pytest`) usando `tests/data/sample_games.csv`.

---

**Pronto!** Depois de seguir os passos (ETL → Treino → API → App), você terá previsões e simulações rodando localmente.
