# Blue Zone Explorer

> **Your postcode matters more than your paycheck.**

*A data-driven policy simulation tool that maps regional flourishing across 240 EU NUTS2 regions, identifies which conditions produce long and healthy lives, and lets anyone simulate what policy change would do to a region's prospects.*

---

## The idea

Some places forgot to die. Sardinia's Barbagia region, the Algarve coast, rural Greece — places where people live decades longer than economic theory would predict. These are Blue Zones: regions where the conditions for human flourishing happen to align. Not wealth. Not medicine. Community, purpose, movement, environment.

The problem is that no systematic tool exists to tell policymakers and citizens which regions are closest to these conditions, which are improving, and what levers actually move the needle.

Blue Zone Explorer is that tool.

---

## What it does

**The map** — Europe colour-coded by Blue Zone cluster. Each of 240 NUTS2 regions carries a score, a trend arrow (improving or declining over 18 years), and a full KPI breakdown. Hover any region for a snapshot. Click to open the scenario panel.

**The score** — a composite Blue Zone Score (0–100) computed via Principal Component Analysis on five regional indicators: poverty rate, unemployment, PM2.5 air quality, life expectancy, and interpersonal trust. The score captures the *conditions* for longevity, not just the outcome. Validated against life expectancy (r > 0.65) and life satisfaction (r > 0.55).

**The sliders** — four KPI domains, each with a slider. Move one and the Blue Zone Score updates live in the browser — no server call, no latency. The projection chart shows where the region is heading under your scenario versus baseline.

**The policy chat** — paste any piece of legislation or policy text. Claude reads it, extracts likely KPI impacts as structured JSON, applies them to the sliders, and reprices the score in seconds.

**The GDP lens** — GDP per capita sits deliberately outside the score. A separate scatter plot shows the non-linear (Preston curve) relationship between regional wealth and flourishing — steep gains at low income, a plateau around €28–32k, and slight decline above. Luxembourg does not score better than Sardinia. That's the point.

---

## Architecture

```
bluezone-explorer/
├── data/
│   ├── raw/
│   │   └── eurostat/                 # .tsv files from Eurostat API
│   ├── processed/
│   │   └── nuts2_wide/               # One parquet per indicator
│   └── artifacts/
│       ├── scaler.joblib
│       ├── pca_loadings.json
│       ├── kmeans_model.joblib
│       ├── scores_by_year.parquet
│       └── region_metadata.parquet
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_pca_scoring.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_validation.ipynb
│   └── 06_gdp_analysis.ipynb
│
├── scripts/
│   ├── 01_data_collection.py
│   ├── 02_preprocessing.py
│   ├── 03_pca_scoring.py
│   ├── 04_clustering.py
│   ├── 05_validation.py
│   └── 06_gdp_analysis.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── eurostat.py               # Eurostat API fetching functions
│   │   └── gcs.py                    # Upload/download to GCS bucket
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── scoring.py
│   │   └── clustering.py
│   └── utils/
│       ├── __init__.py
│       └── logging.py
│
├── app/
│   ├── main.py
│   ├── pages/
│   │   ├── 01_map.py
│   │   ├── 02_scenario.py
│   │   ├── 03_policy_chat.py
│   │   └── 04_methodology.py
│   └── components/
│       ├── map_component.py
│       ├── scorer.py
│       ├── projection_chart.py
│       └── gdp_scatter.py
│
├── secrets/
│   └── service-account.json
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_scoring.py
│   └── test_clustering.py
│
├── .env
├── .env.example
├── .gitignore
├── Makefile
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## The ML pipeline

### 1 — Data collection

Eight datasets pulled from [Eurostat](https://ec.europa.eu/eurostat) and the [European Environment Agency](https://www.eea.europa.eu), covering 2005–2022 across all 240 EU NUTS2 regions.

| Indicator | Source | Code | Resolution |
|---|---|---|---|
| Life expectancy at birth | Eurostat | `demo_r_mlifexp` | NUTS2 |
| Avoidable mortality rate | Eurostat | `hlth_cd_asmr2` | NUTS2 |
| At-risk-of-poverty rate | Eurostat | `tgs00103` | NUTS2 |
| Unemployment rate | Eurostat | `tgs00010` | NUTS2 |
| PM2.5 exposure | EEA | tabular CSV | NUTS2 |
| Interpersonal trust | Eurostat | `ilc_pw06` | Country → assigned to regions |
| Life satisfaction | Eurostat | `ilc_pw01` | Country → validation only |
| GDP per capita (PPS) | Eurostat | `nama_10r_2gdp` | NUTS2 → external lens |

> **On the social indicators:** trust and life satisfaction exist only at country level in Eurostat — the underlying EU-SILC surveys are not designed with large enough regional samples. Country-level trust scores are assigned to all NUTS2 regions within a country, which captures between-country social variation while regional variation is explained by the other four variables. This limitation is documented transparently in the methodology.

### 2 — Preprocessing

For each NUTS2 × KPI combination, two temporal features are computed:

- **Mean** — average value across 2010–2022, capturing the current level
- **Slope** — OLS trend per year, capturing direction of travel

Missing values are handled in two passes:
1. Linear interpolation for isolated gaps (≤ 2 consecutive missing years)
2. Country mean imputation for larger gaps

"Bad" indicators (high poverty, high unemployment, high mortality, high PM2.5) carry naturally negative loadings in PCA — the raw standardised values are fed in unchanged. There is no manual flipping of variables; PCA discovers the direction.

### 3 — Blue Zone Score via PCA

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Fit on cross-sectional means — stable reference, not panel-inflated
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mean)   # 240 regions × 10 features (5 KPIs × mean + slope)

pca = PCA(n_components=1)
pca.fit(X_scaled)

# PC1 sign is arbitrary — orient toward life expectancy
scores = pca.transform(X_scaled)[:, 0]
r, _ = pearsonr(scores, life_expectancy)
if r < 0:
    scores *= -1
    pca.components_[0] *= -1

# Rescale to 0–100 using global min/max across all years
# (ensures year-to-year scores are comparable)
blue_zone_scores = (scores - global_min) / (global_max - global_min) * 100
```

The loadings vector (`pca.components_[0]`) is exported as `pca_loadings.json` and shipped to the frontend. All in-browser rescoring is a dot product — no network call on slider movement.

### 4 — Temporal scoring

PCA is fitted once on the cross-sectional mean matrix. The fixed loadings and scaler are then applied to each year's snapshot independently:

```python
for year in range(2005, 2023):
    X_year = scaler.transform(snapshot[year])   # same scaler, same means/stds
    raw = pca.transform(X_year)[:, 0]
    scores_by_year[year] = (raw - global_min) / (global_max - global_min) * 100
```

This produces a genuine time series of Blue Zone scores per region — used for trend arrows, sparklines, and the projection baseline.

### 5 — Clustering via K-Means

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, random_state=42, n_init=20)
km.fit(blue_zone_scores.reshape(-1, 1))
```

K-Means on the single score dimension, producing four interpretable archetypes:

| Cluster | Score range | Label |
|---|---|---|
| 0 | > 70 | 🔵 Blue Zone |
| 1 | 50–70 | 🟢 Close |
| 2 | 35–50 | 🟡 Moderate |
| 3 | < 35 | 🔴 Low |

### 6 — Validation

| Metric | Target | Interpretation |
|---|---|---|
| r(score, life expectancy) | > 0.65 | Score captures health outcomes |
| r(score, happiness) | > 0.55 | Score captures subjective wellbeing |
| r(score, GDP) | 0.35–0.50 | Score is related to but not reducible to wealth |
| PC1 explained variance | > 35% | A genuine dominant axis exists |

The GDP correlation is deliberately expected to be *moderate* — lower than the life expectancy correlation. This is the empirical demonstration of the project's central claim.

### 7 — GDP non-linearity (Preston Curve)

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

log_gdp = np.log(gdp_per_capita).reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
model = LinearRegression().fit(poly.fit_transform(log_gdp), blue_zone_scores)

# Peak GDP (where marginal return on score = 0)
b, c = model.coef_[1], model.coef_[2]
peak_gdp = np.exp(-b / (2 * c))   # typically €28,000–32,000
```

The fitted polynomial coefficients are saved and used by the GDP slider in the frontend.

---

## Scenario engine

The Blue Zone score for any region, given a hypothetical KPI vector, is:

```
score = dot(standardise(KPI_vector, saved_means, saved_stds), pca_loadings)
```

This is computed in JavaScript in the browser:

```js
// scorer.js
import { loadings, means, stds, globalMin, globalMax } from './artifacts/pca_loadings.json';

export function computeScore(kpiVector) {
  const standardised = kpiVector.map((v, i) => (v - means[i]) / stds[i]);
  const raw = standardised.reduce((sum, v, i) => sum + v * loadings[i], 0);
  return Math.max(0, Math.min(100, (raw - globalMin) / (globalMax - globalMin) * 100));
}
```

No server round-trip. Instantaneous. Safe for live demo.

---

## Policy chat

The chat input sends legislation text to the Claude API with a structured prompt that constrains the response to a JSON object mapping KPI names to estimated percentage-point changes:

```python
SYSTEM_PROMPT = """
You are a policy analyst with expertise in EU regional statistics.
Given the current KPI profile for {region} and the following policy text,
return ONLY a valid JSON object mapping KPI names to estimated
percentage-point changes over 5 years. Use 0 for unaffected KPIs.
Never return text outside the JSON object.

KPI names: poverty_rate, unemployment_rate, pm25, life_expectancy, trust

Current profile: {kpi_dict}
"""
```

The returned JSON is validated, applied as deltas to the slider values, and the score recomputed in-browser.

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- An [Anthropic API key](https://console.anthropic.com) for the policy chat feature

### Backend

```bash
git clone https://github.com/your-org/bluezone-explorer
cd bluezone-explorer

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run data pipeline (downloads ~200MB from Eurostat + EEA)
python notebooks/01_data_collection.py
python notebooks/02_preprocessing.py
python notebooks/03_pca_scoring.py      # generates data/artifacts/
python notebooks/04_clustering.py
python notebooks/05_validation.py

# Start API
cd api && uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # add your Anthropic API key
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### Environment variables

```
ANTHROPIC_API_KEY=sk-ant-...
API_BASE_URL=http://localhost:8000
```

---

## Data sources and licences

| Source | Licence | URL |
|---|---|---|
| Eurostat regional statistics | [CC BY 4.0](https://ec.europa.eu/eurostat/about/policies/copyright) | ec.europa.eu/eurostat |
| EEA air quality data | [EEA standard reuse policy](https://www.eea.europa.eu/legal/copyright) | eea.europa.eu |
| NUTS2 GeoJSON | [CC BY 4.0](https://github.com/eurostat/Nuts2json/blob/master/LICENSE) | github.com/eurostat/Nuts2json |

All source data is publicly available and free to reuse with attribution.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline | Python, pandas, scikit-learn, scipy |
| API | FastAPI, Pydantic |
| Frontend | React 18, Tailwind CSS, Vite |
| Map | Leaflet.js + Eurostat NUTS2 GeoJSON |
| Charts | Chart.js |
| Policy chat | Claude API (claude-sonnet-4-6) |
| Deployment | Render / Railway |

---

## Key design decisions

**PCA loadings in the browser, not the server.** Moving a slider triggers an instant dot product in JavaScript rather than a fetch call. This eliminates network latency during the live demo and removes a backend dependency from the most visible interaction.

**Life expectancy as validator, not feature.** Life expectancy is kept outside the PCA feature matrix and used only to orient the sign of PC1 and validate the result. This ensures the score measures the *conditions for* longevity rather than redundantly capturing longevity itself.

**Country-level social indicators assigned to regions.** Trust and life satisfaction exist only at country level in Eurostat. Rather than dropping the social dimension, country-level values are assigned to all NUTS2 regions within a country. This is documented explicitly and means the score captures within-country variation on four economic and environmental variables, plus between-country variation on social cohesion.

**GDP outside the model.** GDP per capita is deliberately excluded from the PCA feature matrix. Its role is as an external comparator variable — the non-linear scatter plot showing diminishing returns above ~€30k per capita is the project's central empirical argument.

---

## Limitations

- Social indicators (trust, life satisfaction) are at country level — within-country variation on the social dimension is not captured
- Obesity, physical activity, and green space are not available at NUTS2 level in Eurostat and are therefore excluded
- The scenario projection is a linear extrapolation of historical trends — it is indicative, not a causal forecast
- Missing NUTS2 data has been imputed using country means — imputed values are flagged in the data artifacts but not always visible in the UI
- The policy chat relies on an LLM to estimate KPI impacts — effect sizes are approximate and should be treated as illustrative

---

## The research context

This project applies the **Preston Curve** — the well-documented non-linear relationship between income and life expectancy first described by Samuel Preston (1975) — to European regional data. The curve flattens sharply above ~$15,000 PPP and plateaus around $28–32,000. Above this threshold, lifestyle, environment, and social cohesion explain far more of the variation in longevity than income does.

The Blue Zone concept was developed by Dan Buettner in collaboration with National Geographic and demographers including Michel Poulain and Gianni Pes, who identified statistically significant clusters of centenarians in specific geographic regions.

---

## Built at Le Wagon

This project was built in 10 days as a final project for the [Le Wagon](https://www.lewagon.com) Data Science & AI bootcamp.

---

*"Your postcode matters more than your paycheck."*
