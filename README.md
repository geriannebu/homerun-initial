# HomeRun SG

A Streamlit web app that helps Singapore residents discover, price, and compare HDB resale flats. It combines quiz-driven preference matching, amenity proximity scoring, and an ensemble ML price prediction model trained on 228,000 HDB resale transactions (2017–2026).

---

## Features

- **Discover** — Swipe-deck of recommended listings ranked by a weighted amenity + value score based on your quiz preferences
- **Saved** — Cross-session shortlist of flats with predicted prices and amenity scores
- **Compare** — Side-by-side comparison of saved flats
- **Explore** — Look up full transaction history for any HDB block since 2017; predict current value using real block-level spatial data; or model-price any hypothetical flat profile and compare against recent market medians
- **Account** — Edit preferences and view search session history

---

## App Flow

```
Landing → Auth (create account / log in)
       → Onboarding (town, flat type, budget, amenity priorities)
       → Auto-search (deck generated from preferences)
       → Discover (swipe interface)
       → Saved / Compare / Explore / Account
```

---

## Project Structure

```
app.py                          # Streamlit entry point
requirements.txt

frontend/
  pages/
    explore.py                  # Transaction lookup + price prediction (Explore tab)
    saved.py                    # Saved flats view
    comparison_tool.py          # Side-by-side flat comparison
    account.py                  # Auth, preferences, session history
  flat_outputs/
    best_matches.py             # Swipe deck, listing cards, detail modal
    map_view.py                 # Pydeck map with amenity highlights
  components/
    listing_detail.py           # Listing detail modal component
  styles/
    css.py                      # Global CSS injection
  state/
    session.py                  # Session state helpers
  assets/
    homerun_logo.png
    homerun_icon.png

backend/
  services/
    recommender.py              # Core ranking engine (amenity + value scoring)
    predictor_service.py        # Prediction bundle for active listings
    recommendation_service.py   # Town-level recommendations
    listings_service.py         # Listing filters
    map_service.py              # Lat/lon helpers
    quiz.py                     # Onboarding quiz
  utils/
    constants.py                # Towns, flat types, amenity labels, coordinates
    formatters.py               # fmt_sgd(), valuation_tag_html()
    scoring.py                  # compute_listing_scores() — distance-bracket amenity scoring
  schemas/
    inputs.py                   # UserInputs dataclass

data/
  load_data.py                  # Loads and normalises listings CSV with @st.cache_data
  final.csv                     # Base listings dataset

backend_predictor_listings/
  price_predictor/
    notebooks/
      predict_hypothetical.py   # Public API: predict_hypothetical(), predict_with_spatial_overrides()
      data_preprocessing.ipynb
      model_training.ipynb
      predict_current_listings.ipynb
    models/
      cb_model.cbm              # CatBoost
      lgb_model.zip             # LightGBM
      xgb_model.ubj             # XGBoost
      ensemble_weights.npy
    csv_outputs/
      feature_df.csv            # 228k HDB transactions 2017–2026 with spatial features
      feature_df_raw.zip        # Raw version used for group-median imputation
      listings_with_walking_times_full.csv  # Final scored listings used by the app
    json_outputs/
      ci_offsets.json           # 95% confidence interval offsets (real price space)
  datasets/
    HDBResalePriceIndex1Q2009100Quarterly.csv
```

---

## ML Price Prediction

The price prediction pipeline lives in `backend_predictor_listings/price_predictor/notebooks/predict_hypothetical.py` and is loaded once at import time.

**Model**: Equal-weighted ensemble of CatBoost, XGBoost, and LightGBM — all trained on 228,000 HDB resale transactions from January 2017 to early 2026.

**Features (37 total)**:
- Flat attributes: town, flat type, floor area, storey midpoint, remaining lease, lease commence date
- Spatial: lat/lon, nearest 3 distances to MRT, bus stop, primary school, hawker centre, mall, polyclinic, supermarket
- Amenity counts: MRTs within 1 km, primary schools within 1 km, hawkers within 500 m, bus stops within 400 m
- Distance to CBD
- Time index (month_index, anchored Jan 2017 = 0)

**Price scaling**: Model outputs prices in 2009-base real SGD. These are scaled to nominal current prices using the HDB Resale Price Index (current: 203.4 for 2026-Q1 flash estimate).

**Spatial imputation**: When a specific block is not provided (Tab 2 — Explore a flat profile), location features are imputed from the median of all transactions sharing the same (town, flat_type) pair. For Tab 1 (Look up a flat), the actual spatial values for that block are read directly from `feature_df.csv` — all units in the same block share identical spatial features.

**Confidence intervals**: Global 2.5th/97.5th percentile offsets (in real price space) stored in `ci_offsets.json` and applied after scaling.

---

## Amenity Scoring

Two scoring systems are used depending on context:

**Recommender (`recommender.py`)** — used for ranking active listings in the swipe deck:
- Exponential decay on walking times: `score = mean(exp(-t / τ))` where τ is amenity-specific (e.g. MRT τ=18 min, bus τ=4 min)
- Weighted by rank-sum weights from the onboarding quiz
- Combined with a value score: `final = α × amenity_score + (1−α) × value_score`

**Scoring utility (`scoring.py`)** — used for saved/compared flats:
- Distance brackets: ≤300 m → 90, ≤600 m → 75, ≤1000 m → 60, >1000 m → 40
- Weighted average across amenity types by user preferences

---

## Explore Tab

### Tab 1 — Look up a flat
Search any HDB block by address. The app shows all resale transactions recorded since 2017, then lets you predict the current value of a specific unit with user-chosen floor area, sale date (auto-calculates remaining lease from `lease_commence_date`), and storey. The prediction uses the **actual spatial features** for that block from `feature_df.csv` — not estimated averages.

### Tab 2 — Explore a flat profile
Price any hypothetical flat profile by choosing town, flat type, floor area, remaining lease, and storey. Shows two numbers side-by-side:
- **Model estimate** — ML fair-value with 95% confidence range (spatial features imputed from town/flat-type medians)
- **Median transacted** — Median price buyers actually paid for similar flats (same town, flat type, ±20 sqm, similar lease) in the past 6 months

---

## Setup

### Requirements

- Python 3.12
- On macOS, XGBoost requires OpenMP:
  ```bash
  brew install libomp
  ```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

### Run with Docker

#### 1. Build the image
docker build -t homerun-sg .

#### 2. Run the app
docker run -p 8501:8501 homerun-sg

If port 8501 is busy, use:
docker run -p 8502:8501 homerun-sg
---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.55.0 | Web framework |
| pandas | 2.2.2 | Data manipulation |
| numpy | 2.0.0 | Numerical ops |
| pydeck | 0.9.1 | Map visualisation |
| scikit-learn | 1.8.0 | ML utilities |
| lightgbm | 4.6.0 | Ensemble model |
| xgboost | 3.2.0 | Ensemble model |
| catboost | 1.2.10 | Ensemble model |
| joblib | 1.5.3 | Model serialisation |

---

## Data Sources

- **HDB Resale Transactions** — data.gov.sg, Jan 2017 – early 2026
- **HDB Resale Price Index** — HDB quarterly RPI (2009 base = 100); 2026-Q1 flash estimate: 203.4
- **Walking times** — Pre-computed and stored in `listings_with_walking_times_full.csv`
- **Active listings** — Scraped/compiled listings with asking prices, stored in `final.csv`
