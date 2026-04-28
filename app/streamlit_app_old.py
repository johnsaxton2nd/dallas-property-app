from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/full_cached_dataset_cleaned_flat.parquet"
MODELS = ROOT / "models"

st.set_page_config(page_title="UrbanScape Property Evaluator", layout="wide")
st.title("UrbanScape Property Evaluator")
st.caption("Dallas County parcel/property evaluation starter app")

@st.cache_data
def load_data():
    if not DATA.exists():
        return None
    return pd.read_parquet(DATA)

def load_model_bundle(property_type):
    d = MODELS / property_type
    features_path = d / "features.json"
    if not features_path.exists():
        return None
    meta = json.loads(features_path.read_text())
    models = {}
    for label, fname in [("p05","quantile_005.txt"),("p50","quantile_050.txt"),("p95","quantile_095.txt")]:
        path = d / fname
        if path.exists():
            models[label] = lgb.Booster(model_file=str(path))
    return meta["features"], models

df = load_data()
if df is None:
    st.error("No processed dataset found. Run: python scripts/02_build_dataset.py")
    st.stop()

property_type = st.selectbox("Property type", ["residential", "commercial", "agricultural"])
bundle = load_model_bundle(property_type)
if bundle is None:
    st.warning("No trained model found yet. Run: python scripts/03_train_quantile_models.py")
    st.dataframe(df.head(25))
    st.stop()

features, models = bundle

st.sidebar.header("Input parcel features")
input_values = {}
for f in features:
    default = float(pd.to_numeric(df[f], errors="coerce").median()) if f in df.columns else 0.0
    input_values[f] = st.sidebar.number_input(f, value=default)

x = pd.DataFrame([input_values])[features]
preds = {k: float(m.predict(x)[0]) for k, m in models.items()}

c1, c2, c3 = st.columns(3)
c1.metric("Low estimate / P05", f"${preds.get('p05', np.nan):,.0f}")
c2.metric("Median estimate / P50", f"${preds.get('p50', np.nan):,.0f}")
c3.metric("High estimate / P95", f"${preds.get('p95', np.nan):,.0f}")

fig = go.Figure()
fig.add_trace(go.Indicator(
    mode="number+gauge",
    value=preds.get("p50", 0),
    number={"prefix":"$"},
    gauge={"axis":{"range":[max(0, preds.get("p05",0)*0.8), preds.get("p95",1)*1.2]}}
))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Nearest comparable parcels")
usable = [f for f in features if f in df.columns]
work = df[usable].copy()
for c in usable:
    work[c] = pd.to_numeric(work[c], errors="coerce")
work = work.fillna(work.median(numeric_only=True))
nn = NearestNeighbors(n_neighbors=min(10, len(work)))
nn.fit(work)
dist, idx = nn.kneighbors(x[usable])
cols = [c for c in ["lat","lon","shape_area","prev_mkt_val","impr_val","total_taxable_val"] if c in df.columns]
st.dataframe(df.iloc[idx[0]][cols].assign(distance=dist[0]))
