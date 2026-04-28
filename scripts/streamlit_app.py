import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import streamlit as st

# ---------------- CONFIG ---------------- #

DATA_PATH = Path("data/processed/full_cached_dataset_cleaned_flat.parquet")
MODEL_BASE = Path("models")
TARGET = "tot_val"

# ---------------- UI SETUP ---------------- #

st.set_page_config(
    page_title="UrbanScape Property Evaluator",
    layout="wide"
)

st.title("UrbanScape Property Evaluator")
st.caption("Dallas County parcel/property valuation app")

# ---------------- HELPERS ---------------- #

def money(x):
    try:
        return f"${float(x):,.0f}"
    except:
        return "N/A"


def clean_numeric_series(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)

    # Convert key numeric columns
    for col in ["tot_val", "land_val", "impr_val", "shape_area"]:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])

    return df


@st.cache_resource
def load_model_bundle(segment):
    model_dir = MODEL_BASE / segment

    if not model_dir.exists():
        raise ValueError(f"Model folder not found: {segment}")

    # 🔥 FIX: handles BOTH list and dict formats
    with open(model_dir / "features.json", "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        features = raw
    elif isinstance(raw, dict):
        features = raw["features"]
    else:
        raise ValueError("Invalid features.json format")

    models = {
        "low": lgb.Booster(model_file=str(model_dir / "quantile_005.txt")),
        "mid": lgb.Booster(model_file=str(model_dir / "quantile_050.txt")),
        "high": lgb.Booster(model_file=str(model_dir / "quantile_095.txt")),
    }

    return features, models


def prepare_row(row, features):
    X = pd.DataFrame([row])

    # Convert all columns to numeric safely
    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    # Ensure all required features exist
    for col in features:
        if col not in X.columns:
            X[col] = 0

    X = X[features]
    X = X.fillna(0)

    return X


# ---------------- LOAD DATA ---------------- #

if not DATA_PATH.exists():
    st.error(f"Missing dataset: {DATA_PATH}")
    st.stop()

df = load_data()

# ---------------- MODEL SELECTION ---------------- #

model_options = sorted([p.name for p in MODEL_BASE.iterdir() if p.is_dir()])

if not model_options:
    st.error("No trained models found in /models/")
    st.stop()

segment = st.selectbox("Property type", model_options)

# ---------------- LOAD MODEL ---------------- #

try:
    features, models = load_model_bundle(segment)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.success(f"Loaded {segment} model with {len(features)} features")

# ---------------- SAMPLE PREDICTION ---------------- #

sample = df.sample(1).iloc[0]

X = prepare_row(sample, features)

pred_low = models["low"].predict(X)[0]
pred_mid = models["mid"].predict(X)[0]
pred_high = models["high"].predict(X)[0]

# ---------------- DISPLAY ---------------- #

st.subheader("Prediction")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Low (5%)", money(pred_low))

with c2:
    st.metric("Median (50%)", money(pred_mid))

with c3:
    st.metric("High (95%)", money(pred_high))


# ---------------- PROPERTY DETAILS ---------------- #

st.subheader("Sample Property")

show_cols = [
    "account_num",
    "sptd_desc",
    "tot_val",
    "land_val",
    "impr_val",
    "shape_area"
]

existing_cols = [c for c in show_cols if c in sample.index]

st.dataframe(sample[existing_cols].to_frame("value"))