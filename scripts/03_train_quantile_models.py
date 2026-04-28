from pathlib import Path
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/full_cached_dataset_cleaned_flat.parquet"
MODELS = ROOT / "models"

TARGET_CANDIDATES = ["tot_val","total_value","market_value","mkt_val","appraised_value","appraised_val","prev_mkt_val"]
FEATURE_CANDIDATES = ["shape_area","front_dim","depth_dim","lat","lon","zoning_score","prev_mkt_val","impr_val","total_taxable_val"]

def infer_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors="coerce")
            if y.notna().sum() > 100:
                print("Using target:", c)
                return c
    raise ValueError(f"No usable target found. Add one of: {TARGET_CANDIDATES}")

def property_type_series(df):
    if "property_type" in df.columns:
        return df["property_type"].fillna("residential").astype(str).str.lower()
    return pd.Series(["residential"] * len(df), index=df.index)

def train_one(df, property_type):
    target = infer_target(df)
    features = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not features:
        raise ValueError("No usable features found. Check ETL output columns.")

    work = df[features + [target]].copy()
    for c in features + [target]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[target])
    work[features] = work[features].fillna(work[features].median(numeric_only=True))

    if len(work) < 200:
        print(f"Skipping {property_type}: not enough rows ({len(work)})")
        return

    X_train, X_test, y_train, y_test = train_test_split(work[features], work[target], test_size=0.2, random_state=42)

    outdir = MODELS / property_type
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "features.json", "w") as f:
        json.dump({"features": features, "target": target}, f, indent=2)

    for q, name in [(0.05, "quantile_005.txt"), (0.50, "quantile_050.txt"), (0.95, "quantile_095.txt")]:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        model.booster_.save_model(str(outdir / name))
        print(f"{property_type} q={q}: MAE={mae:,.0f}, saved {name}")

def main():
    df = pd.read_parquet(DATA)
    types = property_type_series(df)
    for pt in ["residential","commercial","agricultural"]:
        subset = df[types.str.contains(pt, na=False)].copy()
        if subset.empty and pt == "residential":
            subset = df.copy()
        train_one(subset, pt)

if __name__ == "__main__":
    main()
