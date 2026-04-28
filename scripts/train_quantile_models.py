import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/full_cached_dataset_cleaned_flat.parquet")
BASE_MODEL_DIR = Path("models")

TARGET = "tot_val"

PROPERTY_SEGMENTS = {
    "residential": [
        "SINGLE FAMILY",
        "TOWNHOUSE",
        "CONDOMINIUM",
        "MOBILE HOME",
    ],
    "multifamily": [
        "MFR",
        "DUPLEX",
        "APARTMENT",
    ],
    "commercial": [
        "COMMERCIAL",
    ],
    "land": [
        "SFR - VACANT",
        "RESIDENTIAL - VACANT",
        "RURAL VACANT",
        "QUALIFIED OPEN SPACE",
        "RURAL LAND",
    ],
    "industrial": [
        "INDUSTRIAL",
    ],
}

DROP_COLS = [
    "acct",
    "account_num",
    "recacs",
    "gis_parcel_id",
    "taxpayer_rep",
    "extrnl_cnty_acct",
    "extrnl_city_acct",
    "sptd_desc",
    "zoning",
    "area_uom_desc",
    "pricing_meth_desc",
    "city_juris_desc",
    "county_juris_desc",
    "isd_juris_desc",
    "hospital_juris_desc",
    "college_juris_desc",
    "special_dist_juris_desc",
    "zoning_group",
    "fld_zone",
    "sfha_tf",
]


def clean_numeric_series(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def filter_segment(df, keywords):
    mask = pd.Series(False, index=df.index)

    for kw in keywords:
        mask = mask | df["sptd_desc"].astype(str).str.contains(
            kw,
            case=False,
            na=False,
            regex=False,
        )

    return df[mask].copy()


def prep_data(df, segment_name, keywords):
    print("\n" + "=" * 70)
    print(f"Preparing segment: {segment_name.upper()}")
    print("=" * 70)

    seg = filter_segment(df, keywords)

    print(f"Rows after segment filter: {len(seg):,}")

    if len(seg) < 500:
        print(f"Skipping {segment_name}: not enough rows.")
        return None

    if TARGET not in seg.columns:
        raise ValueError(f"{TARGET} not found. Columns: {list(seg.columns)}")

    seg[TARGET] = clean_numeric_series(seg[TARGET])
    seg = seg[seg[TARGET].notna()]
    seg = seg[seg[TARGET] > 0]

    print(f"Rows after target cleaning: {len(seg):,}")

    if len(seg) < 500:
        print(f"Skipping {segment_name}: not enough valid target rows.")
        return None

    y = seg[TARGET]

    X = seg.drop(columns=[TARGET] + DROP_COLS, errors="ignore")

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    X = X.dropna(axis=1, how="all")
    X = X.fillna(0)

    nunique = X.nunique(dropna=False)
    useful_cols = nunique[nunique > 1].index.tolist()
    X = X[useful_cols]

    print(f"Final training rows: {len(X):,}")
    print(f"Feature count: {X.shape[1]}")
    print("Features:")
    for col in X.columns:
        print(f"  - {col}")

    if X.shape[1] == 0:
        print(f"Skipping {segment_name}: no usable numeric features.")
        return None

    model_dir = BASE_MODEL_DIR / segment_name
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "features.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    metadata = {
        "segment": segment_name,
        "keywords": keywords,
        "target": TARGET,
        "row_count": int(len(X)),
        "feature_count": int(X.shape[1]),
        "features": list(X.columns),
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return train_test_split(X, y, test_size=0.2, random_state=42), model_dir


def train_segment(df, segment_name, keywords):
    result = prep_data(df, segment_name, keywords)

    if result is None:
        return

    (X_train, X_test, y_train, y_test), model_dir = result

    quantiles = [
        (0.05, "quantile_005"),
        (0.50, "quantile_050"),
        (0.95, "quantile_095"),
    ]

    for alpha, name in quantiles:
        print(f"\nTraining {segment_name} model: {name}")

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=40,
            random_state=42,
            force_col_wise=True,
        )

        model.fit(X_train, y_train)

        out_path = model_dir / f"{name}.txt"
        model.booster_.save_model(out_path)

        print(f"Saved model: {out_path}")


def main():
    print(f"Loading dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print(f"Loaded total rows: {len(df):,}")

    print("\nProperty type counts:")
    print(df["sptd_desc"].value_counts(dropna=False).head(30))

    for segment_name, keywords in PROPERTY_SEGMENTS.items():
        train_segment(df, segment_name, keywords)

    print("\nDone. Models saved in:")
    print(BASE_MODEL_DIR.resolve())


if __name__ == "__main__":
    main()