import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_appraisal():
    files = list((RAW_DIR / "dcad_appraisal").glob("*.csv"))
    if not files:
        print("❌ No appraisal CSV found")
        return None

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded appraisal rows: {len(df)}")
    return df

def main():
    df = load_appraisal()
    if df is None:
        return

    # Minimal cleaning
    df = df.dropna()

    out_path = OUT_DIR / "full_cached_dataset_cleaned_flat.parquet"
    df.to_parquet(out_path)

    print(f"✅ Saved dataset → {out_path}")

if __name__ == "__main__":
    main()
