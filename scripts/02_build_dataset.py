from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_APPRAISAL = ROOT / "data/raw/dcad_appraisal"
RAW_GIS = ROOT / "data/raw/dcad_gis"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

JOIN_KEYS = ["ACCOUNT_NUM","ACCT","acct","account","account_num","PARCEL_ID","parcel_id","PROP_ID","property_id"]

def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_join_key(left_cols, right_cols):
    for k in JOIN_KEYS:
        if k in left_cols and k in right_cols:
            return k
    left_lower = {c.lower(): c for c in left_cols}
    right_lower = {c.lower(): c for c in right_cols}
    for k in JOIN_KEYS:
        if k.lower() in left_lower and k.lower() in right_lower:
            return left_lower[k.lower()], right_lower[k.lower()]
    return None

def read_first_appraisal_table():
    candidates = list(RAW_APPRAISAL.glob("*.csv")) + list(RAW_APPRAISAL.glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No CSV/TXT appraisal files found in {RAW_APPRAISAL}")
    print("Using appraisal file:", candidates[0])
    # Try comma, then pipe, then tab
    for sep in [",", "|", "\t"]:
        try:
            df = pd.read_csv(candidates[0], sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return normalize_cols(df)
        except Exception:
            pass
    raise ValueError("Could not parse appraisal file. Check delimiter and encoding.")

def read_first_parcel_shapefile():
    shp = list(RAW_GIS.rglob("*.shp"))
    if not shp:
        raise FileNotFoundError(f"No .shp files found in {RAW_GIS}")
    print("Using shapefile:", shp[0])
    gdf = gpd.read_file(shp[0])
    gdf = normalize_cols(gdf)
    if gdf.crs is None:
        print("WARNING: shapefile CRS missing; assuming EPSG:4326. Change this if wrong.")
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["shape_area"] = gdf.to_crs("EPSG:2276").geometry.area
    return gdf

def add_basic_features(df):
    df = df.copy()
    numeric_candidates = [
        "shape_area", "front_dim", "depth_dim", "lat", "lon",
        "prev_mkt_val", "impr_val", "land_val", "tot_val",
        "CITY_TAXABLE_VAL", "COUNTY_TAXABLE_VAL", "SD_TAXABLE_VAL",
        "HOSPITAL_TAXABLE_VAL", "COLLEGE_TAXABLE_VAL"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    tax_cols = [c for c in ["CITY_TAXABLE_VAL","COUNTY_TAXABLE_VAL","SD_TAXABLE_VAL","HOSPITAL_TAXABLE_VAL","COLLEGE_TAXABLE_VAL"] if c in df.columns]
    if tax_cols:
        df["total_taxable_val"] = df[tax_cols].sum(axis=1)
    if "property_type" not in df.columns:
        # Heuristic placeholder. Replace with your DCAD use-code mapping.
        df["property_type"] = "residential"
    if "zoning_score" not in df.columns:
        df["zoning_score"] = 0.5
    return df

def main():
    appraisal = read_first_appraisal_table()
    parcels = read_first_parcel_shapefile()

    jk = find_join_key(appraisal.columns, parcels.columns)
    if jk is None:
        print("No common join key found. Writing GIS-only dataset with appraisal columns absent.")
        gdf = parcels
    else:
        if isinstance(jk, tuple):
            left_key, right_key = jk
        else:
            left_key = right_key = jk
        print(f"Joining appraisal.{left_key} to parcels.{right_key}")
        appraisal[left_key] = appraisal[left_key].astype(str).str.strip()
        parcels[right_key] = parcels[right_key].astype(str).str.strip()
        gdf = parcels.merge(appraisal, left_on=right_key, right_on=left_key, how="left", suffixes=("", "_appraisal"))

    gdf = add_basic_features(gdf)
    geo_path = OUT / "full_cached_dataset_cleaned_geo.parquet"
    flat_path = OUT / "full_cached_dataset_cleaned_flat.parquet"
    gdf.to_parquet(geo_path)
    pd.DataFrame(gdf.drop(columns="geometry")).to_parquet(flat_path)
    print("Saved:", geo_path)
    print("Saved:", flat_path)
    print("Rows:", len(gdf), "Columns:", len(gdf.columns))

if __name__ == "__main__":
    main()
