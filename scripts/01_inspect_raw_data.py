from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
APPRAISAL_DIR = ROOT / "data/raw/dcad_appraisal"
GIS_DIR = ROOT / "data/raw/dcad_gis"

def inspect_csvs():
    print("\n=== Appraisal files ===")
    for p in sorted(APPRAISAL_DIR.glob("*")):
        if p.suffix.lower() in [".csv", ".txt"]:
            print(f"\n{p.name}")
            try:
                df = pd.read_csv(p, nrows=5, low_memory=False)
                print("columns:", list(df.columns)[:40])
                print(df.head())
            except Exception as e:
                print("Could not read:", e)

def inspect_shapefiles():
    print("\n=== GIS shapefiles ===")
    for p in sorted(GIS_DIR.rglob("*.shp")):
        print(f"\n{p}")
        try:
            gdf = gpd.read_file(p, rows=5)
            print("crs:", gdf.crs)
            print("columns:", list(gdf.columns)[:40])
            print(gdf.head())
        except Exception as e:
            print("Could not read:", e)

if __name__ == "__main__":
    inspect_csvs()
    inspect_shapefiles()
