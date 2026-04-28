# UrbanScape Property Evaluation Starter

This is a clean rebuild scaffold for the Dallas County parcel/property evaluation app.

## What this package includes
- Folder structure for raw DCAD appraisal and GIS shapefile data
- ETL scripts to convert shapefiles/CSV/TXT files into flat and geospatial parquet files
- Training script for 5th/50th/95th percentile LightGBM quantile models
- Streamlit app for address/parcel-style property evaluation
- Placeholder folders for models and processed data

## What this package does NOT include
The original Dallas County Appraisal District data files or shapefiles are not included. Download those directly from Dallas Central Appraisal District data product pages, then place them in:

- `data/raw/dcad_appraisal/`
- `data/raw/dcad_gis/`

## Quick start

```bash
cd urbanscape_property_eval_starter
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Put the DCAD appraisal CSV/TXT files in `data/raw/dcad_appraisal/`.
Put the unzipped parcel shapefile files in `data/raw/dcad_gis/`.

Then run:

```bash
python scripts/01_inspect_raw_data.py
python scripts/02_build_dataset.py
python scripts/03_train_quantile_models.py
streamlit run app/streamlit_app.py
```

## Expected output files

- `data/processed/full_cached_dataset_cleaned_flat.parquet`
- `data/processed/full_cached_dataset_cleaned_geo.parquet`
- `models/<property_type>/quantile_005.txt`
- `models/<property_type>/quantile_050.txt`
- `models/<property_type>/quantile_095.txt`
- `models/<property_type>/features.json`
