import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/full_cached_dataset_cleaned_flat.parquet")
MODEL_BASE = Path("models")
TARGET = "tot_val"

SEGMENT_RULES = {
    "residential": ["SINGLE FAMILY", "TOWNHOUSE", "CONDOMINIUM", "MOBILE HOME"],
    "multifamily": ["MFR", "DUPLEX", "APARTMENT"],
    "commercial": ["COMMERCIAL"],
    "land": ["VACANT", "RURAL", "OPEN SPACE"],
    "industrial": ["INDUSTRIAL"],
}

COMP_FEATURES = [
    "lat",
    "lon",
    "shape_area",
    "front_dim",
    "depth_dim",
    "area_size",
    "land_val",
    "impr_val",
    "prev_mkt_val",
    "zoning_score",
]


def money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "N/A"


def clean_numeric_series(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)

    for col in COMP_FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])

    return df


@st.cache_resource
def load_model_bundle(segment):
    model_dir = MODEL_BASE / segment

    with open(model_dir / "features.json", "r") as f:
        raw = json.load(f)

    features = raw if isinstance(raw, list) else raw["features"]

    models = {
        "low": lgb.Booster(model_file=str(model_dir / "quantile_005.txt")),
        "mid": lgb.Booster(model_file=str(model_dir / "quantile_050.txt")),
        "high": lgb.Booster(model_file=str(model_dir / "quantile_095.txt")),
    }

    return features, models


def detect_segment(desc):
    text = str(desc).upper()

    for seg, keywords in SEGMENT_RULES.items():
        if any(k in text for k in keywords):
            return seg

    return None


def geocode_address(address):
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"

    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "format": "json",
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    matches = r.json().get("result", {}).get("addressMatches", [])

    if not matches:
        return None

    coords = matches[0]["coordinates"]

    return {
        "matched_address": matches[0].get("matchedAddress"),
        "lat": coords["y"],
        "lon": coords["x"],
    }


def nearest_property(df, lat, lon):
    geo = df.dropna(subset=["lat", "lon"]).copy()
    geo["geo_distance"] = ((geo["lat"] - lat) ** 2 + (geo["lon"] - lon) ** 2) ** 0.5
    return geo.sort_values("geo_distance").iloc[0].copy()


def apply_user_inputs(row, lot_size, building_size, beds, baths):
    row = row.copy()

    if lot_size and lot_size > 0:
        if "shape_area" in row.index:
            row["shape_area"] = lot_size
        if "area_size" in row.index:
            row["area_size"] = lot_size

    if building_size and building_size > 0:
        row["building_size_sqft"] = building_size

    row["beds"] = beds if beds else 0
    row["baths"] = baths if baths else 0
    row["lot_size_sqft"] = lot_size if lot_size else 0

    return row


def prepare_features(row, features):
    X = pd.DataFrame([row])

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    for col in features:
        if col not in X.columns:
            X[col] = 0

    X = X[features]
    X = X.fillna(0)

    return X


def find_comps(df, row, segment, k=10):
    comp = df.copy()
    comp["segment"] = comp["sptd_desc"].apply(detect_segment)
    comp = comp[comp["segment"] == segment].copy()

    usable = [c for c in COMP_FEATURES if c in comp.columns and c in row.index]

    comp = comp.dropna(subset=usable + [TARGET])

    if len(comp) < 5:
        comp["similarity_distance"] = None
        return comp.head(k), usable

    query = pd.DataFrame([{c: row[c] for c in usable}])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(comp[usable])
    q_scaled = scaler.transform(query)

    nn = NearestNeighbors(n_neighbors=min(k, len(comp)))
    nn.fit(X_scaled)

    distances, indices = nn.kneighbors(q_scaled)

    comps = comp.iloc[indices[0]].copy()
    comps["similarity_distance"] = distances[0]

    return comps, usable


def confidence_score(pred_mid, comp_median, asking_price, avg_distance):
    agreement_gap = abs(pred_mid - comp_median) / max(pred_mid, comp_median, 1)
    agreement_score = max(0, 1 - agreement_gap)

    distance_score = max(0, 1 - min(avg_distance / 5, 1))

    if asking_price and asking_price > 0:
        ask_gap = abs(pred_mid - asking_price) / max(pred_mid, asking_price, 1)
        ask_score = max(0, 1 - ask_gap)
        score = 0.45 * agreement_score + 0.35 * distance_score + 0.20 * ask_score
    else:
        score = 0.60 * agreement_score + 0.40 * distance_score

    return round(score * 100)


def agreement_label(pred_mid, comp_median):
    gap = abs(pred_mid - comp_median) / max(pred_mid, comp_median, 1)

    if gap <= 0.10:
        return "High"
    if gap <= 0.20:
        return "Medium"
    return "Low"


def make_map(comps, nearest, geo):
    map_df = comps.copy()

    target_point = pd.DataFrame(
        [
            {
                "account_num": "SUBJECT / NEAREST PROPERTY",
                "sptd_desc": str(nearest.get("sptd_desc", "")),
                "tot_val": nearest.get(TARGET),
                "lat": geo["lat"],
                "lon": geo["lon"],
                "point_type": "Subject Property",
            }
        ]
    )

    map_df["point_type"] = "Comparable Property"

    map_cols = ["account_num", "sptd_desc", "tot_val", "lat", "lon", "point_type"]

    for col in map_cols:
        if col not in map_df.columns:
            map_df[col] = None

    map_df = pd.concat(
        [target_point[map_cols], map_df[map_cols]],
        ignore_index=True,
    )

    map_df = map_df.dropna(subset=["lat", "lon"])
    map_df["tot_val_label"] = map_df["tot_val"].apply(money)

    subject_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df[map_df["point_type"] == "Subject Property"],
        get_position="[lon, lat]",
        get_radius=90,
        get_fill_color=[255, 0, 0, 190],
        pickable=True,
    )

    comp_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df[map_df["point_type"] == "Comparable Property"],
        get_position="[lon, lat]",
        get_radius=60,
        get_fill_color=[0, 120, 255, 160],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(geo["lat"]),
        longitude=float(geo["lon"]),
        zoom=14,
        pitch=0,
    )

    tooltip = {
        "html": """
        <b>{point_type}</b><br/>
        Account: {account_num}<br/>
        Type: {sptd_desc}<br/>
        Value: {tot_val_label}
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black",
        },
    }

    return pdk.Deck(
        initial_view_state=view_state,
        layers=[comp_layer, subject_layer],
        tooltip=tooltip,
        map_style=None,
    )


st.set_page_config(page_title="UrbanScape Property Evaluator", layout="wide")

st.title("UrbanScape Property Evaluator")
st.caption("Property input → model valuation → comparable property confidence → map")

df = load_data()

with st.sidebar:
    st.header("Property Input")

    address = st.text_input(
        "Property address *",
        placeholder="Example: 7727 Brownsville Ave, Dallas, TX",
    )

    segment_override = st.selectbox(
        "Property type",
        ["Auto-detect", "residential", "multifamily", "commercial", "land", "industrial"],
    )

    lot_size = st.number_input("Lot size sqft optional", min_value=0.0, value=0.0)
    building_size = st.number_input("Building size sqft optional", min_value=0.0, value=0.0)
    beds = st.number_input("beds optional", min_value=0.0, value=0.0)
    baths = st.number_input("baths optional", min_value=0.0, value=0.0)
    asking_price = st.number_input("Asking price optional", min_value=0.0, value=0.0)

    k = st.slider("Number of comps", 5, 25, 10)

    run = st.button("Evaluate Property")

if not run:
    st.info("Enter a property address to begin.")
    st.stop()

if not address:
    st.error("Property address is required.")
    st.stop()

geo = geocode_address(address)

if geo is None:
    st.error("Could not geocode address. Try adding city/state, e.g. Dallas, TX.")
    st.stop()

nearest = nearest_property(df, geo["lat"], geo["lon"])
detected_segment = detect_segment(nearest["sptd_desc"])
segment = detected_segment if segment_override == "Auto-detect" else segment_override

if segment is None:
    st.error("Could not detect property segment. Choose a property type manually.")
    st.stop()

if not (MODEL_BASE / segment).exists():
    st.error(f"No trained model exists for segment: {segment}")
    st.stop()

input_row = apply_user_inputs(
    nearest,
    lot_size=lot_size,
    building_size=building_size,
    beds=beds,
    baths=baths,
)

features, models = load_model_bundle(segment)
X = prepare_features(input_row, features)

pred_low = float(models["low"].predict(X)[0])
pred_mid = float(models["mid"].predict(X)[0])
pred_high = float(models["high"].predict(X)[0])

comps, usable_features = find_comps(df, input_row, segment, k=k)

comp_median = comps[TARGET].median()
avg_distance = comps["similarity_distance"].mean() if "similarity_distance" in comps.columns else 5

confidence = confidence_score(
    pred_mid=pred_mid,
    comp_median=comp_median,
    asking_price=asking_price,
    avg_distance=avg_distance,
)

agreement = agreement_label(pred_mid, comp_median)

st.subheader("Address Match")

c1, c2, c3 = st.columns(3)

c1.metric("Matched Address", geo["matched_address"])
c2.metric("Latitude", round(geo["lat"], 6))
c3.metric("Longitude", round(geo["lon"], 6))

st.subheader("Nearest Dataset Property Used")

p1, p2, p3, p4 = st.columns(4)

p1.metric("Nearest Account", str(nearest.get("account_num", "N/A")))
p2.metric("Dataset Type", str(nearest.get("sptd_desc", "N/A")))
p3.metric("Model Segment", segment.title())
p4.metric("Dataset Appraised Value", money(nearest.get(TARGET)))

st.subheader("Valuation")

v1, v2, v3, v4 = st.columns(4)

v1.metric("Model Low / 5%", money(pred_low))
v2.metric("Model Median / 50%", money(pred_mid))
v3.metric("Model High / 95%", money(pred_high))
v4.metric("Comp Median", money(comp_median))

st.subheader("Asking Price Comparison")

if asking_price and asking_price > 0:
    delta = pred_mid - asking_price

    a1, a2, a3 = st.columns(3)

    a1.metric("Asking Price", money(asking_price))
    a2.metric("Model vs Asking", money(delta))
    a3.metric("Deal Signal", "Undervalued" if delta > 0 else "Overpriced")

    if delta > 0:
        st.success(f"Asking price is below model median by {money(delta)}.")
    else:
        st.warning(f"Asking price is above model median by {money(abs(delta))}.")
else:
    st.info("No asking price entered.")

st.subheader("Comp Confidence")

cc1, cc2, cc3 = st.columns(3)

cc1.metric("Comp Agreement", agreement)
cc2.metric("Confidence", f"{confidence}%")
cc3.metric("Model vs Comp Median", money(pred_mid - comp_median))

st.markdown(
    f"""
Model median: **{money(pred_mid)}**  
Nearest comp median: **{money(comp_median)}**  
Model range: **{money(pred_low)} – {money(pred_high)}**  
Comp agreement: **{agreement}**  
Confidence: **{confidence}%**
"""
)

st.subheader("Geospatial Map of Comparable Properties")

deck = make_map(comps, nearest, geo)
st.pydeck_chart(deck, use_container_width=True)

st.caption("Red = subject property / nearest matched parcel. Blue = comparable properties.")

st.subheader("Similar Comparable Properties")

show_cols = [
    "account_num",
    "sptd_desc",
    "tot_val",
    "land_val",
    "impr_val",
    "shape_area",
    "front_dim",
    "depth_dim",
    "zoning",
    "zoning_score",
    "lat",
    "lon",
    "similarity_distance",
]

st.dataframe(
    comps[[c for c in show_cols if c in comps.columns]],
    use_container_width=True,
)

st.subheader("Inputs Used")

st.json(
    {
        "address": address,
        "matched_address": geo["matched_address"],
        "lot_size_sqft": lot_size,
        "building_size_sqft": building_size,
        "beds": beds,
        "baths": baths,
        "asking_price": asking_price,
        "model_segment": segment,
        "comp_features_used": usable_features,
    }
)