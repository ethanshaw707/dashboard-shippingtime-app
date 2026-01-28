import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import polars as pl
import pydeck as pdk
from pathlib import Path
from itertools import combinations


DATASETS = {
    "Small": "complete_shipping_data_size_s.csv",
    "Medium": "complete_shipping_data_size_m.csv",
    "Large": "complete_shipping_data_size_l.csv",
}
EXCLUDED_CITIES = {
    "Accord",
    "Acton",
    "Ada",
    "Adams Run",
    "Adamsville",
    "Adirondack",
    "Adrian",
    "Afton",
    "Ahwahnee",
    "Alamo",
    "Alapaha",
    "Alba",
}
STATE_COUNTS = {
    "CA": 832,
    "TX": 759,
    "FL": 573,
    "NY": 555,
    "PA": 440,
    "NC": 400,
    "OH": 386,
    "GA": 357,
    "IL": 334,
    "VA": 317,
    "NJ": 309,
    "MI": 300,
    "CO": 268,
    "MA": 266,
    "TN": 266,
    "WI": 230,
    "IN": 211,
    "WA": 210,
    "AZ": 207,
    "MO": 205,
    "MD": 200,
    "MN": 180,
    "OR": 154,
    "SC": 150,
    "CT": 132,
    "OK": 132,
    "AL": 129,
    "KY": 128,
    "LA": 102,
    "IA": 96,
    "ID": 96,
    "AR": 93,
    "KS": 85,
    "UT": 82,
    "NH": 75,
    "WV": 75,
    "ME": 72,
    "NV": 70,
    "NE": 62,
    "MS": 57,
    "NM": 53,
    "MT": 50,
    "ND": 49,
    "RI": 47,
    "DE": 44,
    "VT": 32,
    "WY": 30,
    "AK": 29,
    "HI": 21,
    "SD": 21,
    "DC": 10,
    "AE": 2,
    "AP": 2,
}
V3_CITY_COUNTS = {
    "brooklyn, ny": 39,
    "phoenix, az": 37,
    "houston, tx": 38,
    "san antonio, tx": 33,
    "jacksonville, fl": 21,
    "chicago, il": 28,
    "dallas, tx": 17,
    "wilmington, de": 7,
    "wilmington, nc": 8,
    "austin, tx": 26,
    "canton, oh": 4,
    "canton, ga": 15,
    "charlotte, nc": 26,
    "new york, ny": 25,
    "atlanta, ga": 24,
    "franklin, tn": 13,
    "lexington, ky": 11,
    "orlando, fl": 24,
    "rochester, ny": 14,
    "las vegas, nv": 23,
    "louisville, ky": 19,
    "marietta, ga": 19,
    "colorado springs, co": 22,
    "columbus, oh": 9,
    "marble falls, tx": 22,
    "tampa, fl": 22,
    "san diego, ca": 22,
    "columbia, sc": 3,
    "columbia, mo": 13,
    "greenville, sc": 4,
    "lebanon, tn": 8,
    "salem, or": 11,
    "cleveland, oh": 2,
    "cleveland, tn": 11,
    "denver, co": 17,
    "omaha, ne": 17,
    "portland, or": 17,
    "raleigh, nc": 19,
    "richmond, va": 7,
    "jackson, ms": 1,
    "albany, ny": 11,
    "fort worth, tx": 18,
    "oklahoma city, ok": 18,
    "chesapeake, va": 17,
    "miami, fl": 17,
    "milford, ct": 4,
    "pittsburgh, pa": 17,
    "san francisco, ca": 17,
    "san jose, ca": 17,
    "tucson, az": 17,
    "virginia beach, va": 17,
    "albuquerque, nm": 16,
    "gilbert, az": 15,
    "springfield, il": 5,
    "alexandria, va": 12,
    "kansas city, mo": 15,
    "knoxville, tn": 14,
    "lakewood, co": 6,
    "nashville, tn": 15,
    "saint louis, mo": 15,
    "seattle, wa": 15,
    "arlington, tx": 5,
    "aurora, co": 9,
    "boise, id": 14,
    "indianapolis, in": 14,
    "naples, fl": 13,
    "reno, nv": 14,
    "riverside, ca": 13,
    "bismarck, nd": 13,
    "fresno, tx": 1,
    "fresno, ca": 12,
    "lincoln, ri": 1,
    "lincoln, ne": 8,
    "los angeles, ca": 13,
    "mansfield, ma": 2,
    "milwaukee, wi": 13,
    "madison, in": 1,
    "mesa, az": 13,
    "monroe, nc": 4,
    "philadelphia, pa": 13,
    "staten island, ny": 13,
    "washington, dc": 10,
    "baltimore, md": 11,
    "decatur, il": 2,
    "durham, nc": 10,
    "fayetteville, ar": 4,
    "glendale, az": 7,
    "midland, ga": 6,
    "scottsdale, az": 12,
    "woodstock, ga": 6,
    "chandler, az": 8,
    "cincinnati, oh": 11,
    "covington, la": 3,
    "fort myers, fl": 11,
    "greensboro, nc": 10,
    "henderson, tx": 2,
    "henderson, nv": 8,
    "hamilton, mi": 1,
    "lewisburg, pa": 10,
    "littleton, co": 10,
    "mount pleasant, sc": 7,
    "pasadena, md": 3,
    "plano, tx": 11,
    "sanford, nc": 4,
    "wayne, me": 2,
    "ashland, pa": 2,
    "burlington, wi": 1,
    "centerville, ia": 1,
    "clayton, nc": 4,
    "encinitas, ca": 10,
    "fort collins, co": 10,
    "golden, co": 9,
    "hudson, nc": 2,
    "katy, tx": 10,
    "lancaster, sc": 1,
    "newark, nj": 3,
    "parker, co": 9,
    "plymouth, ma": 1,
    "rancho cucamonga, ca": 10,
    "rockville, md": 8,
    "sacramento, ca": 10,
    "tallahassee, fl": 10,
    "the villages, fl": 10,
    "westfield, in": 6,
    "waterloo, ia": 1,
    "west chester, pa": 7,
    "bend, or": 10,
    "amarillo, tx": 9,
    "auburn, ma": 2,
    "bakersfield, ca": 9,
    "boca raton, fl": 9,
    "benton, la": 4,
    "bradenton, fl": 9,
    "dayton, oh": 7,
    "duluth, mn": 7,
    "englewood, co": 6,
    "fort wayne, in": 9,
    "frederick, md": 8,
    "fredericksburg, va": 7,
    "gainesville, fl": 4,
    "grand rapids, mi": 9,
    "green bay, wi": 9,
    "morgantown, wv": 8,
    "pensacola, fl": 9,
    "princeton, ma": 1,
    "peoria, az": 6,
    "saint paul, mn": 9,
    "savannah, ga": 8,
    "seaford, ny": 5,
    "seymour, in": 3,
    "spokane, wa": 9,
    "windsor, co": 3,
    "winston-salem, nc": 0,
    "worcester, ma": 9,
    "york, pa": 5,
    "antioch, tn": 2,
    "athens, ga": 3,
    "corona, ca": 8,
    "fairfield, ca": 3,
    "germantown, md": 6,
    "hillsboro, or": 4,
    "hope mills, nc": 8,
    "hendersonville, nc": 4,
    "holly springs, ms": 2,
    "homer, ak": 5,
    "laurel, md": 6,
    "long beach, ca": 7,
    "manteca, ca": 8,
    "maryville, tn": 7,
    "magnolia, ms": 2,
    "memphis, tn": 8,
    "milton, ma": 1,
    "montgomery, tx": 5,
    "myrtle beach, sc": 8,
    "new market, al": 4,
    "ocala, fl": 8,
    "olathe, ks": 8,
    "oxford, mi": 4,
    "palm beach gardens, fl": 8,
    "roseville, ca": 7,
    "roswell, ga": 7,
    "rockwall, tx": 8,
    "spring, tx": 8,
    "salisbury, ma": 2,
    "santa rosa, ca": 8,
    "sarasota, fl": 8,
    "vancouver, wa": 8,
    "westminster, md": 5,
    "arvada, co": 7,
    "anchorage, ak": 7,
    "birmingham, al": 7,
    "bozeman, mt": 7,
    "conroe, tx": 7,
    "carmel valley, ca": 7,
    "castle rock, co": 7,
    "cedar rapids, ia": 7,
    "cooperstown, ny": 7,
    "dayton, oh": 7,
    "deland, fl": 7,
    "duluth, mn": 7,
    "fort mill, sc": 7,
    "fargo, nd": 7,
    "fredericksburg, va": 7,
    "glen allen, va": 7,
    "glendale, az": 7,
    "grand junction, co": 7,
    "highlands ranch, co": 7,
    "joliet, il": 7,
    "kerrville, tx": 7,
    "longmont, co": 7,
    "long beach, ca": 7,
    "maryville, tn": 7,
    "mount pleasant, sc": 7,
    "minneapolis, mn": 7,
    "naperville, il": 7,
    "neosho, mo": 7,
    "puyallup, wa": 7,
    "queen creek, az": 7,
    "richmond, va": 7,
    "roseville, ca": 7,
    "roswell, ga": 7,
    "saline, mi": 7,
    "sevierville, tn": 7,
    "sterling heights, mi": 7,
    "thornton, co": 7,
    "ventura, ca": 7,
    "wilmington, ma": 7,
    "west chester, pa": 7,
    "clermont, fl": 7,
    "abilene, tx": 6,
    "brick, nj": 6,
    "bentonville, ar": 6,
    "bogart, ga": 6,
    "boston, ma": 6,
    "buffalo, ny": 6,
    "cary, nc": 6,
    "dallas, ga": 6,
    "doylestown, pa": 6,
    "englewood, co": 6,
    "eugene, or": 6,
    "edmond, ok": 6,
    "forney, tx": 6,
    "fuquay varina, nc": 6,
    "germantown, md": 6,
    "glendora, ca": 6,
    "henrico, va": 6,
    "hockessin, de": 6,
    "holly springs, nc": 6,
    "irvine, ca": 6,
    "lakeland, fl": 6,
    "lawrence, ks": 6,
    "league city, tx": 6,
    "mansfield, tx": 6,
    "madison, wi": 6,
    "mckinney, tx": 6,
    "mentor, oh": 6,
    "mount airy, md": 6,
    "norfolk, va": 6,
    "nampa, id": 6,
    "noblesville, in": 6,
    "orland park, il": 6,
    "palm harbor, fl": 6,
    "petaluma, ca": 6,
    "placentia, ca": 6,
    "prescott, az": 6,
    "richardson, tx": 6,
    "roanoke, va": 6,
    "richmond, tx": 6,
    "sanford, me": 6,
    "seminole, fl": 6,
    "sheridan, wy": 6,
    "sparks, nv": 6,
    "traverse city, mi": 6,
    "tulsa, ok": 6,
    "wake forest, nc": 6,
    "woodstock, ga": 6,
    "waterloo, sc": 6,
    "waukesha, wi": 6,
    "wichita, ks": 6,
    "winnsboro, tx": 6,
    "woodbury, mn": 6,
    "cumming, ga": 6,
    "muskegon, mi": 6,
    "arlington, tx": 5,
    "akron, oh": 5,
    "albany, or": 5,
    "alpharetta, ga": 5,
    "asheville, nc": 5,
    "bixby, ok": 5,
    "boulder, co": 5,
    "bethlehem, pa": 5,
    "brenham, tx": 5,
    "broken bow, ok": 5,
    "cape coral, fl": 5,
    "chula vista, ca": 5,
    "crawfordville, fl": 5,
    "cypress, tx": 5,
    "clearwater, fl": 5,
    "columbiana, al": 5,
    "columbus, ga": 5,
    "commack, ny": 5,
    "dublin, ca": 5,
    "decatur, ga": 5,
    "deltona, fl": 5,
    "derry, nh": 5,
    "douglassville, pa": 5,
    "easton, pa": 5,
    "el paso, tx": 5,
    "escondido, ca": 5,
    "fairport, ny": 5,
    "flemington, nj": 5,
    "florence, sc": 5,
    "fort lauderdale, fl": 5,
    "frisco, tx": 5,
    "goodyear, az": 5,
    "gays mills, wi": 5,
    "grafton, wi": 5,
    "green cove springs, fl": 5,
    "greensburg, pa": 5,
    "homer, ak": 5,
    "honolulu, hi": 5,
    "humble, tx": 5,
    "huntington, ny": 5,
    "idaho falls, id": 5,
    "jackson, tn": 5,
    "jacksonville, nc": 5,
    "jenkintown, pa": 5,
    "johnson city, tn": 5,
    "kingwood, tx": 5,
    "lansing, mi": 5,
    "lapeer, mi": 5,
    "lawrenceville, ga": 5,
    "lexington, nc": 5,
    "lubbock, tx": 5,
    "medina, oh": 5,
    "macomb, mi": 5,
    "magnolia, tx": 5,
    "martinsburg, wv": 5,
    "meridian, id": 5,
    "mobile, al": 5,
    "montgomery, tx": 5,
    "morrison, co": 5,
    "napa, ca": 5,
    "new braunfels, tx": 5,
    "new lenox, il": 5,
    "new orleans, la": 5,
    "newark, de": 5,
    "oakland, ca": 5,
    "owasso, ok": 5,
    "overland park, ks": 5,
    "panama city, fl": 5,
    "panama city beach, fl": 5,
    "patchogue, ny": 5,
    "port clinton, oh": 5,
    "paducah, ky": 5,
    "palmetto, fl": 5,
    "pearland, tx": 5,
    "pella, ia": 5,
    "pittsford, ny": 5,
    "port orange, fl": 5,
    "ravenna, oh": 5,
    "roaming shores, oh": 5,
    "richmond hill, ga": 5,
    "riverview, fl": 5,
    "rochester, mn": 5,
    "round rock, tx": 5,
    "saint petersburg, fl": 5,
    "san leandro, ca": 5,
    "southington, ct": 5,
    "spartanburg, sc": 5,
    "surprise, az": 5,
    "santa barbara, ca": 5,
    "seaford, ny": 5,
    "shelton, ct": 5,
    "spring branch, tx": 5,
    "springfield, il": 5,
    "st. louis, mo": 5,
    "stroudsburg, pa": 5,
    "sunapee, nh": 5,
    "swedesboro, nj": 5,
    "syracuse, ny": 5,
    "tomball, tx": 5,
    "topeka, ks": 5,
    "tuscaloosa, al": 5,
    "terre haute, in": 5,
    "tinley park, il": 5,
    "toms river, nj": 5,
    "tulare, ca": 5,
    "vienna, va": 5,
    "walnut creek, ca": 5,
    "wayne, nj": 5,
    "waynesboro, va": 5,
    "west sacramento, ca": 5,
    "westminster, md": 5,
    "yuba city, ca": 5,
    "zachary, la": 5,
    "buford, ga": 5,
    "modesto, ca": 5,
    "winter park, fl": 5,
}
DESTINATION_WEIGHTS = {}
DEFAULT_DEST_WEIGHT = 1.0


@st.cache_data
def load_data(path: str, drop_missing: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    if "ToCity" in df.columns:
        df = df[~df["ToCity"].isin(EXCLUDED_CITIES)]
    for col in ["Cost", "ShippingTimeDays"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if drop_missing:
        df = df.dropna(subset=["Cost", "ShippingTimeDays"])
    df["Destination"] = df["ToCity"].str.strip() + ", " + df["ToState"].str.strip()
    return df


def minmax(series: pd.Series) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total == 0:
        return float(np.mean(values)) if len(values) else float("nan")
    return float(np.sum(values * weights) / total)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    total = weights_sorted.sum()
    if total == 0:
        return float(np.median(values_sorted))
    cum_weights = np.cumsum(weights_sorted)
    idx = np.searchsorted(cum_weights, total / 2.0)
    return float(values_sorted[min(idx, len(values_sorted) - 1)])


def nearest_bin(value: float, bins: np.ndarray) -> float:
    if bins.size == 0 or not np.isfinite(value):
        return float("nan")
    idx = int(np.abs(bins - value).argmin())
    return float(bins[idx])


def initial_destination_weights(df: pd.DataFrame, state_counts: dict) -> dict:
    total = sum(state_counts.values())
    if total <= 0 or df.empty:
        return {}
    state_pct = {state: count / total * 100.0 for state, count in state_counts.items()}
    city_counts = df.groupby("ToState")["ToCity"].nunique()
    weights = {}
    unique_rows = df[["ToCity", "ToState"]].drop_duplicates()
    for _, row in unique_rows.iterrows():
        state = row["ToState"]
        city = row["ToCity"]
        dest = f"{city}, {state}"
        if state in state_pct and city_counts.get(state, 0) > 0:
            weights[dest] = state_pct[state] / city_counts[state]
        else:
            weights[dest] = DEFAULT_DEST_WEIGHT
    return weights


def initial_destination_weights_by_city(df: pd.DataFrame, city_counts: dict) -> dict:
    if df.empty:
        return {}
    cities_in_df = {
        f"{str(row['ToCity']).strip().casefold()}, {str(row['ToState']).strip().casefold()}"
        for _, row in df[["ToCity", "ToState"]].dropna().drop_duplicates().iterrows()
    }
    city_counts_filtered = {city: count for city, count in city_counts.items() if city in cities_in_df}
    total = sum(city_counts_filtered.values())
    if total <= 0:
        return {}
    city_pct = {city: count / total * 100.0 for city, count in city_counts_filtered.items()}
    city_dest_counts = df.groupby(["ToCity", "ToState"])["Destination"].nunique()
    weights = {}
    unique_rows = df[["ToCity", "ToState", "Destination"]].drop_duplicates()
    for _, row in unique_rows.iterrows():
        city = str(row["ToCity"]).strip()
        state = str(row["ToState"]).strip()
        dest = str(row["Destination"]).strip()
        key = f"{city.casefold()}, {state.casefold()}"
        if key in city_pct and city_dest_counts.get((city, state), 0) > 0:
            weights[dest] = city_pct[key] / city_dest_counts[(city, state)]
        else:
            weights[dest] = DEFAULT_DEST_WEIGHT
    return weights


def add_weighted_score(df: pd.DataFrame, cost_weight: float) -> pd.DataFrame:
    df = df.copy()
    df["CostNorm"] = minmax(df["Cost"])
    df["TimeNorm"] = minmax(df["ShippingTimeDays"])
    df["WeightedScore"] = cost_weight * df["CostNorm"] + (1 - cost_weight) * df["TimeNorm"]
    return df


def apply_filters(df: pd.DataFrame, cities, origins) -> pd.DataFrame:
    mask = df["ToCity"].isin(cities) & df["FromAddress"].isin(origins)
    return df[mask].copy()

def destination_weights(destinations, weight_map: dict) -> dict:
    weights = {dest: float(weight_map.get(dest, DEFAULT_DEST_WEIGHT)) for dest in destinations}
    total = sum(weights.values())
    if total == 0:
        equal = 1.0 / max(len(destinations), 1)
        return {dest: equal for dest in destinations}
    return {dest: val / total for dest, val in weights.items()}


def _dict_to_items(weight_map: dict) -> tuple:
    return tuple(sorted(weight_map.items()))

def render_top_combos(
    df: pd.DataFrame,
    origin_list: list,
    dest_in_view: list,
    dest_weights: dict,
    build_cost_weight: float,
    avg_cost: float,
    avg_time: float,
    major_weight_map: dict,
    key_prefix: str,
    show_day_percentages: bool = False,
    max_k: int = 10,
    baseline_origin: str | None = None,
) -> None:
    st.markdown("Recommended starting two (synergy pairs)")
    show_key = f"{key_prefix}_show_top_combos"
    if show_key not in st.session_state:
        st.session_state[show_key] = False
    if st.button("Clear combo cache", key=f"{key_prefix}_clear_combo_cache"):
        compute_pair_df_cached.clear()
        compute_top_k_combos_cached.clear()
        compute_top_k_combos_with_required_cached.clear()
        st.caption("Combo cache cleared.")
    if st.button("Compute top combos", key=f"{key_prefix}_compute_top_combos"):
        st.session_state[show_key] = True

    if not st.session_state[show_key]:
        st.caption('Top combos are hidden until you click "Compute top combos".')
        return

    dest_weights_items = _dict_to_items(dest_weights)
    cost_mat, time_mat = build_origin_destination_matrices(
        df,
        tuple(origin_list),
        tuple(dest_in_view),
    )
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_in_view], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_in_view), dtype=float)

    def combo_avg_cost_time(combo_indices):
        best_cost = np.nanmin(cost_mat[combo_indices, :], axis=0)
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        if not valid_mask.any():
            return float("nan"), float("nan")
        weights = weights_vec[valid_mask]
        if weights.sum() == 0:
            weights = np.ones(len(weights), dtype=float)
        avg_cost_local = float(np.sum(best_cost[valid_mask] * weights) / weights.sum())
        avg_time_local = float(np.sum(best_time[valid_mask] * weights) / weights.sum())
        return avg_cost_local, avg_time_local

    def combo_weighted_total(combo_indices):
        avg_cost_local, avg_time_local = combo_avg_cost_time(combo_indices)
        if not np.isfinite(avg_cost_local) or not np.isfinite(avg_time_local):
            return float("nan"), avg_cost_local, avg_time_local
        weighted_total = build_cost_weight * avg_cost_local + (1 - build_cost_weight) * avg_time_local
        return weighted_total, avg_cost_local, avg_time_local

    def best_combo_indices(k_size: int):
        best_combo = None
        best_weighted = float("inf")
        for combo in combinations(range(len(origin_list)), k_size):
            weighted_total, _, _ = combo_weighted_total(list(combo))
            if np.isfinite(weighted_total) and weighted_total < best_weighted:
                best_weighted = weighted_total
                best_combo = list(combo)
        return best_combo

    def combo_time_improvements(combo_indices, prev_indices):
        if not prev_indices:
            return 0, 0
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        prev_time = np.nanmin(time_mat[prev_indices, :], axis=0)
        valid = ~np.isnan(best_time) & ~np.isnan(prev_time)
        if not valid.any():
            return 0, 0
        three_to_two = np.sum((prev_time[valid] > 2.0) & (best_time[valid] <= 2.0) & (best_time[valid] > 1.0))
        two_to_one = np.sum((prev_time[valid] > 1.0) & (best_time[valid] <= 1.0))
        return int(three_to_two), int(two_to_one)

    def combo_improved_cities(prev_indices, combo_indices):
        if not prev_indices:
            return []
        prev_time = np.nanmin(time_mat[prev_indices, :], axis=0)
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        improved = []
        for i, (p, b) in enumerate(zip(prev_time, best_time)):
            if np.isfinite(p) and np.isfinite(b) and b < p:
                improved.append(dest_in_view[i])
        return improved

    def combo_not_covered(combo_indices, threshold: float):
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        not_covered = []
        for i, t in enumerate(best_time):
            if np.isfinite(t) and t > threshold:
                not_covered.append(dest_in_view[i])
        return not_covered

    def format_not_covered(destinations):
        if not destinations:
            return "None"
        parts = []
        for dest in destinations:
            weight = dest_weights.get(dest, 0.0) * 100.0
            parts.append(f"{dest} ({weight:.1f}%)")
        return ", ".join(parts)

    def combo_day_percentages(combo_indices):
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        valid = np.isfinite(best_time)
        total = int(np.sum(valid))
        if total == 0:
            return 0.0, 0.0
        one_day = float(np.sum(best_time[valid] <= 1.0))
        two_day = float(np.sum(best_time[valid] <= 2.0))
        return one_day / total * 100.0, two_day / total * 100.0

    def expand_from_best(prev_list, k_size):
        if not prev_list:
            return []
        best_name = prev_list[0][0]
        best_origins = best_name.split(" + ")
        if len(best_origins) != k_size - 1:
            return []
        results = []
        for origin in origin_list:
            if origin in best_origins:
                continue
            combo = best_origins + [origin]
            combo_indices = [origin_list.index(name) for name in combo]
            weighted_total, _, _ = combo_weighted_total(combo_indices)
            if np.isfinite(weighted_total):
                results.append((" + ".join(combo), weighted_total))
        results.sort(key=lambda x: x[1])
        if len(results) <= 5:
            return results
        cutoff = results[4][1]
        return [row for row in results if row[1] <= cutoff]

    pair_df = compute_pair_df_cached(
        df,
        tuple(origin_list),
        tuple(dest_in_view),
        build_cost_weight,
        _dict_to_items(dest_weights),
        _dict_to_items(major_weight_map),
    )

    if not pair_df.empty:
        best_value = float(pair_df["WeightedTotal"].iloc[0])
        tied_pairs = pair_df[pair_df["WeightedTotal"] == best_value]["Pair"].tolist()
        if len(tied_pairs) > 5:
            top_pairs = pair_df[pair_df["WeightedTotal"] == best_value][["Pair", "WeightedTotal"]].copy()
        else:
            top_pairs = pair_df.head(5)[["Pair", "WeightedTotal"]].copy()
        pair_labels = [f"{row['Pair']} ({row['WeightedTotal']:.2f})" for _, row in top_pairs.iterrows()]
        selected_pair = st.selectbox(
            "Top 5 pairs (weighted)",
            pair_labels,
            index=0,
            key=f"{key_prefix}_top5_pairs",
        )
        selected_pair_name = selected_pair.rsplit(" (", 1)[0]
        selected_row = pair_df[pair_df["Pair"] == selected_pair_name].iloc[0]
        combo_indices = [origin_list.index(name) for name in selected_pair_name.split(" + ")]
        combo_avg_cost, combo_avg_time = combo_avg_cost_time(combo_indices)
        cost_delta = avg_cost - combo_avg_cost if np.isfinite(combo_avg_cost) else float("nan")
        time_delta = avg_time - combo_avg_time if np.isfinite(combo_avg_time) else float("nan")
        baseline_indices = None
        if baseline_origin and baseline_origin in origin_list:
            baseline_indices = [origin_list.index(baseline_origin)]
        prev_combo = baseline_indices or best_combo_indices(1)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - combo_avg_cost
            prev_time_delta = prev_time - combo_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(combo_indices, prev_combo or [])
        st.caption(
            f"Weighted total: {selected_row['WeightedTotal']:.2f} - Avg cost: {combo_avg_cost:.2f} "
            f"(delta {cost_delta:.2f}) - Avg time: {combo_avg_time:.2f} (delta {time_delta:.2f}) - "
            f"Major coverage: {selected_row['MajorCoveragePct']:.1f}%"
        )
        st.caption(f"Delta vs best 1 origin: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        pair_improved = combo_improved_cities(prev_combo or [], combo_indices)
        st.selectbox(
            "Improved destinations vs best 1 origin (pair)",
            pair_improved or ["None"],
            key=f"{key_prefix}_pair_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(combo_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(combo_indices, 1.0)
            not_two_day = combo_not_covered(combo_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate pairs.")

    st.markdown("Top 5 trios")
    trio_list = (
        compute_top_k_combos_cached(
            df,
            tuple(origin_list),
            tuple(dest_in_view),
            3,
            build_cost_weight,
            dest_weights_items,
        ) if len(origin_list) >= 3 else []
    )
    if trio_list:
        trio_labels = [f"{name} ({value:.2f})" for name, value in trio_list]
        selected_trio = st.selectbox(
            "Top 5 trios (weighted)",
            trio_labels,
            index=0,
            key=f"{key_prefix}_top5_trios",
        )
        trio_name = selected_trio.rsplit(" (", 1)[0]
        trio_indices = [origin_list.index(name) for name in trio_name.split(" + ")]
        trio_avg_cost, trio_avg_time = combo_avg_cost_time(trio_indices)
        prev_combo = best_combo_indices(2)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - trio_avg_cost
            prev_time_delta = prev_time - trio_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(trio_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {trio_avg_cost:.2f} (delta {avg_cost - trio_avg_cost:.2f}) - "
            f"Avg time: {trio_avg_time:.2f} (delta {avg_time - trio_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 2 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        trio_improved = combo_improved_cities(prev_combo or [], trio_indices)
        st.selectbox(
            "Improved destinations vs best 2 origins (trio)",
            trio_improved or ["None"],
            key=f"{key_prefix}_trio_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(trio_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(trio_indices, 1.0)
            not_two_day = combo_not_covered(trio_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate trios.")

    st.markdown("Top 5 quads")
    quad_list = (
        compute_top_k_combos_cached(
            df,
            tuple(origin_list),
            tuple(dest_in_view),
            4,
            build_cost_weight,
            dest_weights_items,
        ) if len(origin_list) >= 4 else []
    )
    if quad_list:
        quad_labels = [f"{name} ({value:.2f})" for name, value in quad_list]
        selected_quad = st.selectbox(
            "Top 5 quads (weighted)",
            quad_labels,
            index=0,
            key=f"{key_prefix}_top5_quads",
        )
        quad_name = selected_quad.rsplit(" (", 1)[0]
        quad_indices = [origin_list.index(name) for name in quad_name.split(" + ")]
        quad_avg_cost, quad_avg_time = combo_avg_cost_time(quad_indices)
        prev_combo = best_combo_indices(3)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - quad_avg_cost
            prev_time_delta = prev_time - quad_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(quad_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {quad_avg_cost:.2f} (delta {avg_cost - quad_avg_cost:.2f}) - "
            f"Avg time: {quad_avg_time:.2f} (delta {avg_time - quad_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 3 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        quad_improved = combo_improved_cities(prev_combo or [], quad_indices)
        st.selectbox(
            "Improved destinations vs best 3 origins (quad)",
            quad_improved or ["None"],
            key=f"{key_prefix}_quad_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(quad_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(quad_indices, 1.0)
            not_two_day = combo_not_covered(quad_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate quads.")

    st.markdown("Top 5 (5 locations)")
    five_list = (
        compute_top_k_combos_cached(
            df,
            tuple(origin_list),
            tuple(dest_in_view),
            5,
            build_cost_weight,
            dest_weights_items,
        ) if len(origin_list) >= 5 else []
    )
    common_origins = []
    if five_list:
        five_labels = [f"{name} ({value:.2f})" for name, value in five_list]
        selected_five = st.selectbox(
            "Top 5 (5 locations) (weighted)",
            five_labels,
            index=0,
            key=f"{key_prefix}_top5_fives",
        )
        five_name = selected_five.rsplit(" (", 1)[0]
        five_indices = [origin_list.index(name) for name in five_name.split(" + ")]
        five_avg_cost, five_avg_time = combo_avg_cost_time(five_indices)
        prev_combo = best_combo_indices(4)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - five_avg_cost
            prev_time_delta = prev_time - five_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(five_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {five_avg_cost:.2f} (delta {avg_cost - five_avg_cost:.2f}) - "
            f"Avg time: {five_avg_time:.2f} (delta {avg_time - five_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 4 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        five_origin_sets = [set(name.split(" + ")) for name, _ in five_list]
        common_origins = sorted(set.intersection(*five_origin_sets)) if five_origin_sets else []
        if common_origins:
            st.caption(f"Auto-included origins for 6/7: {', '.join(common_origins)}")
        five_improved = combo_improved_cities(prev_combo or [], five_indices)
        st.selectbox(
            "Improved destinations vs best 4 origins (5 locations)",
            five_improved or ["None"],
            key=f"{key_prefix}_five_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(five_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(five_indices, 1.0)
            not_two_day = combo_not_covered(five_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 5-location combos.")

    st.markdown("Top 5 (6 locations)")
    required_origins = tuple(common_origins) if common_origins else tuple()
    six_list = (
        compute_top_k_combos_with_required_cached(
            df,
            tuple(origin_list),
            tuple(dest_in_view),
            required_origins,
            6,
            build_cost_weight,
            dest_weights_items,
        ) if len(origin_list) >= 6 else []
    )
    common_origins6 = []
    if six_list:
        six_labels = [f"{name} ({value:.2f})" for name, value in six_list]
        selected_six = st.selectbox(
            "Top 5 (6 locations) (weighted)",
            six_labels,
            index=0,
            key=f"{key_prefix}_top5_sixes",
        )
        six_name = selected_six.rsplit(" (", 1)[0]
        six_indices = [origin_list.index(name) for name in six_name.split(" + ")]
        six_avg_cost, six_avg_time = combo_avg_cost_time(six_indices)
        prev_combo = best_combo_indices(5)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - six_avg_cost
            prev_time_delta = prev_time - six_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(six_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {six_avg_cost:.2f} (delta {avg_cost - six_avg_cost:.2f}) - "
            f"Avg time: {six_avg_time:.2f} (delta {avg_time - six_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 5 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        six_origin_sets = [set(name.split(" + ")) for name, _ in six_list]
        common_origins6 = sorted(set.intersection(*six_origin_sets)) if six_origin_sets else []
        if common_origins6:
            st.caption(f"Auto-included origins for 7/8: {', '.join(common_origins6)}")
        six_improved = combo_improved_cities(prev_combo or [], six_indices)
        st.selectbox(
            "Improved destinations vs best 5 origins (6 locations)",
            six_improved or ["None"],
            key=f"{key_prefix}_six_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(six_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(six_indices, 1.0)
            not_two_day = combo_not_covered(six_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 6-location combos.")

    st.markdown("Top 5 (7 locations)")
    seven_list = (expand_from_best(six_list, 7) if len(origin_list) >= 7 else [])
    if seven_list:
        seven_labels = [f"{name} ({value:.2f})" for name, value in seven_list]
        selected_seven = st.selectbox(
            "Top 5 (7 locations) (weighted)",
            seven_labels,
            index=0,
            key=f"{key_prefix}_top5_sevens",
        )
        seven_name = selected_seven.rsplit(" (", 1)[0]
        seven_indices = [origin_list.index(name) for name in seven_name.split(" + ")]
        seven_avg_cost, seven_avg_time = combo_avg_cost_time(seven_indices)
        prev_combo = best_combo_indices(6)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - seven_avg_cost
            prev_time_delta = prev_time - seven_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(seven_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {seven_avg_cost:.2f} (delta {avg_cost - seven_avg_cost:.2f}) - "
            f"Avg time: {seven_avg_time:.2f} (delta {avg_time - seven_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 6 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        seven_improved = combo_improved_cities(prev_combo or [], seven_indices)
        st.selectbox(
            "Improved destinations vs best 6 origins (7 locations)",
            seven_improved or ["None"],
            key=f"{key_prefix}_seven_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(seven_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(seven_indices, 1.0)
            not_two_day = combo_not_covered(seven_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 7-location combos.")

    st.markdown("Top 5 (8 locations)")
    eight_list = (expand_from_best(seven_list, 8) if len(origin_list) >= 8 else [])
    if eight_list:
        eight_labels = [f"{name} ({value:.2f})" for name, value in eight_list]
        selected_eight = st.selectbox(
            "Top 5 (8 locations) (weighted)",
            eight_labels,
            index=0,
            key=f"{key_prefix}_top5_eights",
        )
        eight_name = selected_eight.rsplit(" (", 1)[0]
        eight_indices = [origin_list.index(name) for name in eight_name.split(" + ")]
        eight_avg_cost, eight_avg_time = combo_avg_cost_time(eight_indices)
        prev_combo = best_combo_indices(7)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - eight_avg_cost
            prev_time_delta = prev_time - eight_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(eight_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {eight_avg_cost:.2f} (delta {avg_cost - eight_avg_cost:.2f}) - "
            f"Avg time: {eight_avg_time:.2f} (delta {avg_time - eight_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 7 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        eight_improved = combo_improved_cities(prev_combo or [], eight_indices)
        st.selectbox(
            "Improved destinations vs best 7 origins (8 locations)",
            eight_improved or ["None"],
            key=f"{key_prefix}_eight_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(eight_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(eight_indices, 1.0)
            not_two_day = combo_not_covered(eight_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 8-location combos.")

    st.markdown("Top 5 (9 locations)")
    nine_list = (expand_from_best(eight_list, 9) if len(origin_list) >= 9 else [])
    if nine_list:
        nine_labels = [f"{name} ({value:.2f})" for name, value in nine_list]
        selected_nine = st.selectbox(
            "Top 5 (9 locations) (weighted)",
            nine_labels,
            index=0,
            key=f"{key_prefix}_top5_nines",
        )
        nine_name = selected_nine.rsplit(" (", 1)[0]
        nine_indices = [origin_list.index(name) for name in nine_name.split(" + ")]
        nine_avg_cost, nine_avg_time = combo_avg_cost_time(nine_indices)
        prev_combo = best_combo_indices(8)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - nine_avg_cost
            prev_time_delta = prev_time - nine_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(nine_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {nine_avg_cost:.2f} (delta {avg_cost - nine_avg_cost:.2f}) - "
            f"Avg time: {nine_avg_time:.2f} (delta {avg_time - nine_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 8 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        nine_improved = combo_improved_cities(prev_combo or [], nine_indices)
        st.selectbox(
            "Improved destinations vs best 8 origins (9 locations)",
            nine_improved or ["None"],
            key=f"{key_prefix}_nine_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2 = combo_day_percentages(nine_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
            not_one_day = combo_not_covered(nine_indices, 1.0)
            not_two_day = combo_not_covered(nine_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 9-location combos.")

    if max_k >= 10:
        st.markdown("Top 5 (10 locations)")
        ten_list = (expand_from_best(nine_list, 10) if len(origin_list) >= 10 else [])
        if ten_list:
            ten_labels = [f"{name} ({value:.2f})" for name, value in ten_list]
            selected_ten = st.selectbox(
                "Top 5 (10 locations) (weighted)",
                ten_labels,
                index=0,
                key=f"{key_prefix}_top5_tens",
            )
            ten_name = selected_ten.rsplit(" (", 1)[0]
            ten_indices = [origin_list.index(name) for name in ten_name.split(" + ")]
            ten_avg_cost, ten_avg_time = combo_avg_cost_time(ten_indices)
            prev_combo = best_combo_indices(9)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - ten_avg_cost
                prev_time_delta = prev_time - ten_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(ten_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {ten_avg_cost:.2f} (delta {avg_cost - ten_avg_cost:.2f}) - "
                f"Avg time: {ten_avg_time:.2f} (delta {avg_time - ten_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 9 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            ten_improved = combo_improved_cities(prev_combo or [], ten_indices)
            st.selectbox(
                "Improved destinations vs best 9 origins (10 locations)",
                ten_improved or ["None"],
                key=f"{key_prefix}_ten_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(ten_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(ten_indices, 1.0)
                not_two_day = combo_not_covered(ten_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 10-location combos.")

    if max_k >= 11:
        st.markdown("Top 5 (11 locations)")
        eleven_list = (expand_from_best(ten_list, 11) if len(origin_list) >= 11 else [])
        if eleven_list:
            eleven_labels = [f"{name} ({value:.2f})" for name, value in eleven_list]
            selected_eleven = st.selectbox(
                "Top 5 (11 locations) (weighted)",
                eleven_labels,
                index=0,
                key=f"{key_prefix}_top5_elevens",
            )
            eleven_name = selected_eleven.rsplit(" (", 1)[0]
            eleven_indices = [origin_list.index(name) for name in eleven_name.split(" + ")]
            eleven_avg_cost, eleven_avg_time = combo_avg_cost_time(eleven_indices)
            prev_combo = best_combo_indices(10)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - eleven_avg_cost
                prev_time_delta = prev_time - eleven_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(eleven_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {eleven_avg_cost:.2f} (delta {avg_cost - eleven_avg_cost:.2f}) - "
                f"Avg time: {eleven_avg_time:.2f} (delta {avg_time - eleven_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 10 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            eleven_improved = combo_improved_cities(prev_combo or [], eleven_indices)
            st.selectbox(
                "Improved destinations vs best 10 origins (11 locations)",
                eleven_improved or ["None"],
                key=f"{key_prefix}_eleven_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(eleven_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(eleven_indices, 1.0)
                not_two_day = combo_not_covered(eleven_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 11-location combos.")

    if max_k >= 12:
        st.markdown("Top 5 (12 locations)")
        twelve_list = (expand_from_best(eleven_list, 12) if len(origin_list) >= 12 else [])
        if twelve_list:
            twelve_labels = [f"{name} ({value:.2f})" for name, value in twelve_list]
            selected_twelve = st.selectbox(
                "Top 5 (12 locations) (weighted)",
                twelve_labels,
                index=0,
                key=f"{key_prefix}_top5_twelves",
            )
            twelve_name = selected_twelve.rsplit(" (", 1)[0]
            twelve_indices = [origin_list.index(name) for name in twelve_name.split(" + ")]
            twelve_avg_cost, twelve_avg_time = combo_avg_cost_time(twelve_indices)
            prev_combo = best_combo_indices(11)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - twelve_avg_cost
                prev_time_delta = prev_time - twelve_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(twelve_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {twelve_avg_cost:.2f} (delta {avg_cost - twelve_avg_cost:.2f}) - "
                f"Avg time: {twelve_avg_time:.2f} (delta {avg_time - twelve_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 11 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            twelve_improved = combo_improved_cities(prev_combo or [], twelve_indices)
            st.selectbox(
                "Improved destinations vs best 11 origins (12 locations)",
                twelve_improved or ["None"],
                key=f"{key_prefix}_twelve_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(twelve_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(twelve_indices, 1.0)
                not_two_day = combo_not_covered(twelve_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 12-location combos.")

    if max_k >= 13:
        st.markdown("Top 5 (13 locations)")
        thirteen_list = (expand_from_best(twelve_list, 13) if len(origin_list) >= 13 else [])
        if thirteen_list:
            thirteen_labels = [f"{name} ({value:.2f})" for name, value in thirteen_list]
            selected_thirteen = st.selectbox(
                "Top 5 (13 locations) (weighted)",
                thirteen_labels,
                index=0,
                key=f"{key_prefix}_top5_thirteens",
            )
            thirteen_name = selected_thirteen.rsplit(" (", 1)[0]
            thirteen_indices = [origin_list.index(name) for name in thirteen_name.split(" + ")]
            thirteen_avg_cost, thirteen_avg_time = combo_avg_cost_time(thirteen_indices)
            prev_combo = best_combo_indices(12)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - thirteen_avg_cost
                prev_time_delta = prev_time - thirteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(thirteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {thirteen_avg_cost:.2f} (delta {avg_cost - thirteen_avg_cost:.2f}) - "
                f"Avg time: {thirteen_avg_time:.2f} (delta {avg_time - thirteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 12 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            thirteen_improved = combo_improved_cities(prev_combo or [], thirteen_indices)
            st.selectbox(
                "Improved destinations vs best 12 origins (13 locations)",
                thirteen_improved or ["None"],
                key=f"{key_prefix}_thirteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(thirteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(thirteen_indices, 1.0)
                not_two_day = combo_not_covered(thirteen_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 13-location combos.")

    if max_k >= 14:
        st.markdown("Top 5 (14 locations)")
        fourteen_list = (expand_from_best(thirteen_list, 14) if len(origin_list) >= 14 else [])
        if fourteen_list:
            fourteen_labels = [f"{name} ({value:.2f})" for name, value in fourteen_list]
            selected_fourteen = st.selectbox(
                "Top 5 (14 locations) (weighted)",
                fourteen_labels,
                index=0,
                key=f"{key_prefix}_top5_fourteens",
            )
            fourteen_name = selected_fourteen.rsplit(" (", 1)[0]
            fourteen_indices = [origin_list.index(name) for name in fourteen_name.split(" + ")]
            fourteen_avg_cost, fourteen_avg_time = combo_avg_cost_time(fourteen_indices)
            prev_combo = best_combo_indices(13)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - fourteen_avg_cost
                prev_time_delta = prev_time - fourteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(fourteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {fourteen_avg_cost:.2f} (delta {avg_cost - fourteen_avg_cost:.2f}) - "
                f"Avg time: {fourteen_avg_time:.2f} (delta {avg_time - fourteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 13 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            fourteen_improved = combo_improved_cities(prev_combo or [], fourteen_indices)
            st.selectbox(
                "Improved destinations vs best 13 origins (14 locations)",
                fourteen_improved or ["None"],
                key=f"{key_prefix}_fourteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(fourteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(fourteen_indices, 1.0)
                not_two_day = combo_not_covered(fourteen_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 14-location combos.")

    if max_k >= 15:
        st.markdown("Top 5 (15 locations)")
        fifteen_list = (expand_from_best(fourteen_list, 15) if len(origin_list) >= 15 else [])
        if fifteen_list:
            fifteen_labels = [f"{name} ({value:.2f})" for name, value in fifteen_list]
            selected_fifteen = st.selectbox(
                "Top 5 (15 locations) (weighted)",
                fifteen_labels,
                index=0,
                key=f"{key_prefix}_top5_fifteens",
            )
            fifteen_name = selected_fifteen.rsplit(" (", 1)[0]
            fifteen_indices = [origin_list.index(name) for name in fifteen_name.split(" + ")]
            fifteen_avg_cost, fifteen_avg_time = combo_avg_cost_time(fifteen_indices)
            prev_combo = best_combo_indices(14)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - fifteen_avg_cost
                prev_time_delta = prev_time - fifteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(fifteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {fifteen_avg_cost:.2f} (delta {avg_cost - fifteen_avg_cost:.2f}) - "
                f"Avg time: {fifteen_avg_time:.2f} (delta {avg_time - fifteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 14 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            fifteen_improved = combo_improved_cities(prev_combo or [], fifteen_indices)
            st.selectbox(
                "Improved destinations vs best 14 origins (15 locations)",
                fifteen_improved or ["None"],
                key=f"{key_prefix}_fifteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(fifteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(fifteen_indices, 1.0)
                not_two_day = combo_not_covered(fifteen_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 15-location combos.")

    if max_k >= 16:
        st.markdown("Top 5 (16 locations)")
        sixteen_list = (expand_from_best(fifteen_list, 16) if len(origin_list) >= 16 else [])
        if sixteen_list:
            sixteen_labels = [f"{name} ({value:.2f})" for name, value in sixteen_list]
            selected_sixteen = st.selectbox(
                "Top 5 (16 locations) (weighted)",
                sixteen_labels,
                index=0,
                key=f"{key_prefix}_top5_sixteens",
            )
            sixteen_name = selected_sixteen.rsplit(" (", 1)[0]
            sixteen_indices = [origin_list.index(name) for name in sixteen_name.split(" + ")]
            sixteen_avg_cost, sixteen_avg_time = combo_avg_cost_time(sixteen_indices)
            prev_combo = best_combo_indices(15)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - sixteen_avg_cost
                prev_time_delta = prev_time - sixteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(sixteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {sixteen_avg_cost:.2f} (delta {avg_cost - sixteen_avg_cost:.2f}) - "
                f"Avg time: {sixteen_avg_time:.2f} (delta {avg_time - sixteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 15 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            sixteen_improved = combo_improved_cities(prev_combo or [], sixteen_indices)
            st.selectbox(
                "Improved destinations vs best 15 origins (16 locations)",
                sixteen_improved or ["None"],
                key=f"{key_prefix}_sixteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(sixteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(sixteen_indices, 1.0)
                not_two_day = combo_not_covered(sixteen_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 16-location combos.")

    if max_k >= 17:
        st.markdown("Top 5 (17 locations)")
        seventeen_list = (expand_from_best(sixteen_list, 17) if len(origin_list) >= 17 else [])
        if seventeen_list:
            seventeen_labels = [f"{name} ({value:.2f})" for name, value in seventeen_list]
            selected_seventeen = st.selectbox(
                "Top 5 (17 locations) (weighted)",
                seventeen_labels,
                index=0,
                key=f"{key_prefix}_top5_seventeens",
            )
            seventeen_name = selected_seventeen.rsplit(" (", 1)[0]
            seventeen_indices = [origin_list.index(name) for name in seventeen_name.split(" + ")]
            seventeen_avg_cost, seventeen_avg_time = combo_avg_cost_time(seventeen_indices)
            prev_combo = best_combo_indices(16)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - seventeen_avg_cost
                prev_time_delta = prev_time - seventeen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(seventeen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {seventeen_avg_cost:.2f} (delta {avg_cost - seventeen_avg_cost:.2f}) - "
                f"Avg time: {seventeen_avg_time:.2f} (delta {avg_time - seventeen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 16 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            seventeen_improved = combo_improved_cities(prev_combo or [], seventeen_indices)
            st.selectbox(
                "Improved destinations vs best 16 origins (17 locations)",
                seventeen_improved or ["None"],
                key=f"{key_prefix}_seventeen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(seventeen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(seventeen_indices, 1.0)
                not_two_day = combo_not_covered(seventeen_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 17-location combos.")

    if max_k >= 18:
        st.markdown("Top 5 (18 locations)")
        eighteen_list = (expand_from_best(seventeen_list, 18) if len(origin_list) >= 18 else [])
        if eighteen_list:
            eighteen_labels = [f"{name} ({value:.2f})" for name, value in eighteen_list]
            selected_eighteen = st.selectbox(
                "Top 5 (18 locations) (weighted)",
                eighteen_labels,
                index=0,
                key=f"{key_prefix}_top5_eighteens",
            )
            eighteen_name = selected_eighteen.rsplit(" (", 1)[0]
            eighteen_indices = [origin_list.index(name) for name in eighteen_name.split(" + ")]
            eighteen_avg_cost, eighteen_avg_time = combo_avg_cost_time(eighteen_indices)
            prev_combo = best_combo_indices(17)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - eighteen_avg_cost
                prev_time_delta = prev_time - eighteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(eighteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {eighteen_avg_cost:.2f} (delta {avg_cost - eighteen_avg_cost:.2f}) - "
                f"Avg time: {eighteen_avg_time:.2f} (delta {avg_time - eighteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 17 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            eighteen_improved = combo_improved_cities(prev_combo or [], eighteen_indices)
            st.selectbox(
                "Improved destinations vs best 17 origins (18 locations)",
                eighteen_improved or ["None"],
                key=f"{key_prefix}_eighteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(eighteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(eighteen_indices, 1.0)
                not_two_day = combo_not_covered(eighteen_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 18-location combos.")

    if max_k >= 19:
        st.markdown("Top 5 (19 locations)")
        nineteen_list = (expand_from_best(eighteen_list, 19) if len(origin_list) >= 19 else [])
        if nineteen_list:
            nineteen_labels = [f"{name} ({value:.2f})" for name, value in nineteen_list]
            selected_nineteen = st.selectbox(
                "Top 5 (19 locations) (weighted)",
                nineteen_labels,
                index=0,
                key=f"{key_prefix}_top5_nineteens",
            )
            nineteen_name = selected_nineteen.rsplit(" (", 1)[0]
            nineteen_indices = [origin_list.index(name) for name in nineteen_name.split(" + ")]
            nineteen_avg_cost, nineteen_avg_time = combo_avg_cost_time(nineteen_indices)
            prev_combo = best_combo_indices(18)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - nineteen_avg_cost
                prev_time_delta = prev_time - nineteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(nineteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {nineteen_avg_cost:.2f} (delta {avg_cost - nineteen_avg_cost:.2f}) - "
                f"Avg time: {nineteen_avg_time:.2f} (delta {avg_time - nineteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 18 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            nineteen_improved = combo_improved_cities(prev_combo or [], nineteen_indices)
            st.selectbox(
                "Improved destinations vs best 18 origins (19 locations)",
                nineteen_improved or ["None"],
                key=f"{key_prefix}_nineteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(nineteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(nineteen_indices, 1.0)
                not_two_day = combo_not_covered(nineteen_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 19-location combos.")

    if max_k >= 20:
        st.markdown("Top 5 (20 locations)")
        twenty_list = (expand_from_best(nineteen_list, 20) if len(origin_list) >= 20 else [])
        if twenty_list:
            twenty_labels = [f"{name} ({value:.2f})" for name, value in twenty_list]
            selected_twenty = st.selectbox(
                "Top 5 (20 locations) (weighted)",
                twenty_labels,
                index=0,
                key=f"{key_prefix}_top5_twenties",
            )
            twenty_name = selected_twenty.rsplit(" (", 1)[0]
            twenty_indices = [origin_list.index(name) for name in twenty_name.split(" + ")]
            twenty_avg_cost, twenty_avg_time = combo_avg_cost_time(twenty_indices)
            prev_combo = best_combo_indices(19)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - twenty_avg_cost
                prev_time_delta = prev_time - twenty_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(twenty_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {twenty_avg_cost:.2f} (delta {avg_cost - twenty_avg_cost:.2f}) - "
                f"Avg time: {twenty_avg_time:.2f} (delta {avg_time - twenty_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 19 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            twenty_improved = combo_improved_cities(prev_combo or [], twenty_indices)
            st.selectbox(
                "Improved destinations vs best 19 origins (20 locations)",
                twenty_improved or ["None"],
                key=f"{key_prefix}_twenty_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2 = combo_day_percentages(twenty_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}%")
                not_one_day = combo_not_covered(twenty_indices, 1.0)
                not_two_day = combo_not_covered(twenty_indices, 2.0)
                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 20-location combos.")


@st.cache_data
def compute_pair_df_cached(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
    build_cost_weight: float,
    dest_weights_items: tuple,
    major_weight_items: tuple,
) -> pd.DataFrame:
    dest_weights = dict(dest_weights_items)
    major_weight_map = dict(major_weight_items)
    total_major_weight = sum(major_weight_map.values())

    cost_mat, time_mat = build_origin_destination_matrices(df, origin_list, dest_list)
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_list], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_list), dtype=float)

    major_dest_indices = [i for i, dest in enumerate(dest_list) if dest in major_weight_map]
    pair_rows = []
    for i, j in combinations(range(len(origin_list)), 2):
        best_cost = np.nanmin(cost_mat[[i, j], :], axis=0)
        best_time = np.nanmin(time_mat[[i, j], :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        if valid_mask.any():
            weights = weights_vec[valid_mask]
            if weights.sum() == 0:
                weights = np.ones(len(weights), dtype=float)
            weighted = build_cost_weight * best_cost[valid_mask] + (1 - build_cost_weight) * best_time[valid_mask]
            weighted_total = float(np.sum(weighted * (weights / weights.sum())))
        else:
            weighted_total = float("nan")

        pair_cost = float(np.nansum(best_cost))
        pair_time = float(np.nansum(best_time))

        if major_weight_map and total_major_weight:
            coverage_weight = 0.0
            for idx in major_dest_indices:
                if not np.isnan(best_time[idx]) and best_time[idx] <= 1.0:
                    coverage_weight += major_weight_map.get(dest_list[idx], 0.0)
            major_coverage_pct = coverage_weight / total_major_weight * 100.0
        else:
            major_coverage_pct = 0.0

        pair_rows.append({
            "Pair": f"{origin_list[i]} + {origin_list[j]}",
            "WeightedTotal": weighted_total,
            "PairCost": pair_cost,
            "PairTime": pair_time,
            "MajorCoveragePct": major_coverage_pct,
        })
    return pd.DataFrame(pair_rows).sort_values("WeightedTotal")


@st.cache_data
def compute_top_k_combos_cached(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
    k: int,
    build_cost_weight: float,
    dest_weights_items: tuple,
    limit: int = 5,
) -> list[tuple[str, float]]:
    dest_weights = dict(dest_weights_items)
    cost_mat, time_mat = build_origin_destination_matrices(df, origin_list, dest_list)
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_list], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_list), dtype=float)

    combos = []
    for combo in combinations(range(len(origin_list)), k):
        best_cost = np.nanmin(cost_mat[list(combo), :], axis=0)
        best_time = np.nanmin(time_mat[list(combo), :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        if valid_mask.any():
            weights = weights_vec[valid_mask]
            if weights.sum() == 0:
                weights = np.ones(len(weights), dtype=float)
            weighted = build_cost_weight * best_cost[valid_mask] + (1 - build_cost_weight) * best_time[valid_mask]
            total = float(np.sum(weighted * (weights / weights.sum())))
        else:
            total = float("nan")
        combos.append((" + ".join(origin_list[idx] for idx in combo), total))
    combos.sort(key=lambda item: item[1])
    if len(combos) <= limit:
        return combos
    cutoff = combos[limit - 1][1]
    if not np.isfinite(cutoff):
        return combos[:limit]
    return [item for item in combos if item[1] <= cutoff]


@st.cache_data
def compute_top_k_combos_with_required_cached(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
    required_origins: tuple,
    k: int,
    build_cost_weight: float,
    dest_weights_items: tuple,
    limit: int = 5,
) -> list[tuple[str, float]]:
    if k < len(required_origins):
        return []
    dest_weights = dict(dest_weights_items)
    cost_mat, time_mat = build_origin_destination_matrices(df, origin_list, dest_list)
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_list], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_list), dtype=float)

    origin_index = {origin: idx for idx, origin in enumerate(origin_list)}
    required_indices = [origin_index[o] for o in required_origins if o in origin_index]
    remaining_indices = [i for i in range(len(origin_list)) if i not in required_indices]
    choose_n = k - len(required_indices)

    combos = []
    for combo in combinations(remaining_indices, choose_n):
        combo_indices = list(required_indices) + list(combo)
        best_cost = np.nanmin(cost_mat[combo_indices, :], axis=0)
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        if valid_mask.any():
            weights = weights_vec[valid_mask]
            if weights.sum() == 0:
                weights = np.ones(len(weights), dtype=float)
            weighted = build_cost_weight * best_cost[valid_mask] + (1 - build_cost_weight) * best_time[valid_mask]
            total = float(np.sum(weighted * (weights / weights.sum())))
        else:
            total = float("nan")
        combos.append((" + ".join(origin_list[idx] for idx in combo_indices), total))
    combos.sort(key=lambda item: item[1])
    if len(combos) <= limit:
        return combos
    cutoff = combos[limit - 1][1]
    if not np.isfinite(cutoff):
        return combos[:limit]
    return [item for item in combos if item[1] <= cutoff]


@st.cache_data
def build_origin_destination_matrices(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    pl_df = pl.from_pandas(df)
    grouped = pl_df.group_by(["FromAddress", "Destination"]).agg(
        pl.col("Cost").min().alias("MinCost"),
        pl.col("ShippingTimeDays").min().alias("MinTime"),
    )

    origin_index = {origin: idx for idx, origin in enumerate(origin_list)}
    dest_index = {dest: idx for idx, dest in enumerate(dest_list)}
    cost_mat = np.full((len(origin_list), len(dest_list)), np.nan, dtype=float)
    time_mat = np.full((len(origin_list), len(dest_list)), np.nan, dtype=float)
    for row in grouped.iter_rows(named=True):
        o_idx = origin_index.get(row["FromAddress"])
        d_idx = dest_index.get(row["Destination"])
        if o_idx is None or d_idx is None:
            continue
        cost_mat[o_idx, d_idx] = row["MinCost"]
        time_mat[o_idx, d_idx] = row["MinTime"]
    return cost_mat, time_mat


def weighted_origin_stats(savings_df: pd.DataFrame, dest_weights: dict) -> pd.DataFrame:
    def compute(group: pd.DataFrame) -> pd.Series:
        weights = group["Destination"].map(dest_weights).fillna(0.0).to_numpy()
        if weights.sum() == 0:
            weights = np.ones(len(group))
        mean_savings = weighted_mean(group["SavingsPerSign"].to_numpy(), weights)
        median_savings = weighted_median(group["SavingsPerSign"].to_numpy(), weights)
        mean_time = weighted_mean(group["TimeSavingsPerPlace"].to_numpy(), weights)
        median_time = weighted_median(group["TimeSavingsPerPlace"].to_numpy(), weights)
        return pd.Series({
            "MeanSavingsPerSign": mean_savings,
            "MedianSavingsPerSign": median_savings,
            "MeanTimeSavingsPerPlace": mean_time,
            "MedianTimeSavingsPerPlace": median_time,
        })
    return savings_df.groupby("FromAddress").apply(compute).reset_index()


def best_total(df: pd.DataFrame, origins, metric: str) -> float:
    subset = df[df["FromAddress"].isin(origins)]
    if subset.empty:
        return float("nan")
    return subset.groupby("Destination")[metric].min().sum()


def best_total_weighted(df: pd.DataFrame, origins, cost_weight: float, dest_weights: dict) -> float:
    subset = df[df["FromAddress"].isin(origins)]
    if subset.empty:
        return float("nan")
    best = subset.groupby("Destination", as_index=False).agg(
        BestCost=("Cost", "min"),
        BestTime=("ShippingTimeDays", "min"),
    )
    best["Weighted"] = cost_weight * best["BestCost"] + (1 - cost_weight) * best["BestTime"]
    weights = best["Destination"].map(dest_weights).fillna(0.0).to_numpy()
    if weights.sum() == 0:
        weights = np.ones(len(best))
    return float(np.sum(best["Weighted"].to_numpy() * (weights / weights.sum())))


def greedy_diminishing_returns(df: pd.DataFrame, baseline_origin: str, metric: str) -> pd.DataFrame:
    origins = sorted(df["FromAddress"].unique())
    if baseline_origin not in origins:
        return pd.DataFrame()
    remaining = [o for o in origins if o != baseline_origin]
    selected = [baseline_origin]
    baseline_total = best_total(df, selected, metric)
    current_total = baseline_total
    rows = [{
        "ShopCount": 1,
        "AddedOrigin": baseline_origin,
        "IncrementalSavings": 0.0,
        "CumulativeSavings": 0.0,
    }]
    # Greedy add: pick the origin that maximizes improvement each step.
    while remaining:
        best_origin = None
        best_improvement = -np.inf
        best_next_total = None
        for origin in remaining:
            total = best_total(df, selected + [origin], metric)
            improvement = current_total - total
            if improvement > best_improvement:
                best_improvement = improvement
                best_origin = origin
                best_next_total = total
        if best_origin is None:
            break
        selected.append(best_origin)
        remaining.remove(best_origin)
        current_total = best_next_total
        rows.append({
            "ShopCount": len(selected),
            "AddedOrigin": best_origin,
            "IncrementalSavings": best_improvement,
            "CumulativeSavings": baseline_total - current_total,
        })
    return pd.DataFrame(rows)


def savings_vs_baseline(df: pd.DataFrame, baseline_origin: str) -> pd.DataFrame:
    baseline = df[df["FromAddress"] == baseline_origin]
    baseline_dest = baseline.groupby("Destination", as_index=False).agg(BaselineCost=("Cost", "mean"))
    baseline_time = baseline.groupby("Destination", as_index=False).agg(BaselineTime=("ShippingTimeDays", "mean"))
    other = df.merge(baseline_dest, on="Destination", how="left").merge(baseline_time, on="Destination", how="left")
    other["SavingsPerSign"] = other["BaselineCost"] - other["Cost"]
    other["TimeSavingsPerPlace"] = other["BaselineTime"] - other["ShippingTimeDays"]
    return other


def greedy_savings_per_sign(
    df: pd.DataFrame,
    baseline_origin: str,
    build_cost_weight: float,
    dest_weights: dict,
) -> pd.DataFrame:
    origins = sorted(df["FromAddress"].unique())
    if baseline_origin not in origins:
        return pd.DataFrame()
    remaining = [o for o in origins if o != baseline_origin]
    selected = [baseline_origin]
    baseline_total = best_total_weighted(df, selected, build_cost_weight, dest_weights)
    current_total = baseline_total
    rows = [{
        "ShopCount": 1,
        "AddedOrigin": baseline_origin,
        "IncrementalSavingsPerSign_Mean": 0.0,
        "IncrementalSavingsPerSign_Median": 0.0,
        "IncrementalTimeSavingsPerPlace_Mean": 0.0,
        "IncrementalTimeSavingsPerPlace_Median": 0.0,
        "IncrementalWeightedSavings_Mean": 0.0,
        "IncrementalWeightedSavings_Median": 0.0,
        "CumulativeSavingsPerSign_Mean": 0.0,
        "CumulativeSavingsPerSign_Median": 0.0,
        "CumulativeTimeSavingsPerPlace_Mean": 0.0,
        "CumulativeTimeSavingsPerPlace_Median": 0.0,
        "CumulativeWeightedSavings_Mean": 0.0,
        "CumulativeWeightedSavings_Median": 0.0,
    }]
    while remaining:
        best_origin = None
        best_improvement = -np.inf
        for origin in remaining:
            total = best_total_weighted(df, selected + [origin], build_cost_weight, dest_weights)
            improvement = current_total - total
            if improvement > best_improvement:
                best_improvement = improvement
                best_origin = origin
        if best_origin is None:
            break
        selected.append(best_origin)
        remaining.remove(best_origin)
        current_total = best_total_weighted(df, selected, build_cost_weight, dest_weights)
        baseline = df[df["FromAddress"] == baseline_origin].groupby("Destination", as_index=False).agg(
            BaselineCost=("Cost", "mean"),
            BaselineTime=("ShippingTimeDays", "mean"),
        )
        best = df[df["FromAddress"].isin(selected)].groupby("Destination", as_index=False).agg(
            BestCost=("Cost", "min"),
            BestTime=("ShippingTimeDays", "min"),
        )
        merged = best.merge(baseline, on="Destination", how="left")
        merged["SavingsPerSign"] = merged["BaselineCost"] - merged["BestCost"]
        merged["TimeSavingsPerPlace"] = merged["BaselineTime"] - merged["BestTime"]
        merged["CostSavingsNorm"] = minmax(merged["SavingsPerSign"])
        merged["TimeSavingsNorm"] = minmax(merged["TimeSavingsPerPlace"])
        merged["WeightedSavings"] = (
            build_cost_weight * merged["CostSavingsNorm"] + (1 - build_cost_weight) * merged["TimeSavingsNorm"]
        )
        weights = merged["Destination"].map(dest_weights).fillna(0.0).to_numpy()
        if weights.sum() == 0:
            weights = np.ones(len(merged))
        mean_savings = weighted_mean(merged["SavingsPerSign"].to_numpy(), weights)
        median_savings = weighted_median(merged["SavingsPerSign"].to_numpy(), weights)
        mean_time = weighted_mean(merged["TimeSavingsPerPlace"].to_numpy(), weights)
        median_time = weighted_median(merged["TimeSavingsPerPlace"].to_numpy(), weights)
        mean_weighted = weighted_mean(merged["WeightedSavings"].to_numpy(), weights)
        median_weighted = weighted_median(merged["WeightedSavings"].to_numpy(), weights)
        prev_best = df[df["FromAddress"].isin(selected[:-1])].groupby("Destination", as_index=False).agg(
            BestCost=("Cost", "min"),
            BestTime=("ShippingTimeDays", "min"),
        )
        prev_merged = prev_best.merge(baseline, on="Destination", how="left")
        prev_merged["SavingsPerSign"] = prev_merged["BaselineCost"] - prev_merged["BestCost"]
        prev_merged["TimeSavingsPerPlace"] = prev_merged["BaselineTime"] - prev_merged["BestTime"]
        prev_merged["CostSavingsNorm"] = minmax(prev_merged["SavingsPerSign"])
        prev_merged["TimeSavingsNorm"] = minmax(prev_merged["TimeSavingsPerPlace"])
        prev_merged["WeightedSavings"] = (
            build_cost_weight * prev_merged["CostSavingsNorm"] + (1 - build_cost_weight) * prev_merged["TimeSavingsNorm"]
        )
        prev_weights = prev_merged["Destination"].map(dest_weights).fillna(0.0).to_numpy()
        if prev_weights.sum() == 0:
            prev_weights = np.ones(len(prev_merged))
        prev_mean = weighted_mean(prev_merged["SavingsPerSign"].to_numpy(), prev_weights)
        prev_median = weighted_median(prev_merged["SavingsPerSign"].to_numpy(), prev_weights)
        prev_mean_time = weighted_mean(prev_merged["TimeSavingsPerPlace"].to_numpy(), prev_weights)
        prev_median_time = weighted_median(prev_merged["TimeSavingsPerPlace"].to_numpy(), prev_weights)
        prev_mean_weighted = weighted_mean(prev_merged["WeightedSavings"].to_numpy(), prev_weights)
        prev_median_weighted = weighted_median(prev_merged["WeightedSavings"].to_numpy(), prev_weights)
        rows.append({
            "ShopCount": len(selected),
            "AddedOrigin": best_origin,
            "IncrementalSavingsPerSign_Mean": mean_savings - prev_mean,
            "IncrementalSavingsPerSign_Median": median_savings - prev_median,
            "IncrementalTimeSavingsPerPlace_Mean": mean_time - prev_mean_time,
            "IncrementalTimeSavingsPerPlace_Median": median_time - prev_median_time,
            "IncrementalWeightedSavings_Mean": mean_weighted - prev_mean_weighted,
            "IncrementalWeightedSavings_Median": median_weighted - prev_median_weighted,
            "CumulativeSavingsPerSign_Mean": mean_savings,
            "CumulativeSavingsPerSign_Median": median_savings,
            "CumulativeTimeSavingsPerPlace_Mean": mean_time,
            "CumulativeTimeSavingsPerPlace_Median": median_time,
            "CumulativeWeightedSavings_Mean": mean_weighted,
            "CumulativeWeightedSavings_Median": median_weighted,
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Shipping Cost Dashboard", layout="wide")
st.title("Shipping Cost Dashboard")
tab_dashboard, tab_regionals_v3 = st.tabs(
    ["Dashboard", "Shipping Summary Regionals v3"]
)

with tab_dashboard:

    dataset_name = st.sidebar.radio("Page", list(DATASETS.keys()), index=0)
    data_path = Path(DATASETS[dataset_name])
    df_raw = load_data(str(data_path))

    cities = sorted(df_raw["ToCity"].unique())
    origins = sorted(df_raw["FromAddress"].unique())

    default_baseline = "Indianapolis" if "Indianapolis" in df_raw["FromAddress"].unique() else sorted(df_raw["FromAddress"].unique())[0]
    baseline_origin = st.sidebar.selectbox("Baseline origin", sorted(df_raw["FromAddress"].unique()), index=sorted(df_raw["FromAddress"].unique()).index(default_baseline))

    cost_weight = st.sidebar.slider("Cost weight (Time weight = 1 - Cost)", 0.0, 1.0, 0.0, 0.05)
    signs_per_month = st.sidebar.number_input("Signs sold per month", min_value=0, value=500, step=50)
    build_cost_weight = cost_weight
    st.sidebar.markdown("Size mix (%)")
    size_small_pct = st.sidebar.number_input("Small", min_value=0.0, max_value=100.0, value=45.0, step=1.0, format="%.2f")
    size_medium_pct = st.sidebar.number_input("Medium", min_value=0.0, max_value=100.0, value=33.0, step=1.0, format="%.2f")
    size_large_pct = st.sidebar.number_input("Large", min_value=0.0, max_value=100.0, value=22.0, step=1.0, format="%.2f")
    size_mix_total = size_small_pct + size_medium_pct + size_large_pct
    if size_mix_total <= 0:
        size_mix = {"Small": 0.0, "Medium": 0.0, "Large": 0.0}
    else:
        size_mix = {
            "Small": size_small_pct / size_mix_total,
            "Medium": size_medium_pct / size_mix_total,
            "Large": size_large_pct / size_mix_total,
        }
    if abs(size_mix_total - 100.0) > 0.5:
        st.sidebar.caption("Size mix is normalized to 100%.")

    df_filtered = apply_filters(df_raw, cities, origins)
    if df_filtered.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    df = add_weighted_score(df_filtered, cost_weight)
    if baseline_origin not in df["FromAddress"].unique():
        effective_baseline = sorted(df["FromAddress"].unique())[0]
        st.warning(f"Baseline origin '{baseline_origin}' is not in the filtered data. Using '{effective_baseline}' instead.")
        baseline_origin = effective_baseline

    dest_in_view = sorted(df["Destination"].unique())
    if st.session_state.get("destination_weights_version") != "state_counts_v1":
        st.session_state.destination_weights = initial_destination_weights(df_raw, STATE_COUNTS)
        st.session_state.destination_weights_version = "state_counts_v1"
    st.sidebar.markdown("Destination weights")
    selected_dest = st.sidebar.selectbox(
        "Edit destination weight",
        dest_in_view,
        index=0,
        key="dest_weight_select",
    )
    current_weight = float(st.session_state.destination_weights.get(selected_dest, DEFAULT_DEST_WEIGHT))
    new_weight = st.sidebar.number_input(
        "Weight",
        min_value=0.0,
        value=current_weight,
        step=0.5,
        format="%.2f",
        key="dest_weight_value",
    )
    st.session_state.destination_weights[selected_dest] = float(new_weight)
    dest_weights = destination_weights(dest_in_view, st.session_state.destination_weights)
    st.sidebar.caption(f"Custom weights applied across {len(dest_in_view)} destinations in view.")

    st.subheader(f"{dataset_name} shipments")

    best_time_by_dest = df.groupby("Destination", as_index=False).agg(BestTime=("ShippingTimeDays", "min"))
    best_time_by_dest["Weight"] = best_time_by_dest["Destination"].map(dest_weights).fillna(0.0)
    optimal_time_avg = weighted_mean(best_time_by_dest["BestTime"].to_numpy(), best_time_by_dest["Weight"].to_numpy())
    baseline_time_by_dest = df[df["FromAddress"] == baseline_origin].groupby(
        "Destination",
        as_index=False,
    ).agg(AvgTime=("ShippingTimeDays", "mean"))
    baseline_time_by_dest["Weight"] = baseline_time_by_dest["Destination"].map(dest_weights).fillna(0.0)
    avg_time = weighted_mean(
        baseline_time_by_dest["AvgTime"].to_numpy(),
        baseline_time_by_dest["Weight"].to_numpy(),
    )
    time_savings_pct = ((avg_time - optimal_time_avg) / avg_time * 100.0) if avg_time else 0.0
    baseline_cost_by_dest = df[df["FromAddress"] == baseline_origin].groupby(
        "Destination",
        as_index=False,
    ).agg(AvgCost=("Cost", "mean"))
    baseline_cost_by_dest["Weight"] = baseline_cost_by_dest["Destination"].map(dest_weights).fillna(0.0)
    avg_cost = weighted_mean(
        baseline_cost_by_dest["AvgCost"].to_numpy(),
        baseline_cost_by_dest["Weight"].to_numpy(),
    )
    best_cost_by_dest = df.groupby("Destination", as_index=False).agg(BestCost=("Cost", "min"))
    best_cost_by_dest["Weight"] = best_cost_by_dest["Destination"].map(dest_weights).fillna(0.0)
    optimal_cost_avg = weighted_mean(best_cost_by_dest["BestCost"].to_numpy(), best_cost_by_dest["Weight"].to_numpy())
    cost_savings_pct = ((avg_cost - optimal_cost_avg) / avg_cost * 100.0) if avg_cost else 0.0

    avg_by_origin = df.groupby("FromAddress", as_index=False).agg(
        AvgCost=("Cost", "mean"),
        AvgTime=("ShippingTimeDays", "mean"),
        AvgWeighted=("WeightedScore", "mean"),
        Destinations=("Destination", "nunique"),
    )

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
    col1.metric("Avg cost", f"{avg_cost:.2f}")
    col2.metric("Avg shipping days", f"{avg_time:.2f}")
    col3.metric("Optimal avg time", f"{optimal_time_avg:.2f}")
    col4.metric("Optimal avg cost", f"{optimal_cost_avg:.2f}")
    col5.metric("% time saved vs optimal", f"{time_savings_pct:.1f}%")
    col6.metric("% cost saved vs optimal", f"{cost_savings_pct:.1f}%")
    best_cost = avg_by_origin.sort_values("AvgCost").iloc[0]["FromAddress"]
    best_time = avg_by_origin.sort_values("AvgTime").iloc[0]["FromAddress"]
    best_weighted = avg_by_origin.sort_values("AvgWeighted").iloc[0]["FromAddress"]
    col7.metric("Best origin (cost)", best_cost)
    col8.metric("Best origin (time)", best_time)
    col9.metric("Best origin (weighted)", best_weighted)

    top3_cost = avg_by_origin.sort_values("AvgCost").head(3)["FromAddress"].tolist()
    top3_time = avg_by_origin.sort_values("AvgTime").head(3)["FromAddress"].tolist()
    top3_weighted = avg_by_origin.sort_values("AvgWeighted").head(3)["FromAddress"].tolist()

    top_cols = st.columns(3)
    top_cols[0].selectbox("Top 3 origins (cost)", top3_cost, index=0, key="top3_cost")
    top_cols[1].selectbox("Top 3 origins (time)", top3_time, index=0, key="top3_time")
    top_cols[2].selectbox("Top 3 origins (weighted)", top3_weighted, index=0, key="top3_weighted")

    origin_list_all = sorted(df["FromAddress"].unique())
    # Candidate origins for combos: baseline + (has any 1-day) OR (top 10 avg time reducers)
    one_day_origins = sorted(
        df[df["ShippingTimeDays"] <= 1.0]["FromAddress"].unique().tolist()
    )
    avg_time_by_origin = df.groupby("FromAddress", as_index=False).agg(AvgTime=("ShippingTimeDays", "mean"))
    baseline_avg_time = float(
        avg_time_by_origin[avg_time_by_origin["FromAddress"] == baseline_origin]["AvgTime"].iloc[0]
    ) if baseline_origin in avg_time_by_origin["FromAddress"].values else float("nan")
    avg_time_by_origin["TimeReductionVsBaseline"] = baseline_avg_time - avg_time_by_origin["AvgTime"]
    top_time_reducers = (
        avg_time_by_origin.sort_values("TimeReductionVsBaseline", ascending=False)
        .head(10)["FromAddress"]
        .tolist()
    )
    origin_candidates = sorted(
        {baseline_origin} | set(one_day_origins) | set(top_time_reducers)
    )
    origin_list = [o for o in origin_candidates if o in origin_list_all]
    major_destinations = [
        dest for dest in dest_in_view
        if float(st.session_state.destination_weights.get(dest, DEFAULT_DEST_WEIGHT)) > 1.0
    ]
    major_weight_map = {
        dest: float(st.session_state.destination_weights.get(dest, DEFAULT_DEST_WEIGHT))
        for dest in major_destinations
    }
    st.markdown("Built Network")
    built_network_origins = st.multiselect(
        "Included origins",
        sorted(df["FromAddress"].unique()),
        default=[],
    )
    built_subset = df[df["FromAddress"].isin(built_network_origins)]
    if built_subset.empty:
        built_optimal_time_avg = float("nan")
        built_optimal_cost_avg = float("nan")
        slow_destinations = []
        slower_destinations = []
        slowest_destinations = []
        built_best_time = pd.DataFrame(columns=["Destination", "BestTime"])
    else:
        built_best_time = built_subset.groupby("Destination", as_index=False).agg(BestTime=("ShippingTimeDays", "min"))
        built_best_time["Weight"] = built_best_time["Destination"].map(dest_weights).fillna(0.0)
        built_optimal_time_avg = weighted_mean(
            built_best_time["BestTime"].to_numpy(),
            built_best_time["Weight"].to_numpy(),
        )
        built_best_cost = built_subset.groupby("Destination", as_index=False).agg(BestCost=("Cost", "min"))
        built_best_cost["Weight"] = built_best_cost["Destination"].map(dest_weights).fillna(0.0)
        built_optimal_cost_avg = weighted_mean(
            built_best_cost["BestCost"].to_numpy(),
            built_best_cost["Weight"].to_numpy(),
        )
        slow_destinations = (
            built_best_time[built_best_time["BestTime"] > 1.0]["Destination"]
            .sort_values()
            .tolist()
        )
        slower_destinations = (
            built_best_time[built_best_time["BestTime"] > 2.0]["Destination"]
            .sort_values()
            .tolist()
        )
        slowest_destinations = (
            built_best_time[built_best_time["BestTime"] > 3.0]["Destination"]
            .sort_values()
            .tolist()
        )

    st.markdown("Reduce costs suggested build")
    st.markdown("Reduce shipping times suggested build")
    available_origins = [o for o in origins if o not in built_network_origins]
    if available_origins:
        current_avg_cost = built_optimal_cost_avg
        current_avg_time = built_optimal_time_avg
        current_best_time = built_best_time.set_index("Destination")["BestTime"] if not built_best_time.empty else pd.Series(dtype=float)

        best_cost_origin = None
        best_cost_delta = -np.inf
        best_cost_info = {}

        best_time_origin = None
        best_time_delta = -np.inf
        best_time_info = {}

        for origin in available_origins:
            candidate_subset = df[df["FromAddress"].isin(built_network_origins + [origin])]
            candidate_best_time = candidate_subset.groupby("Destination", as_index=False).agg(
                BestTime=("ShippingTimeDays", "min")
            )
            candidate_best_time["Weight"] = candidate_best_time["Destination"].map(dest_weights).fillna(0.0)
            candidate_avg_time = weighted_mean(
                candidate_best_time["BestTime"].to_numpy(),
                candidate_best_time["Weight"].to_numpy(),
            )
            candidate_best_cost = candidate_subset.groupby("Destination", as_index=False).agg(
                BestCost=("Cost", "min")
            )
            candidate_best_cost["Weight"] = candidate_best_cost["Destination"].map(dest_weights).fillna(0.0)
            candidate_avg_cost = weighted_mean(
                candidate_best_cost["BestCost"].to_numpy(),
                candidate_best_cost["Weight"].to_numpy(),
            )

            candidate_time_map = candidate_best_time.set_index("Destination")["BestTime"]
            common_dest = candidate_time_map.index.union(current_best_time.index)
            current_times = current_best_time.reindex(common_dest)
            new_times = candidate_time_map.reindex(common_dest)
            moved_3_to_2 = common_dest[(current_times > 2.0) & (new_times <= 2.0) & (new_times > 1.0)].tolist()
            moved_2_to_1 = common_dest[(current_times > 1.0) & (new_times <= 1.0)].tolist()

            cost_delta = (current_avg_cost - candidate_avg_cost) if np.isfinite(current_avg_cost) else -candidate_avg_cost
            time_delta = (current_avg_time - candidate_avg_time) if np.isfinite(current_avg_time) else -candidate_avg_time

            if cost_delta > best_cost_delta:
                best_cost_delta = cost_delta
                best_cost_origin = origin
                best_cost_info = {
                    "avg_cost": candidate_avg_cost,
                    "avg_time": candidate_avg_time,
                    "moved_3_to_2": moved_3_to_2,
                    "moved_2_to_1": moved_2_to_1,
                }
            if time_delta > best_time_delta:
                best_time_delta = time_delta
                best_time_origin = origin
                best_time_info = {
                    "avg_cost": candidate_avg_cost,
                    "avg_time": candidate_avg_time,
                    "moved_3_to_2": moved_3_to_2,
                    "moved_2_to_1": moved_2_to_1,
                }

        if best_cost_origin is not None:
            st.caption(
                f"Reduce costs suggested build: {best_cost_origin} "
                f"(avg cost ↓ {best_cost_delta:.2f}, avg time {best_cost_info['avg_time']:.2f})"
            )
            st.caption(
                f"3→2 day cities: {', '.join(best_cost_info['moved_3_to_2']) or 'None'}"
            )
            st.caption(
                f"2→1 day cities: {', '.join(best_cost_info['moved_2_to_1']) or 'None'}"
            )
        else:
            st.caption("Reduce costs suggested build: n/a")

        if best_time_origin is not None:
            st.caption(
                f"Reduce shipping times suggested build: {best_time_origin} "
                f"(avg time ↓ {best_time_delta:.2f}, avg cost {best_time_info['avg_cost']:.2f})"
            )
            st.caption(
                f"3→2 day cities: {', '.join(best_time_info['moved_3_to_2']) or 'None'}"
            )
            st.caption(
                f"2→1 day cities: {', '.join(best_time_info['moved_2_to_1']) or 'None'}"
            )
        else:
            st.caption("Reduce shipping times suggested build: n/a")
    else:
        st.caption("Reduce costs suggested build: n/a")
        st.caption("Reduce shipping times suggested build: n/a")

    def size_mix_monthly_savings(size_mix: dict, signs_per_month: float) -> float:
        total = 0.0
        for size, share in size_mix.items():
            if share <= 0:
                continue
            size_df = load_data(DATASETS[size])
            if size_df.empty:
                continue
            origins = sorted(size_df["FromAddress"].unique())
            if not origins:
                continue
            baseline = baseline_origin if baseline_origin in origins else origins[0]
            built_origins = [o for o in built_network_origins if o in origins]
            if not built_origins:
                built_origins = [baseline]
            dest_in_view = sorted(size_df["Destination"].unique())
            base_weights = (
                st.session_state.destination_weights
                if "destination_weights" in st.session_state
                else initial_destination_weights(size_df, STATE_COUNTS)
            )
            dest_weights_size = destination_weights(dest_in_view, base_weights)
            baseline_cost_by_dest = size_df[size_df["FromAddress"] == baseline].groupby(
                "Destination",
                as_index=False,
            ).agg(AvgCost=("Cost", "mean"))
            baseline_cost_by_dest["Weight"] = baseline_cost_by_dest["Destination"].map(dest_weights_size).fillna(0.0)
            baseline_avg_cost = weighted_mean(
                baseline_cost_by_dest["AvgCost"].to_numpy(),
                baseline_cost_by_dest["Weight"].to_numpy(),
            )
            built_subset_size = size_df[size_df["FromAddress"].isin(built_origins)]
            if built_subset_size.empty:
                continue
            built_best_cost = built_subset_size.groupby("Destination", as_index=False).agg(BestCost=("Cost", "min"))
            built_best_cost["Weight"] = built_best_cost["Destination"].map(dest_weights_size).fillna(0.0)
            built_avg_cost = weighted_mean(
                built_best_cost["BestCost"].to_numpy(),
                built_best_cost["Weight"].to_numpy(),
            )
            if np.isfinite(baseline_avg_cost) and np.isfinite(built_avg_cost):
                total += share * signs_per_month * (baseline_avg_cost - built_avg_cost)
        return total

    monthly_savings = size_mix_monthly_savings(size_mix, float(signs_per_month))
    yearly_savings = monthly_savings * 12 if np.isfinite(monthly_savings) else float("nan")
    col10, col11, col12, col13 = st.columns(4)
    col10.metric("Built network avg time", f"{built_optimal_time_avg:.2f}" if np.isfinite(built_optimal_time_avg) else "n/a")
    col11.metric("Built network avg cost", f"{built_optimal_cost_avg:.2f}" if np.isfinite(built_optimal_cost_avg) else "n/a")
    col12.metric("Monthly savings (size mix)", f"{monthly_savings:,.2f}" if np.isfinite(monthly_savings) else "n/a")
    col13.metric("Yearly savings (size mix)", f"{yearly_savings:,.2f}" if np.isfinite(yearly_savings) else "n/a")

    st.markdown("Major City Score")
    major_weight_map = st.session_state.destination_weights
    major_destinations = [dest for dest in dest_in_view if float(major_weight_map.get(dest, DEFAULT_DEST_WEIGHT)) > 1.0]
    if built_subset.empty or not major_destinations:
        st.caption("Major City Score: n/a")
    else:
        major_weights = {
            dest: float(major_weight_map.get(dest, DEFAULT_DEST_WEIGHT))
            for dest in major_destinations
        }
        total_major_weight = sum(major_weights.values())
        major_best = built_best_time[built_best_time["Destination"].isin(major_destinations)]
        one_day_dest = major_best[major_best["BestTime"] <= 1.0]["Destination"].tolist()
        one_day_weight = sum(major_weights.get(dest, 0.0) for dest in one_day_dest)
        major_score = (one_day_weight / total_major_weight * 100.0) if total_major_weight else 0.0
        st.caption(f"Major City Score: {major_score:.1f}% of weighted major destinations at 1-day shipping.")


    st.markdown("Cities without 1-day shipping in built network")
    if slow_destinations:
        selected_slow_dest = st.selectbox(
            "Destinations with best shipping time > 1 day",
            slow_destinations,
        )
        selected_rows = built_subset[built_subset["Destination"] == selected_slow_dest]
        if selected_rows.empty:
            st.caption("No routes found for the selected destination in the built network.")
        else:
            best_row = selected_rows.sort_values(["ShippingTimeDays", "Cost"]).iloc[0]
            st.caption(
                f"Fastest from: {best_row['FromAddress']} - {best_row['ShippingTimeDays']:.2f} days"
            )
    else:
        st.caption("All destinations have 1-day shipping in the current built network.")

    st.markdown("Cities without 2-day shipping in built network")
    if slower_destinations:
        selected_slower_dest = st.selectbox(
            "Destinations with best shipping time > 2 days",
            slower_destinations,
            key="slow_destinations_2day",
        )
        selected_rows = built_subset[built_subset["Destination"] == selected_slower_dest]
        if selected_rows.empty:
            st.caption("No routes found for the selected destination in the built network.")
        else:
            best_row = selected_rows.sort_values(["ShippingTimeDays", "Cost"]).iloc[0]
            st.caption(
                f"Fastest from: {best_row['FromAddress']} - {best_row['ShippingTimeDays']:.2f} days"
            )
    else:
        st.caption("All destinations have 2-day shipping in the current built network.")

    st.markdown("Cities without 3-day shipping in built network")
    if slowest_destinations:
        selected_slowest_dest = st.selectbox(
            "Destinations with best shipping time > 3 days",
            slowest_destinations,
            key="slow_destinations_3day",
        )
        selected_rows = built_subset[built_subset["Destination"] == selected_slowest_dest]
        if selected_rows.empty:
            st.caption("No routes found for the selected destination in the built network.")
        else:
            best_row = selected_rows.sort_values(["ShippingTimeDays", "Cost"]).iloc[0]
            st.caption(
                f"Fastest from: {best_row['FromAddress']} - {best_row['ShippingTimeDays']:.2f} days"
            )
    else:
        st.caption("All destinations have 3-day shipping in the current built network.")

    st.markdown("---")

    origin_count = avg_by_origin.shape[0]
    bar_height = max(300, origin_count * 34)

    max_scatter_cost = float(avg_by_origin["AvgCost"].max()) if not avg_by_origin.empty else 5.75
    scatter = alt.Chart(avg_by_origin).mark_circle(size=140).encode(
        x=alt.X("AvgCost:Q", title="Avg Cost", scale=alt.Scale(domain=[5.75, max_scatter_cost])),
        y=alt.Y("AvgTime:Q", title="Avg Shipping Days"),
        size=alt.Size("Destinations:Q", title="Destinations"),
        color=alt.Color(
            "FromAddress:N",
            legend=alt.Legend(title="Origin", labelLimit=220),
        ),
        tooltip=["FromAddress", "AvgCost", "AvgTime", "Destinations"],
    ).properties(height=320, title="Cost vs time by origin")

    baseline_df = df[df["FromAddress"] == baseline_origin]
    baseline_dest = baseline_df.groupby("Destination", as_index=False).agg(
        BaselineCost=("Cost", "mean"),
        BaselineTime=("ShippingTimeDays", "mean"),
        BaselineWeighted=("WeightedScore", "mean"),
    )
    best_dest = df.groupby("Destination", as_index=False).agg(
        BestCost=("Cost", "min"),
        BestTime=("ShippingTimeDays", "min"),
        BestWeighted=("WeightedScore", "min"),
    )
    savings = best_dest.merge(baseline_dest, on="Destination", how="left")
    savings["CostSavings"] = savings["BaselineCost"] - savings["BestCost"]
    savings["TimeSavings"] = savings["BaselineTime"] - savings["BestTime"]
    savings["WeightedSavings"] = savings["BaselineWeighted"] - savings["BestWeighted"]

    savings_signs = savings_vs_baseline(df, baseline_origin)
    origin_savings = weighted_origin_stats(savings_signs, dest_weights)
    origin_savings["MeanMonthlySavings"] = origin_savings["MeanSavingsPerSign"] * signs_per_month
    origin_savings["MedianMonthlySavings"] = origin_savings["MedianSavingsPerSign"] * signs_per_month

    bar_weighted = alt.Chart(avg_by_origin).mark_bar().encode(
        x=alt.X("AvgWeighted:Q", title="Avg Weighted Score"),
        y=alt.Y("FromAddress:N", sort="x", axis=alt.Axis(labelLimit=200)),
        tooltip=["FromAddress", "AvgWeighted"],
    ).properties(height=bar_height, title="Weighted score by origin (lower is better)")

    returns_signs = greedy_savings_per_sign(df, baseline_origin, build_cost_weight, dest_weights)
    st.caption("Cost vs time trade-off by origin. Points show the average cost and average time together.")
    st.altair_chart(scatter, use_container_width=True)

    st.caption("Weighted score by origin using the cost/time weight slider. Lower is better.")
    st.altair_chart(bar_weighted, use_container_width=True)
    metric_choice = st.selectbox(
        "Diminishing returns metric",
        ["Cost", "ShippingTimeDays", "WeightedScore"],
    )
    returns = greedy_diminishing_returns(df, baseline_origin, metric_choice)
    line_returns = alt.Chart(returns).mark_line(point=True).encode(
        x=alt.X("ShopCount:Q", title="Shop count (including baseline)"),
        y=alt.Y("CumulativeSavings:Q", title=f"Cumulative savings ({metric_choice})"),
        tooltip=["ShopCount", "AddedOrigin", "IncrementalSavings", "CumulativeSavings"],
    ).properties(height=300, title="Diminishing returns as shops are added")
    line_returns_labels = alt.Chart(returns).mark_text(dy=-10, color="white").encode(
        x=alt.X("ShopCount:Q"),
        y=alt.Y("CumulativeSavings:Q"),
        text=alt.Text("AddedOrigin:N"),
    )
    line_returns = line_returns + line_returns_labels
    st.caption("Diminishing returns using the selected metric. Each step adds the next best origin.")
    st.altair_chart(line_returns, use_container_width=True)
    st.caption("Per-origin savings per sign table vs Indianapolis, weighted by destination share and sorted by mean savings.")
    if not returns_signs.empty:
        st.subheader("Recommended shop build order (weighted cost/time and destination share)")
        display_cols = [
            "ShopCount",
            "AddedOrigin",
            "IncrementalSavingsPerSign_Mean",
            "IncrementalSavingsPerSign_Median",
            "IncrementalTimeSavingsPerPlace_Mean",
            "IncrementalTimeSavingsPerPlace_Median",
            "IncrementalWeightedSavings_Mean",
            "IncrementalWeightedSavings_Median",
            "CumulativeSavingsPerSign_Mean",
            "CumulativeSavingsPerSign_Median",
            "CumulativeTimeSavingsPerPlace_Mean",
            "CumulativeTimeSavingsPerPlace_Median",
            "CumulativeWeightedSavings_Mean",
            "CumulativeWeightedSavings_Median",
        ]
        st.dataframe(returns_signs[display_cols].head(11), use_container_width=True)

    with st.expander("Show filtered data"):
        st.dataframe(df, use_container_width=True)

    render_top_combos(
        df,
        origin_list,
        dest_in_view,
        dest_weights,
        build_cost_weight,
        avg_cost,
        avg_time,
        major_weight_map,
        "main",
    )
    

    st.markdown("Final origin rankings")
    origin_rank = df.groupby("FromAddress", as_index=False).agg(
        AvgWeighted=("WeightedScore", "mean"),
        AvgCost=("Cost", "mean"),
        AvgTime=("ShippingTimeDays", "mean"),
        MinTime=("ShippingTimeDays", "min"),
    )
    origin_rank = origin_rank.sort_values(
        ["AvgWeighted", "AvgTime", "AvgCost"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    origin_rank["Rank"] = np.arange(1, len(origin_rank) + 1)
    st.dataframe(
        origin_rank[["Rank", "FromAddress", "AvgWeighted", "AvgCost", "AvgTime", "MinTime"]],
        use_container_width=True,
    )

with tab_regionals_v3:
    st.subheader("Shipping Summary Regionals v3")
    st.caption("Top combos for the regional v3 data (page3.csv).")
    regionals_v3_path = Path("page3.csv")
    if regionals_v3_path.exists():
        regionals_v3_df = load_data(str(regionals_v3_path))
        if regionals_v3_df.empty:
            st.info("No data available in page3.csv.")
        else:
            st.metric("Rows", f"{len(regionals_v3_df):,}")
            st.markdown("Data sanity checks")
            null_counts = regionals_v3_df.isna().sum()
            nulls_df = (
                null_counts[null_counts > 0]
                .sort_values(ascending=False)
                .rename("NullCount")
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            if nulls_df.empty:
                st.caption("Null values: none")
            else:
                st.dataframe(nulls_df, use_container_width=True)

            unique_from = sorted(regionals_v3_df["FromAddress"].unique())
            unique_to = sorted(regionals_v3_df["Destination"].unique())
            total_possible_pairs = len(unique_from) * len(unique_to)
            actual_pairs = regionals_v3_df[["FromAddress", "Destination"]].drop_duplicates()
            missing_pairs = total_possible_pairs - len(actual_pairs)
            coverage_pct = (len(actual_pairs) / total_possible_pairs * 100.0) if total_possible_pairs else 0.0
            st.caption(
                f"From/To coverage: {len(actual_pairs):,} of {total_possible_pairs:,} "
                f"pairs ({coverage_pct:.1f}%), missing {missing_pairs:,}"
            )
            if missing_pairs == 0:
                st.caption("All FromAddress entries have routes to all Destination entries.")
            else:
                missing_from = [
                    from_city
                    for from_city in unique_from
                    if len(actual_pairs[actual_pairs["FromAddress"] == from_city]) != len(unique_to)
                ]
                if missing_from:
                    st.caption(f"From cities missing destinations: {len(missing_from)}")
                    st.selectbox(
                        "Example FromAddress missing destinations",
                        missing_from,
                        index=0,
                        key="v3_missing_from_example",
                    )
                    selected_missing_from = st.selectbox(
                        "Show missing destinations for",
                        missing_from,
                        index=0,
                        key="v3_missing_from_detail",
                    )
                    present = set(
                        actual_pairs[actual_pairs["FromAddress"] == selected_missing_from]["Destination"]
                    )
                    missing_destinations = [dest for dest in unique_to if dest not in present]
                    st.selectbox(
                        "Missing destinations for selected FromAddress",
                        missing_destinations or ["None"],
                        key="v3_missing_from_destinations",
                    )
                else:
                    st.caption("From cities missing destinations: 0")

            duplicates = regionals_v3_df.duplicated(
                subset=["FromAddress", "ToCity", "ToState", "PackageSize", "Service"],
                keep=False,
            )
            dup_count = int(duplicates.sum())
            st.caption(f"Duplicate rows (same from/to/size/service): {dup_count:,}")

            lat_long_path = Path("lat_long_from_to.csv")
            if lat_long_path.exists():
                lat_long_df = pd.read_csv(lat_long_path)
                lat_long_df["direction"] = lat_long_df["direction"].astype(str).str.upper()
                lat_long_df["city_key"] = (
                    lat_long_df["city"].astype(str).str.strip().str.casefold()
                    + "||"
                    + lat_long_df["state"].astype(str).str.strip().str.casefold()
                )
                from_keys = set(lat_long_df[lat_long_df["direction"] == "FROM"]["city_key"])
                map_points = lat_long_df.dropna(subset=["lat", "lng"]).copy()
                map_points["direction"] = np.where(
                    map_points["city_key"].isin(from_keys),
                    "FROM",
                    map_points["direction"],
                )
                if not map_points.empty:
                    st.markdown("Origin vs destination map")
                    map_points["color"] = np.where(
                        map_points["direction"] == "FROM",
                        "FROM",
                        "TO",
                    )
                    color_map = {
                        "FROM": [27, 124, 212, 180],
                        "TO": [217, 83, 79, 180],
                    }
                    map_points["color_rgba"] = map_points["color"].map(color_map)
                    view_state = pdk.ViewState(
                        latitude=float(map_points["lat"].mean()),
                        longitude=float(map_points["lng"].mean()),
                        zoom=3.5,
                    )
                    scatter = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_points,
                        get_position="[lng, lat]",
                        get_fill_color="color_rgba",
                        get_radius=35000,
                        pickable=True,
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[scatter],
                            initial_view_state=view_state,
                            tooltip={"text": "{direction} - {name}\n{city}, {state} {postal_code}"},
                        )
                    )
                else:
                    st.caption("No valid lat/long rows found in lat_long_from_to.csv.")
            else:
                st.caption("lat_long_from_to.csv not found for the map.")
            origin_list_all_v3 = sorted(regionals_v3_df["FromAddress"].unique())
            if not origin_list_all_v3:
                st.info("No origins available in shipping_summary_regionals_v3.csv.")
            else:
                default_baseline_v3 = (
                    "Harrisburg" if "Harrisburg" in origin_list_all_v3 else origin_list_all_v3[0]
                )
                baseline_origin_v3 = st.selectbox(
                    "Baseline origin (v3)",
                    origin_list_all_v3,
                    index=origin_list_all_v3.index(default_baseline_v3),
                    key="v3_baseline_origin",
                )
                cost_weight_v3 = st.slider(
                    "Cost weight (Time weight = 1 - Cost) (v3)",
                    0.0,
                    1.0,
                    0.0,
                    0.05,
                    key="v3_cost_weight",
                )
                build_cost_weight_v3 = cost_weight_v3

                dest_in_view_v3 = sorted(regionals_v3_df["Destination"].unique())
                if st.session_state.get("destination_weights_v3_version") != "city_counts_v1":
                    st.session_state.destination_weights_v3 = initial_destination_weights_by_city(
                        regionals_v3_df,
                        V3_CITY_COUNTS,
                    )
                    st.session_state.destination_weights_v3_version = "city_counts_v1"

                dest_weights_v3 = destination_weights(
                    dest_in_view_v3,
                    st.session_state.destination_weights_v3,
                )
                missing_weight_keys = []
                weight_keys = set(V3_CITY_COUNTS.keys())
                for _, row in regionals_v3_df[["ToCity", "ToState"]].dropna().drop_duplicates().iterrows():
                    key = f"{str(row['ToCity']).strip().casefold()}, {str(row['ToState']).strip().casefold()}"
                    if key not in weight_keys:
                        missing_weight_keys.append(f"{row['ToCity']}, {row['ToState']}")
                if missing_weight_keys:
                    st.warning(
                        f"Missing weights for {len(missing_weight_keys)} destinations.",
                        icon="⚠️",
                    )
                    st.text(", ".join(sorted(missing_weight_keys)))
                else:
                    st.caption("All destinations in page3.csv have weights.")

                avg_time_by_origin_v3 = (
                    regionals_v3_df.groupby(["FromAddress", "Destination"], as_index=False)
                    .agg(AvgTime=("ShippingTimeDays", "mean"))
                )
                avg_time_by_origin_v3["Weight"] = avg_time_by_origin_v3["Destination"].map(
                    dest_weights_v3
                ).fillna(0.0)
                weighted_avg_time_by_origin_v3 = (
                    avg_time_by_origin_v3.groupby("FromAddress")
                    .apply(lambda g: weighted_mean(g["AvgTime"].to_numpy(), g["Weight"].to_numpy()))
                    .reset_index(name="WeightedAvgTime")
                )
                if not weighted_avg_time_by_origin_v3.empty:
                    best_time_origin = weighted_avg_time_by_origin_v3.sort_values(
                        "WeightedAvgTime"
                    ).iloc[0]
                    st.metric(
                        "Best origin by weighted avg time (page3)",
                        best_time_origin["FromAddress"],
                        f"{best_time_origin['WeightedAvgTime']:.2f} days",
                    )

                st.markdown("Build network (page3)")
                built_network_origins_v3 = st.multiselect(
                    "Included origins (page3)",
                    sorted(regionals_v3_df["FromAddress"].unique()),
                    default=[],
                    key="page3_built_origins",
                )
                built_subset_v3 = regionals_v3_df[
                    regionals_v3_df["FromAddress"].isin(built_network_origins_v3)
                ]
                if built_subset_v3.empty:
                    st.caption("No origins selected for the page3 network.")
                else:
                    built_best_time_v3 = built_subset_v3.groupby("Destination", as_index=False).agg(
                        BestTime=("ShippingTimeDays", "min")
                    )
                    built_best_time_v3["Weight"] = built_best_time_v3["Destination"].map(
                        dest_weights_v3
                    ).fillna(0.0)
                    built_avg_time_v3 = weighted_mean(
                        built_best_time_v3["BestTime"].to_numpy(),
                        built_best_time_v3["Weight"].to_numpy(),
                    )
                    built_best_cost_v3 = built_subset_v3.groupby("Destination", as_index=False).agg(
                        BestCost=("Cost", "min")
                    )
                    built_best_cost_v3["Weight"] = built_best_cost_v3["Destination"].map(
                        dest_weights_v3
                    ).fillna(0.0)
                    built_avg_cost_v3 = weighted_mean(
                        built_best_cost_v3["BestCost"].to_numpy(),
                        built_best_cost_v3["Weight"].to_numpy(),
                    )
                    col_a, col_b = st.columns(2)
                    col_a.metric("Built network avg time (page3)", f"{built_avg_time_v3:.2f}")
                    col_b.metric("Built network avg cost (page3)", f"{built_avg_cost_v3:.2f}")
                    slow_1_v3 = built_best_time_v3[built_best_time_v3["BestTime"] > 1.0][
                        "Destination"
                    ].sort_values().tolist()
                    slow_2_v3 = built_best_time_v3[built_best_time_v3["BestTime"] > 2.0][
                        "Destination"
                    ].sort_values().tolist()
                    slow_3_v3 = built_best_time_v3[built_best_time_v3["BestTime"] > 3.0][
                        "Destination"
                    ].sort_values().tolist()
                    st.markdown("Cities without 1-day shipping (page3)")
                    if slow_1_v3:
                        st.selectbox("Destinations > 1 day (page3)", slow_1_v3, key="page3_slow_1")
                    else:
                        st.caption("All destinations have 1-day shipping in the page3 network.")
                    st.markdown("Cities without 2-day shipping (page3)")
                    if slow_2_v3:
                        st.selectbox("Destinations > 2 days (page3)", slow_2_v3, key="page3_slow_2")
                    else:
                        st.caption("All destinations have 2-day shipping in the page3 network.")
                    st.markdown("Cities without 3-day shipping (page3)")
                    if slow_3_v3:
                        st.selectbox("Destinations > 3 days (page3)", slow_3_v3, key="page3_slow_3")
                    else:
                        st.caption("All destinations have 3-day shipping in the page3 network.")

                if baseline_origin_v3 not in regionals_v3_df["FromAddress"].unique():
                    baseline_origin_v3 = origin_list_all_v3[0]

                baseline_time_by_dest_v3 = regionals_v3_df[
                    regionals_v3_df["FromAddress"] == baseline_origin_v3
                ].groupby("Destination", as_index=False).agg(AvgTime=("ShippingTimeDays", "mean"))
                baseline_time_by_dest_v3["Weight"] = baseline_time_by_dest_v3["Destination"].map(
                    dest_weights_v3
                ).fillna(0.0)
                avg_time_v3 = weighted_mean(
                    baseline_time_by_dest_v3["AvgTime"].to_numpy(),
                    baseline_time_by_dest_v3["Weight"].to_numpy(),
                )
                baseline_cost_by_dest_v3 = regionals_v3_df[
                    regionals_v3_df["FromAddress"] == baseline_origin_v3
                ].groupby("Destination", as_index=False).agg(AvgCost=("Cost", "mean"))
                baseline_cost_by_dest_v3["Weight"] = baseline_cost_by_dest_v3["Destination"].map(
                    dest_weights_v3
                ).fillna(0.0)
                avg_cost_v3 = weighted_mean(
                    baseline_cost_by_dest_v3["AvgCost"].to_numpy(),
                    baseline_cost_by_dest_v3["Weight"].to_numpy(),
                )

                one_day_origins_v3 = sorted(
                    regionals_v3_df[regionals_v3_df["ShippingTimeDays"] <= 1.0][
                        "FromAddress"
                    ].unique().tolist()
                )
                avg_time_by_origin_v3 = regionals_v3_df.groupby("FromAddress", as_index=False).agg(
                    AvgTime=("ShippingTimeDays", "mean")
                )
                baseline_avg_time_v3 = float(
                    avg_time_by_origin_v3[
                        avg_time_by_origin_v3["FromAddress"] == baseline_origin_v3
                    ]["AvgTime"].iloc[0]
                ) if baseline_origin_v3 in avg_time_by_origin_v3["FromAddress"].values else float("nan")
                avg_time_by_origin_v3["TimeReductionVsBaseline"] = (
                    baseline_avg_time_v3 - avg_time_by_origin_v3["AvgTime"]
                )
                top_time_reducers_v3 = (
                    avg_time_by_origin_v3.sort_values("TimeReductionVsBaseline", ascending=False)
                    .head(10)["FromAddress"]
                    .tolist()
                )
                origin_candidates_v3 = sorted(
                    {baseline_origin_v3} | set(one_day_origins_v3) | set(top_time_reducers_v3)
                )
                origin_list_v3 = [o for o in origin_candidates_v3 if o in origin_list_all_v3]

                major_destinations_v3 = [
                    dest
                    for dest in dest_in_view_v3
                    if float(st.session_state.destination_weights_v3.get(dest, DEFAULT_DEST_WEIGHT)) > 1.0
                ]
                major_weight_map_v3 = {
                    dest: float(st.session_state.destination_weights_v3.get(dest, DEFAULT_DEST_WEIGHT))
                    for dest in major_destinations_v3
                }

                render_top_combos(
                    regionals_v3_df,
                    origin_list_v3,
                    dest_in_view_v3,
                    dest_weights_v3,
                    build_cost_weight_v3,
                    avg_cost_v3,
                    avg_time_v3,
                    major_weight_map_v3,
                    "regionals_v3",
                    show_day_percentages=True,
                    max_k=20,
                    baseline_origin=baseline_origin_v3,
                )
    else:
        st.info("page3.csv not found in the app folder.")
