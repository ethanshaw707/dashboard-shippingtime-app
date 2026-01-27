import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import polars as pl
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
DESTINATION_WEIGHTS = {
    "New York, NY": 8.0,
    "Brooklyn, NY": 6.0,
    "Los Angeles, CA": 6.0,
    "Chicago, IL": 6.0,
    "Houston, TX": 5.0,
    "Atlanta, GA": 5.0,
    "Phoenix, AZ": 4.0,
    "Dallas, TX": 4.0,
    "San Francisco, CA": 4.0,
    "Miami, FL": 3.0,
    "Seattle, WA": 3.0,
    "Denver, CO": 3.0,
    "San Diego, CA": 3.0,
    "Austin, TX": 1.6,
    "San Jose, CA": 1.6,
    "San Antonio, TX": 1.4,
    "Charlotte, NC": 1.3,
    "Orlando, FL": 1.2,
    "Las Vegas, NV": 1.2,
    "Nashville, TN": 1.2,
    "Tampa, FL": 1.1,
    "Raleigh, NC": 1.1,
    "Portland, OR": 1.1,
    "Jacksonville, FL": 1.0,
    "Knoxville, TN": 1.0,
    "Louisville, KY": 1.0,
    "Kansas City, MO": 1.0,
    "Saint Louis, MO": 1.0,
    "Indianapolis, IN": 1.0,
    "Albuquerque, NM": 0.9,
    "Fort Worth, TX": 0.9,
    "Oklahoma City, OK": 0.8,
    "Colorado Springs, CO": 0.7,
    "Pittsburgh, PA": 0.7,
    "Tucson, AZ": 0.7,
    "Reno, NV": 0.6,
    "Boise, ID": 0.6,
    "Mesa, AZ": 0.6,
    "Omaha, NE": 0.6,
    "Milwaukee, WI": 0.6,
    "Virginia Beach, VA": 0.6,
    "Gilbert, AZ": 0.5,
    "Rochester, NY": 0.5,
    "Columbia, MO": 0.5,
    "Chesapeake, VA": 0.5,
    "Marietta, GA": 0.5,
    "Canton, GA": 0.4,
    "Franklin, TN": 0.4,
    "Marble Falls, TX": 0.3,
    "Bismarck, ND": 0.3,
}
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
tab_dashboard, tab_regionals = st.tabs(["Dashboard", "Shipping Summary Regionals"])

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
    if "destination_weights" not in st.session_state:
        st.session_state.destination_weights = {**DESTINATION_WEIGHTS}
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
    st.markdown("Recommended starting two (synergy pairs)")
    if "show_top_combos" not in st.session_state:
        st.session_state.show_top_combos = False
    if st.button("Compute top combos", key="compute_top_combos"):
        st.session_state.show_top_combos = True
    
    if st.session_state.show_top_combos:
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
            avg_cost = float(np.sum(best_cost[valid_mask] * weights) / weights.sum())
            avg_time = float(np.sum(best_time[valid_mask] * weights) / weights.sum())
            return avg_cost, avg_time
    
        def combo_weighted_total(combo_indices):
            avg_cost, avg_time = combo_avg_cost_time(combo_indices)
            if not np.isfinite(avg_cost) or not np.isfinite(avg_time):
                return float("nan"), avg_cost, avg_time
            weighted_total = build_cost_weight * avg_cost + (1 - build_cost_weight) * avg_time
            return weighted_total, avg_cost, avg_time
    
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
            selected_pair = st.selectbox("Top 5 pairs (weighted)", pair_labels, index=0, key="top5_pairs")
            selected_pair_name = selected_pair.rsplit(" (", 1)[0]
            selected_row = pair_df[pair_df["Pair"] == selected_pair_name].iloc[0]
            combo_indices = [origin_list.index(name) for name in selected_pair_name.split(" + ")]
            combo_avg_cost, combo_avg_time = combo_avg_cost_time(combo_indices)
            cost_delta = avg_cost - combo_avg_cost if np.isfinite(combo_avg_cost) else float("nan")
            time_delta = avg_time - combo_avg_time if np.isfinite(combo_avg_time) else float("nan")
            prev_combo = best_combo_indices(1)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - combo_avg_cost
                prev_time_delta = prev_time - combo_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(combo_indices, prev_combo or [])
            st.caption(f"Weighted total: {selected_row['WeightedTotal']:.2f} - Avg cost: {combo_avg_cost:.2f} (delta {cost_delta:.2f}) - Avg time: {combo_avg_time:.2f} (delta {time_delta:.2f}) - Major coverage: {selected_row['MajorCoveragePct']:.1f}%")
            st.caption(f"Delta vs best 1 origin: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            pair_improved = combo_improved_cities(prev_combo or [], combo_indices)
            st.selectbox("Improved destinations vs best 1 origin (pair)", pair_improved or ["None"], key="pair_no1_day")
        else:
            st.caption("Not enough origins to calculate pairs.")
    
        st.markdown("Top 5 trios")
        trio_list = (compute_top_k_combos_cached(df, tuple(origin_list), tuple(dest_in_view), 3, build_cost_weight, dest_weights_items) if len(origin_list) >= 3 else [])
        if trio_list:
            trio_labels = [f"{name} ({value:.2f})" for name, value in trio_list]
            selected_trio = st.selectbox("Top 5 trios (weighted)", trio_labels, index=0, key="top5_trios")
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
            st.caption(f"Avg cost: {trio_avg_cost:.2f} (delta {avg_cost - trio_avg_cost:.2f}) - Avg time: {trio_avg_time:.2f} (delta {avg_time - trio_avg_time:.2f})")
            st.caption(f"Delta vs best 2 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            trio_improved = combo_improved_cities(prev_combo or [], trio_indices)
            st.selectbox("Improved destinations vs best 2 origins (trio)", trio_improved or ["None"], key="trio_no1_day")
        else:
            st.caption("Not enough origins to calculate trios.")
    
        st.markdown("Top 5 quads")
        quad_list = (compute_top_k_combos_cached(df, tuple(origin_list), tuple(dest_in_view), 4, build_cost_weight, dest_weights_items) if len(origin_list) >= 4 else [])
        if quad_list:
            quad_labels = [f"{name} ({value:.2f})" for name, value in quad_list]
            selected_quad = st.selectbox("Top 5 quads (weighted)", quad_labels, index=0, key="top5_quads")
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
            st.caption(f"Avg cost: {quad_avg_cost:.2f} (delta {avg_cost - quad_avg_cost:.2f}) - Avg time: {quad_avg_time:.2f} (delta {avg_time - quad_avg_time:.2f})")
            st.caption(f"Delta vs best 3 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            quad_improved = combo_improved_cities(prev_combo or [], quad_indices)
            st.selectbox("Improved destinations vs best 3 origins (quad)", quad_improved or ["None"], key="quad_no1_day")
        else:
            st.caption("Not enough origins to calculate quads.")
    
        st.markdown("Top 5 (5 locations)")
        five_list = (compute_top_k_combos_cached(df, tuple(origin_list), tuple(dest_in_view), 5, build_cost_weight, dest_weights_items) if len(origin_list) >= 5 else [])
        common_origins = []
        if five_list:
            five_labels = [f"{name} ({value:.2f})" for name, value in five_list]
            selected_five = st.selectbox("Top 5 (5 locations) (weighted)", five_labels, index=0, key="top5_fives")
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
            st.caption(f"Avg cost: {five_avg_cost:.2f} (delta {avg_cost - five_avg_cost:.2f}) - Avg time: {five_avg_time:.2f} (delta {avg_time - five_avg_time:.2f})")
            st.caption(f"Delta vs best 4 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            five_origin_sets = [set(name.split(" + ")) for name, _ in five_list]
            common_origins = sorted(set.intersection(*five_origin_sets)) if five_origin_sets else []
            if common_origins:
                st.caption(f"Auto-included origins for 6/7: {', '.join(common_origins)}")
            five_improved = combo_improved_cities(prev_combo or [], five_indices)
            st.selectbox("Improved destinations vs best 4 origins (5 locations)", five_improved or ["None"], key="five_no1_day")
        else:
            st.caption("Not enough origins to calculate 5-location combos.")
    
        st.markdown("Top 5 (6 locations)")
        required_origins = tuple(common_origins) if common_origins else tuple()
        six_list = (compute_top_k_combos_with_required_cached(df, tuple(origin_list), tuple(dest_in_view), required_origins, 6, build_cost_weight, dest_weights_items) if len(origin_list) >= 6 else [])
        common_origins6 = []
        if six_list:
            six_labels = [f"{name} ({value:.2f})" for name, value in six_list]
            selected_six = st.selectbox("Top 5 (6 locations) (weighted)", six_labels, index=0, key="top5_sixes")
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
            st.caption(f"Avg cost: {six_avg_cost:.2f} (delta {avg_cost - six_avg_cost:.2f}) - Avg time: {six_avg_time:.2f} (delta {avg_time - six_avg_time:.2f})")
            st.caption(f"Delta vs best 5 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            six_origin_sets = [set(name.split(" + ")) for name, _ in six_list]
            common_origins6 = sorted(set.intersection(*six_origin_sets)) if six_origin_sets else []
            if common_origins6:
                st.caption(f"Auto-included origins for 7/8: {', '.join(common_origins6)}")
            six_improved = combo_improved_cities(prev_combo or [], six_indices)
            st.selectbox("Improved destinations vs best 5 origins (6 locations)", six_improved or ["None"], key="six_no1_day")
        else:
            st.caption("Not enough origins to calculate 6-location combos.")
    
        st.markdown("Top 5 (7 locations)")
        req7 = tuple(common_origins6) if common_origins6 else required_origins
        seven_list = (compute_top_k_combos_with_required_cached(df, tuple(origin_list), tuple(dest_in_view), req7, 7, build_cost_weight, dest_weights_items) if len(origin_list) >= 7 else [])
        common_origins7 = []
        if seven_list:
            seven_labels = [f"{name} ({value:.2f})" for name, value in seven_list]
            selected_seven = st.selectbox("Top 5 (7 locations) (weighted)", seven_labels, index=0, key="top5_sevens")
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
            st.caption(f"Avg cost: {seven_avg_cost:.2f} (delta {avg_cost - seven_avg_cost:.2f}) - Avg time: {seven_avg_time:.2f} (delta {avg_time - seven_avg_time:.2f})")
            st.caption(f"Delta vs best 6 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            seven_origin_sets = [set(name.split(" + ")) for name, _ in seven_list]
            common_origins7 = sorted(set.intersection(*seven_origin_sets)) if seven_origin_sets else []
            if common_origins7:
                st.caption(f"Auto-included origins for 8/9: {', '.join(common_origins7)}")
            seven_improved = combo_improved_cities(prev_combo or [], seven_indices)
            st.selectbox("Improved destinations vs best 6 origins (7 locations)", seven_improved or ["None"], key="seven_no1_day")
        else:
            st.caption("Not enough origins to calculate 7-location combos.")
    
        st.markdown("Top 5 (8 locations)")
        req8 = tuple(common_origins7) if common_origins7 else tuple()
        eight_list = (compute_top_k_combos_with_required_cached(df, tuple(origin_list), tuple(dest_in_view), req8, 8, build_cost_weight, dest_weights_items) if len(origin_list) >= 8 else [])
        common_origins8 = []
        if eight_list:
            eight_labels = [f"{name} ({value:.2f})" for name, value in eight_list]
            selected_eight = st.selectbox("Top 5 (8 locations) (weighted)", eight_labels, index=0, key="top5_eights")
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
            st.caption(f"Avg cost: {eight_avg_cost:.2f} (delta {avg_cost - eight_avg_cost:.2f}) - Avg time: {eight_avg_time:.2f} (delta {avg_time - eight_avg_time:.2f})")
            st.caption(f"Delta vs best 7 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            eight_origin_sets = [set(name.split(" + ")) for name, _ in eight_list]
            common_origins8 = sorted(set.intersection(*eight_origin_sets)) if eight_origin_sets else []
            if common_origins8:
                st.caption(f"Auto-included origins for 9/10: {', '.join(common_origins8)}")
            eight_improved = combo_improved_cities(prev_combo or [], eight_indices)
            st.selectbox("Improved destinations vs best 7 origins (8 locations)", eight_improved or ["None"], key="eight_no1_day")
        else:
            st.caption("Not enough origins to calculate 8-location combos.")
    
        st.markdown("Top 5 (9 locations)")
        req9 = tuple(common_origins8) if common_origins8 else tuple()
        nine_list = (compute_top_k_combos_with_required_cached(df, tuple(origin_list), tuple(dest_in_view), req9, 9, build_cost_weight, dest_weights_items) if len(origin_list) >= 9 else [])
        common_origins9 = []
        if nine_list:
            nine_labels = [f"{name} ({value:.2f})" for name, value in nine_list]
            selected_nine = st.selectbox("Top 5 (9 locations) (weighted)", nine_labels, index=0, key="top5_nines")
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
            st.caption(f"Avg cost: {nine_avg_cost:.2f} (delta {avg_cost - nine_avg_cost:.2f}) - Avg time: {nine_avg_time:.2f} (delta {avg_time - nine_avg_time:.2f})")
            st.caption(f"Delta vs best 8 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            nine_origin_sets = [set(name.split(" + ")) for name, _ in nine_list]
            common_origins9 = sorted(set.intersection(*nine_origin_sets)) if nine_origin_sets else []
            if common_origins9:
                st.caption(f"Auto-included origins for 10: {', '.join(common_origins9)}")
            nine_improved = combo_improved_cities(prev_combo or [], nine_indices)
            st.selectbox("Improved destinations vs best 8 origins (9 locations)", nine_improved or ["None"], key="nine_no1_day")
        else:
            st.caption("Not enough origins to calculate 9-location combos.")
    
        st.markdown("Top 5 (10 locations)")
        req10 = tuple(common_origins9) if common_origins9 else tuple()
        ten_list = (compute_top_k_combos_with_required_cached(df, tuple(origin_list), tuple(dest_in_view), req10, 10, build_cost_weight, dest_weights_items) if len(origin_list) >= 10 else [])
        if ten_list:
            ten_labels = [f"{name} ({value:.2f})" for name, value in ten_list]
            selected_ten = st.selectbox("Top 5 (10 locations) (weighted)", ten_labels, index=0, key="top5_tens")
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
            st.caption(f"Avg cost: {ten_avg_cost:.2f} (delta {avg_cost - ten_avg_cost:.2f}) - Avg time: {ten_avg_time:.2f} (delta {avg_time - ten_avg_time:.2f})")
            st.caption(f"Delta vs best 9 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            ten_improved = combo_improved_cities(prev_combo or [], ten_indices)
            st.selectbox("Improved destinations vs best 9 origins (10 locations)", ten_improved or ["None"], key="ten_no1_day")
        else:
            st.caption("Not enough origins to calculate 10-location combos.")
    else:
        st.caption('Top combos are hidden until you click "Compute top combos".')
    
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
            dest_weights_size = destination_weights(dest_in_view, DESTINATION_WEIGHTS)
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

with tab_regionals:
    st.subheader("Shipping Summary Regionals")
    st.caption("In-depth analysis of regional from cities.")
    regionals_path = Path("shipping_summary_regionals.csv")
    if regionals_path.exists():
        regionals_df = load_data(str(regionals_path))
        if regionals_df.empty:
            st.info("No data available in shipping_summary_regionals.csv.")
        else:
            st.metric("Rows", f"{len(regionals_df):,}")
            from_cities = sorted(regionals_df["FromAddress"].unique())
            selected_from = st.selectbox("From city", from_cities, index=0)
            subset = regionals_df[regionals_df["FromAddress"] == selected_from]
            if subset.empty:
                st.info("No routes found for the selected origin.")
            else:
                st.subheader(f"{selected_from} summary")
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Avg cost", f"{subset["Cost"].mean():.2f}")
                kpi_cols[1].metric("Avg shipping days", f"{subset["ShippingTimeDays"].mean():.2f}")
                kpi_cols[2].metric("Destinations", f"{subset["Destination"].nunique():,}")
                kpi_cols[3].metric("Rows", f"{len(subset):,}")
                st.markdown("Build a network (regionals)")
                regional_origins = sorted(regionals_df["FromAddress"].unique())
                regional_destinations = sorted(regionals_df["Destination"].unique())
                regional_weights_map = st.session_state.destination_weights if "destination_weights" in st.session_state else DESTINATION_WEIGHTS
                regional_dest_weights = destination_weights(regional_destinations, regional_weights_map)
                regional_built = st.multiselect(
                    "Included origins (regionals)",
                    regional_origins,
                    default=[],
                    key="regionals_built_origins",
                )
                regional_subset = regionals_df[regionals_df["FromAddress"].isin(regional_built)]
                if regional_subset.empty:
                    st.caption("No origins selected for the regional network.")
                else:
                    built_best_time = regional_subset.groupby("Destination", as_index=False).agg(
                        BestTime=("ShippingTimeDays", "min")
                    )
                    built_best_time["Weight"] = built_best_time["Destination"].map(regional_dest_weights).fillna(0.0)
                    built_avg_time = weighted_mean(
                        built_best_time["BestTime"].to_numpy(),
                        built_best_time["Weight"].to_numpy(),
                    )
                    built_best_cost = regional_subset.groupby("Destination", as_index=False).agg(
                        BestCost=("Cost", "min")
                    )
                    built_best_cost["Weight"] = built_best_cost["Destination"].map(regional_dest_weights).fillna(0.0)
                    built_avg_cost = weighted_mean(
                        built_best_cost["BestCost"].to_numpy(),
                        built_best_cost["Weight"].to_numpy(),
                    )
                    reg_col1, reg_col2 = st.columns(2)
                    reg_col1.metric("Built network avg time", f"{built_avg_time:.2f}")
                    reg_col2.metric("Built network avg cost", f"{built_avg_cost:.2f}")
                    slow_1 = built_best_time[built_best_time["BestTime"] > 1.0]["Destination"].sort_values().tolist()
                    slow_2 = built_best_time[built_best_time["BestTime"] > 2.0]["Destination"].sort_values().tolist()
                    slow_3 = built_best_time[built_best_time["BestTime"] > 3.0]["Destination"].sort_values().tolist()
                    st.markdown("Cities without 1-day shipping (regionals)")
                    if slow_1:
                        st.selectbox("Destinations > 1 day", slow_1, key="regionals_slow_1")
                    else:
                        st.caption("All destinations have 1-day shipping in the regional network.")
                    st.markdown("Cities without 2-day shipping (regionals)")
                    if slow_2:
                        st.selectbox("Destinations > 2 days", slow_2, key="regionals_slow_2")
                    else:
                        st.caption("All destinations have 2-day shipping in the regional network.")
                    st.markdown("Cities without 3-day shipping (regionals)")
                    if slow_3:
                        st.selectbox("Destinations > 3 days", slow_3, key="regionals_slow_3")
                    else:
                        st.caption("All destinations have 3-day shipping in the regional network.")
                by_dest = subset.groupby("Destination", as_index=False).agg(
                    AvgCost=("Cost", "mean"),
                    AvgTime=("ShippingTimeDays", "mean"),
                    Rows=("Destination", "count"),
                )
                st.markdown("Average cost by destination")
                st.altair_chart(
                    alt.Chart(by_dest).mark_bar().encode(
                        x=alt.X("AvgCost:Q", title="Avg Cost"),
                        y=alt.Y("Destination:N", sort="-x", axis=alt.Axis(labelLimit=200)),
                        tooltip=["Destination", "AvgCost", "Rows"],
                    ).properties(height=max(300, len(by_dest) * 18)),
                    use_container_width=True,
                )
                st.markdown("Average shipping days by destination")
                st.altair_chart(
                    alt.Chart(by_dest).mark_bar().encode(
                        x=alt.X("AvgTime:Q", title="Avg Shipping Days"),
                        y=alt.Y("Destination:N", sort="-x", axis=alt.Axis(labelLimit=200)),
                        tooltip=["Destination", "AvgTime", "Rows"],
                    ).properties(height=max(300, len(by_dest) * 18)),
                    use_container_width=True,
                )
                st.markdown("Cost vs time")
                st.altair_chart(
                    alt.Chart(by_dest).mark_circle(size=120).encode(
                        x=alt.X("AvgCost:Q", title="Avg Cost"),
                        y=alt.Y("AvgTime:Q", title="Avg Shipping Days"),
                        tooltip=["Destination", "AvgCost", "AvgTime", "Rows"],
                    ).properties(height=320),
                    use_container_width=True,
                )
                with st.expander("Show data for selected origin"):
                    st.dataframe(subset, use_container_width=True)
    else:
        st.info("shipping_summary_regionals.csv not found in the app folder.")
