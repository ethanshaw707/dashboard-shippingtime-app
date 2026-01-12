import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path


DATASETS = {
    "Small": {
        "Monday": "Shipping Location Analysis - Small - Monday.csv",
        "Friday": "Shipping Location Analysis - Small - Friday.csv",
    },
    "Medium": {
        "Monday": "Shipping Location Analysis - Medium - Monday.csv",
        "Friday": "Shipping Location Analysis - Medium - Friday.csv",
    },
    "Large": {
        "Monday": "Shipping Location Analysis - Large - Monday.csv",
        "Friday": "Shipping Location Analysis - Large - Friday.csv",
    },
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    for col in ["Driving (Estimate)", "Straight Line", "Cost", "ShippingTimeDays"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Destination"] = df["ToCity"].str.strip() + ", " + df["ToState"].str.strip()
    return df


@st.cache_data
def load_day_data(day: str, size: str | None = None) -> pd.DataFrame:
    if size:
        return load_data(DATASETS[size][day])
    frames = [load_data(DATASETS[size_key][day]) for size_key in DATASETS]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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

def normalize_destination_weights(destinations, pct_map: dict) -> dict:
    weights = {dest: float(pct_map.get(dest, 0.0)) for dest in destinations}
    total = sum(weights.values())
    if total == 0:
        equal = 1.0 / max(len(destinations), 1)
        return {dest: equal for dest in destinations}
    return {dest: val / total for dest, val in weights.items()}


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

tab_dashboard, tab_regression = st.tabs(["Dashboard", "Shipping Time Model"])

with tab_dashboard:

    shipping_day = st.sidebar.selectbox("Date You are Shipping", ["Monday", "Friday"], index=0)
    dataset_name = st.sidebar.radio("Page", list(DATASETS.keys()), index=0)
    data_path = Path(DATASETS[dataset_name][shipping_day])
    df_raw = load_data(str(data_path))

    cities = sorted(df_raw["ToCity"].unique())
    origins = sorted(df_raw["FromAddress"].unique())

    default_baseline = "Indianapolis" if "Indianapolis" in df_raw["FromAddress"].unique() else sorted(df_raw["FromAddress"].unique())[0]
    baseline_origin = st.sidebar.selectbox("Baseline origin", sorted(df_raw["FromAddress"].unique()), index=sorted(df_raw["FromAddress"].unique()).index(default_baseline))

    cost_weight = st.sidebar.slider("Cost weight (Time weight = 1 - Cost)", 0.0, 1.0, 0.6, 0.05)
    signs_per_month = st.sidebar.number_input("Signs sold per month", min_value=0, value=500, step=50)
    build_cost_weight = cost_weight

    st.sidebar.markdown("Destination share (%)")
    destination_list = sorted(df_raw["Destination"].unique())
    dest_pct_map = {}
    default_state_pct = {
        "California": 17.00,
        "Texas": 15.32,
        "Florida": 11.57,
        "New York": 11.21,
        "Pennsylvania": 8.88,
        "North Carolina": 8.08,
        "Ohio": 7.79,
        "Georgia": 7.21,
        "Illinois": 6.74,
        "Virginia": 6.40,
    }
    with st.sidebar.expander("Signs by destination", expanded=False):
        for i, dest in enumerate(destination_list):
            state = dest.split(", ", 1)[1] if ", " in dest else dest
            dest_pct_map[dest] = st.number_input(
                dest,
                min_value=0.0,
                max_value=100.0,
                value=float(default_state_pct.get(state, 0.0)),
                step=0.01,
                format="%.2f",
                key=f"dest_pct_{i}",
            )

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
    dest_weights = normalize_destination_weights(dest_in_view, dest_pct_map)
    st.sidebar.caption(f"Destination shares normalized across {len(dest_in_view)} destinations in view.")

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
    avg_cost = float(df["Cost"].mean())
    best_cost_by_dest = df.groupby("Destination", as_index=False).agg(BestCost=("Cost", "min"))
    best_cost_by_dest["Weight"] = best_cost_by_dest["Destination"].map(dest_weights).fillna(0.0)
    optimal_cost_avg = weighted_mean(best_cost_by_dest["BestCost"].to_numpy(), best_cost_by_dest["Weight"].to_numpy())
    cost_savings_pct = ((avg_cost - optimal_cost_avg) / avg_cost * 100.0) if avg_cost else 0.0

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
    col1.metric("Avg cost", f"{avg_cost:.2f}")
    col2.metric("Avg shipping days", f"{avg_time:.2f}")
    col3.metric("Optimal avg time", f"{optimal_time_avg:.2f}")
    col4.metric("Optimal avg cost", f"{optimal_cost_avg:.2f}")
    col5.metric("% time saved vs optimal", f"{time_savings_pct:.1f}%")
    col6.metric("% cost saved vs optimal", f"{cost_savings_pct:.1f}%")
    best_cost = df.groupby("FromAddress")["Cost"].mean().idxmin()
    best_time = df.groupby("FromAddress")["ShippingTimeDays"].mean().idxmin()
    best_weighted = df.groupby("FromAddress")["WeightedScore"].mean().idxmin()
    col7.metric("Best origin (cost)", best_cost)
    col8.metric("Best origin (time)", best_time)
    col9.metric("Best origin (weighted)", best_weighted)

    st.markdown("Built Network")
    built_network_origins = st.multiselect(
        "Included origins",
        sorted(df["FromAddress"].unique()),
        default=sorted(df["FromAddress"].unique()),
    )
    built_subset = df[df["FromAddress"].isin(built_network_origins)]
    if built_subset.empty:
        built_optimal_time_avg = float("nan")
        built_optimal_cost_avg = float("nan")
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
    col10, col11 = st.columns(2)
    col10.metric("Built network avg time", f"{built_optimal_time_avg:.2f}" if np.isfinite(built_optimal_time_avg) else "n/a")
    col11.metric("Built network avg cost", f"{built_optimal_cost_avg:.2f}" if np.isfinite(built_optimal_cost_avg) else "n/a")

    st.markdown("---")

    avg_by_origin = df.groupby("FromAddress", as_index=False).agg(
        AvgCost=("Cost", "mean"),
        AvgTime=("ShippingTimeDays", "mean"),
        AvgWeighted=("WeightedScore", "mean"),
        Destinations=("Destination", "nunique"),
    )

    origin_count = avg_by_origin.shape[0]
    bar_height = max(300, origin_count * 34)

    bar_cost = alt.Chart(avg_by_origin).mark_bar().encode(
        x=alt.X("AvgCost:Q", title="Avg Cost"),
        y=alt.Y("FromAddress:N", sort="-x", axis=alt.Axis(labelLimit=200, labelOverlap=False, labelAngle=0)),
        tooltip=["FromAddress", "AvgCost", "Destinations"],
    ).properties(height=bar_height, title="Average cost by origin")

    bar_time = alt.Chart(avg_by_origin).mark_bar().encode(
        x=alt.X("AvgTime:Q", title="Avg Shipping Days"),
        y=alt.Y("FromAddress:N", sort="-x", axis=alt.Axis(labelLimit=200, labelOverlap=False, labelAngle=0)),
        tooltip=["FromAddress", "AvgTime", "Destinations"],
    ).properties(height=bar_height, title="Average shipping time by origin")

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
    st.caption("Average cost by origin. Lower bars indicate cheaper shipping for the selected destinations.")
    st.altair_chart(bar_cost, use_container_width=True)
    st.caption("Average shipping time by origin. Lower bars indicate faster delivery for the selected destinations.")
    st.altair_chart(bar_time, use_container_width=True)
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

with tab_regression:
    st.subheader("Shipping time model")
    st.caption("Linear regression using driving and straight-line miles to predict shipping days.")
    model_size = st.selectbox("Size for model", list(DATASETS.keys()), index=0)
    monday_df = load_day_data("Monday", model_size)
    friday_df = load_day_data("Friday", model_size)

    st.markdown("City to City")
    city_day = st.selectbox("City to city day", ["Monday", "Friday"], index=0, key="city_day_regression")
    city_df = monday_df if city_day == "Monday" else friday_df
    if city_df.empty:
        st.info("No data available for the selected day.")
    else:
        city_col1, city_col2, city_col3 = st.columns([2, 2, 5])
        with city_col1:
            city_from = st.selectbox(
                "From city",
                sorted(city_df["FromAddress"].unique()),
                key="city_from_regression",
            )
        with city_col2:
            city_to = st.selectbox(
                "To city",
                sorted(city_df["ToCity"].unique()),
                key="city_to_regression",
            )
        city_pair = city_df[(city_df["FromAddress"] == city_from) & (city_df["ToCity"] == city_to)]
        if city_pair.empty:
            st.info("No matching route found for the selected cities.")
        else:
            drive_miles = float(city_pair["Driving (Estimate)"].mean())
            ship_days = float(city_pair["ShippingTimeDays"].mean())
            ship_cost = float(city_pair["Cost"].mean())
            city_kpi1, city_kpi2, city_kpi3 = st.columns(3)
            city_kpi1.metric("Driving miles", f"{drive_miles:.0f}")
            city_kpi2.metric("Shipping days", f"{ship_days:.2f}")
            city_kpi3.metric("Cost", f"{ship_cost:.2f}")

    def fit_linear_model(df: pd.DataFrame, target_col: str):
        if df.empty:
            return None, float("nan")
        clean = df[["Driving (Estimate)", "Straight Line", target_col]].astype(float)
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return None, float("nan")
        base_features = clean[["Driving (Estimate)", "Straight Line"]].to_numpy()
        drive_sq = (base_features[:, 0] ** 2).reshape(-1, 1)
        drive_cu = (base_features[:, 0] ** 3).reshape(-1, 1)
        features = np.column_stack([base_features, drive_sq, drive_cu])
        target = clean[target_col].to_numpy()
        design = np.column_stack([np.ones(len(features)), features])
        coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        pred = design @ coeffs
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot else float("nan")
        return coeffs, r2

    monday_coeffs, monday_r2 = fit_linear_model(monday_df, "ShippingTimeDays")
    friday_coeffs, friday_r2 = fit_linear_model(friday_df, "ShippingTimeDays")
    monday_time_bins = np.sort(monday_df["ShippingTimeDays"].astype(float).unique())
    friday_time_bins = np.sort(friday_df["ShippingTimeDays"].astype(float).unique())

    model_cols = st.columns(4)
    model_cols[0].metric("Monday R2", f"{monday_r2:.3f}" if np.isfinite(monday_r2) else "n/a")
    model_cols[1].metric("Friday R2", f"{friday_r2:.3f}" if np.isfinite(friday_r2) else "n/a")
    model_cols[2].metric("Monday rows", f"{len(monday_df):,}")
    model_cols[3].metric("Friday rows", f"{len(friday_df):,}")

    st.markdown("Estimate shipping time")
    drive_miles = st.slider("Driving miles", min_value=0.0, max_value=3000.0, value=100.0, step=10.0)
    straight_miles = drive_miles

    if monday_coeffs is not None:
        monday_pred = float(
            monday_coeffs[0]
            + monday_coeffs[1] * drive_miles
            + monday_coeffs[2] * straight_miles
            + monday_coeffs[3] * (drive_miles ** 2)
            + monday_coeffs[4] * (drive_miles ** 3)
        )
        friday_pred = float(
            friday_coeffs[0]
            + friday_coeffs[1] * drive_miles
            + friday_coeffs[2] * straight_miles
            + friday_coeffs[3] * (drive_miles ** 2)
            + friday_coeffs[4] * (drive_miles ** 3)
        ) if friday_coeffs is not None else float("nan")
        monday_bin = nearest_bin(monday_pred, monday_time_bins)
        friday_bin = nearest_bin(friday_pred, friday_time_bins)
        pred_cols = st.columns(2)
        pred_cols[0].metric("Predicted shipping days (Monday)", f"{monday_pred:.2f}")
        pred_cols[1].metric("Predicted shipping days (Friday)", f"{friday_pred:.2f}")
        bin_cols = st.columns(2)
        bin_cols[0].metric("Bin shipping days (Monday)", f"{monday_bin:.2f}")
        bin_cols[1].metric("Bin shipping days (Friday)", f"{friday_bin:.2f}")
    else:
        st.info("Not enough data to fit the model.")

    st.markdown("---")
    st.subheader("Shipping cost model")
    st.caption("Linear regression using driving and straight-line miles to predict cost.")

    monday_cost_coeffs, monday_cost_r2 = fit_linear_model(monday_df, "Cost")
    friday_cost_coeffs, friday_cost_r2 = fit_linear_model(friday_df, "Cost")
    monday_cost_bins = np.sort(monday_df["Cost"].astype(float).unique())
    friday_cost_bins = np.sort(friday_df["Cost"].astype(float).unique())

    cost_cols = st.columns(4)
    cost_cols[0].metric("Monday R2", f"{monday_cost_r2:.3f}" if np.isfinite(monday_cost_r2) else "n/a")
    cost_cols[1].metric("Friday R2", f"{friday_cost_r2:.3f}" if np.isfinite(friday_cost_r2) else "n/a")
    cost_cols[2].metric("Monday rows", f"{len(monday_df):,}")
    cost_cols[3].metric("Friday rows", f"{len(friday_df):,}")

    st.markdown("Estimate shipping cost")
    if monday_cost_coeffs is not None:
        monday_cost_pred = float(
            monday_cost_coeffs[0]
            + monday_cost_coeffs[1] * drive_miles
            + monday_cost_coeffs[2] * straight_miles
            + monday_cost_coeffs[3] * (drive_miles ** 2)
            + monday_cost_coeffs[4] * (drive_miles ** 3)
        )
        friday_cost_pred = float(
            friday_cost_coeffs[0]
            + friday_cost_coeffs[1] * drive_miles
            + friday_cost_coeffs[2] * straight_miles
            + friday_cost_coeffs[3] * (drive_miles ** 2)
            + friday_cost_coeffs[4] * (drive_miles ** 3)
        ) if friday_cost_coeffs is not None else float("nan")
        monday_cost_bin = nearest_bin(monday_cost_pred, monday_cost_bins)
        friday_cost_bin = nearest_bin(friday_cost_pred, friday_cost_bins)
        cost_pred_cols = st.columns(2)
        cost_pred_cols[0].metric("Predicted cost (Monday)", f"{monday_cost_pred:.2f}")
        cost_pred_cols[1].metric("Predicted cost (Friday)", f"{friday_cost_pred:.2f}")
        cost_bin_cols = st.columns(2)
        cost_bin_cols[0].metric("Bin cost (Monday)", f"{monday_cost_bin:.2f}")
        cost_bin_cols[1].metric("Bin cost (Friday)", f"{friday_cost_bin:.2f}")
    else:
        st.info("Not enough data to fit the cost model.")
