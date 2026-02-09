import pandas as pd


def compute_best_origin_map(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    required_cols = {"Destination", "FromAddress", "ShippingTimeDays"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    min_times = (
        df.groupby("Destination", as_index=False)["ShippingTimeDays"]
        .min()
        .rename(columns={"ShippingTimeDays": "BestTime"})
    )
    best_rows = df.merge(min_times, on="Destination")
    best_rows = best_rows[best_rows["ShippingTimeDays"] == best_rows["BestTime"]]
    return (
        best_rows.groupby("Destination")["FromAddress"]
        .apply(lambda values: ", ".join(sorted({str(v).strip() for v in values if str(v).strip()})))
        .to_dict()
    )
