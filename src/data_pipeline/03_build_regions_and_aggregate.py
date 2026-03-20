# 03_build_regions_and_aggregate.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# If you want “HK only bbox tighter”: this does it automatically,
# because it builds polygons from the actual ERA5 lat/lon points you downloaded.

CITIES = ["HK", "BKK", "HCM", "KL"]

# Metrics in daily_points_*.parquet
VALUE_COLS = [
    "t2m_mean", "rh2m", "tp_sum_mm",
    "sp_mean", "u10_mean", "v10_mean", "ssrd_sum"
]

def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    # try fastparquet first (works well if your files were written by fastparquet)
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")
            return None

def infer_grid_step(vals: np.ndarray) -> float:
    """Infer typical spacing from sorted unique coordinate values."""
    u = np.unique(np.round(vals.astype(float), 10))
    if len(u) < 2:
        return np.nan
    diffs = np.diff(np.sort(u))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs))

def make_grid_from_points(city: str, df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Create one polygon cell per ERA5 gridpoint using inferred dlat/dlon.
    region_id = f"{city}_{lat}_{lon}" (stable + human-readable)
    """
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    dlat = infer_grid_step(lats)
    dlon = infer_grid_step(lons)

    # fallback if inference fails
    if not np.isfinite(dlat): dlat = 0.1
    if not np.isfinite(dlon): dlon = 0.1

    half_lat = dlat / 2.0
    half_lon = dlon / 2.0

    pts = df[["lat", "lon"]].drop_duplicates().copy()
    # stable ids
    pts["region_id"] = pts.apply(lambda r: f"{city}_{r['lat']:.5f}_{r['lon']:.5f}", axis=1)

    geoms = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(pts["lat"].to_numpy(), pts["lon"].to_numpy())
    ]

    grid = gpd.GeoDataFrame(
        {
            "region_id": pts["region_id"].values,
            "city": city,
            "lat_center": pts["lat"].values,
            "lon_center": pts["lon"].values,
            "dlat": dlat,
            "dlon": dlon,
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    return grid

def assign_and_aggregate(city: str):
    in_path = OUT_DIR / f"daily_points_{city}.parquet"
    df = safe_read_parquet(in_path)

    if df is None:
        print(f"[WARN] Missing/unreadable {in_path}. Skipping {city}.")
        return
    if df.empty:
        print(f"[WARN] {city}: daily_points is empty. Skipping.")
        return

    # required columns
    need = {"time", "lat", "lon"} | set(VALUE_COLS)
    missing = need - set(df.columns)
    if missing:
        print(f"[WARN] {city}: missing columns {sorted(missing)}. Skipping.")
        return

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Build a grid that exactly matches your downloaded ERA5 points
    grid = make_grid_from_points(city, df)

    # Assign region_id by exact (lat,lon) match (no spatial join needed)
    lookup = df[["lat", "lon"]].drop_duplicates().copy()
    lookup["region_id"] = lookup.apply(lambda r: f"{city}_{r['lat']:.5f}_{r['lon']:.5f}", axis=1)

    df = df.merge(lookup, on=["lat", "lon"], how="left")

    # Aggregate to region_id per day (often identical to point values, but safe)
    reg = (
        df.groupby(["time", "region_id"], as_index=False)
          .agg(
              t2m_mean=("t2m_mean", "mean"),
              rh2m_mean=("rh2m", "mean"),
              tp_sum_mm=("tp_sum_mm", "mean"),
              sp_mean=("sp_mean", "mean"),
              u10_mean=("u10_mean", "mean"),
              v10_mean=("v10_mean", "mean"),
              ssrd_sum=("ssrd_sum", "mean"),
          )
    )
    reg["city"] = city

    out = OUT_DIR / f"regional_daily_{city}.parquet"
    reg.to_parquet(out, index=False, engine="fastparquet")
    print(f"{city}: saved {out} rows={len(reg)} regions={grid.shape[0]}")

    grid_out = OUT_DIR / f"grid_{city}.geojson"
    grid.to_file(grid_out, driver="GeoJSON")
    print(f"{city}: saved {grid_out}")

def main():
    for city in CITIES:
        assign_and_aggregate(city)
    print("\nDone. (Missing cities were skipped.)")

if __name__ == "__main__":
    main()
