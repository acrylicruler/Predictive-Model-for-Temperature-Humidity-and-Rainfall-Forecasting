from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

LAND_CANDIDATES = [
    INTERIM_DIR / "hk_raw_ib1000.geojson",
    INTERIM_DIR / "hk_land_boundary.geojson",
]

GRID_CANDIDATES = [
    INTERIM_DIR / "grid_HK.geojson",
    OUT_DIR / "grid_HK.geojson",
]

OUT_BOUNDARY = INTERIM_DIR / "hk_land_boundary.geojson"
OUT_GRID = INTERIM_DIR / "grid_HK_land_clipped.geojson"
OUT_SUMMARY = INTERIM_DIR / "grid_HK_land_clipped_summary.csv"

# Tweakable thresholds
MIN_LAND_FRACTION = 0.05      # drop cells that are overwhelmingly sea
MIN_PIECE_AREA_M2 = 2_500     # remove tiny slivers after clipping
KEEP_IF_CENTER_ON_LAND = True # preserve cells whose original center falls on land


def read_first(paths: list[Path]) -> tuple[Path, gpd.GeoDataFrame]:
    for p in paths:
        if p.exists():
            gdf = gpd.read_file(p).to_crs("EPSG:4326")
            if not gdf.empty:
                return p, gdf
    raise FileNotFoundError(f"None of these files exist or are readable: {paths}")


def clean_polygon_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out = out[out.geometry.notna()].copy()
    out = out[~out.geometry.is_empty].copy()
    out = out[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if out.empty:
        raise ValueError("No polygon geometries found after cleaning.")
    return out


def build_land_boundary(raw_land: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    land = clean_polygon_gdf(raw_land)
    metric = land.to_crs(3857)
    keep = metric.geometry.area > 100.0
    land = land.loc[keep.values].copy().to_crs("EPSG:4326")

    try:
        merged = land.union_all()
    except AttributeError:
        merged = land.unary_union

    try:
        merged = merged.buffer(0)
    except Exception:
        pass

    boundary = gpd.GeoDataFrame({"name": ["Hong Kong land"]}, geometry=[merged], crs="EPSG:4326")
    return boundary


def representative_points_on_land(grid: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> pd.Series:
    reps = gpd.GeoSeries(grid.geometry.representative_point(), crs=grid.crs)
    land_geom = boundary.geometry.iloc[0]
    return reps.apply(lambda p: bool(land_geom.covers(p)))


def make_clipped_grid(grid: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    g = clean_polygon_gdf(grid)
    if "region_id" not in g.columns:
        raise ValueError("Grid file must contain region_id")
    g["region_id"] = g["region_id"].astype(str)

    original = g[[c for c in g.columns if c != "geometry"] + ["geometry"]].copy()

    original_metric = original.to_crs(3857)
    original_area = pd.DataFrame({
        "region_id": original["region_id"].values,
        "cell_area_m2": original_metric.geometry.area.values,
    })

    center_on_land = representative_points_on_land(original, boundary)
    center_df = pd.DataFrame({
        "region_id": original["region_id"].values,
        "center_on_land": center_on_land.values,
    })

    try:
        clipped = gpd.overlay(
            original[["region_id", "geometry"]].copy(),
            boundary[["geometry"]].copy(),
            how="intersection",
        )
    except Exception:
        clipped = gpd.clip(original[["region_id", "geometry"]].copy(), boundary[["geometry"]].copy())

    try:
        clipped = clipped.explode(index_parts=False).reset_index(drop=True)
    except TypeError:
        clipped = clipped.explode().reset_index(drop=True)

    clipped = clean_polygon_gdf(clipped)

    clipped_metric = clipped.to_crs(3857)
    clipped["piece_area_m2"] = clipped_metric.geometry.area.values
    clipped = clipped[clipped["piece_area_m2"] >= MIN_PIECE_AREA_M2].copy()

    if clipped.empty:
        raise ValueError("All clipped pieces were removed. Lower MIN_PIECE_AREA_M2.")

    land_area = (
        clipped.groupby("region_id", as_index=False)["piece_area_m2"]
        .sum()
        .rename(columns={"piece_area_m2": "land_area_m2"})
    )

    summary = original.drop(columns="geometry").merge(original_area, on="region_id", how="left")
    summary = summary.merge(center_df, on="region_id", how="left")
    summary = summary.merge(land_area, on="region_id", how="left")
    summary["land_area_m2"] = summary["land_area_m2"].fillna(0.0)
    summary["land_fraction"] = summary["land_area_m2"] / summary["cell_area_m2"]

    keep_mask = summary["land_fraction"] >= MIN_LAND_FRACTION
    if KEEP_IF_CENTER_ON_LAND:
        keep_mask = keep_mask | summary["center_on_land"].fillna(False)

    kept_regions = set(summary.loc[keep_mask, "region_id"])
    clipped = clipped[clipped["region_id"].isin(kept_regions)].copy()
    summary = summary[summary["region_id"].isin(kept_regions)].copy()

    if clipped.empty:
        raise ValueError("All regions were filtered out. Lower MIN_LAND_FRACTION.")

    dissolved = clipped.dissolve(by="region_id", aggfunc="sum").reset_index()
    keep_cols = [c for c in original.columns if c != "geometry"]
    dissolved = dissolved.merge(summary[keep_cols + ["cell_area_m2", "land_area_m2", "land_fraction"]], on="region_id", how="left")
    dissolved = dissolved.to_crs("EPSG:4326")

    return dissolved, summary.sort_values(["land_fraction", "region_id"], ascending=[False, True]).reset_index(drop=True)


def main():
    land_path, raw_land = read_first(LAND_CANDIDATES)
    grid_path, raw_grid = read_first(GRID_CANDIDATES)

    print(f"Using land source: {land_path}")
    print(f"Using grid source: {grid_path}")

    boundary = build_land_boundary(raw_land)
    clipped_grid, summary = make_clipped_grid(raw_grid, boundary)

    boundary.to_file(OUT_BOUNDARY, driver="GeoJSON")
    clipped_grid.to_file(OUT_GRID, driver="GeoJSON")
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved land boundary -> {OUT_BOUNDARY}")
    print(f"Saved clipped grid  -> {OUT_GRID}")
    print(f"Saved summary       -> {OUT_SUMMARY}")
    print(f"Original grid cells: {len(raw_grid)}")
    print(f"Kept clipped cells : {len(clipped_grid)}")
    print(summary[["region_id", "land_fraction", "center_on_land"]])


if __name__ == "__main__":
    main()
