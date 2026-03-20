# 02_build_daily_from_nc.py
import zipfile
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("data/era5_land_hourly")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = DATA_DIR / "_cache_extracted"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["HK", "BKK", "HCM", "KL"]

# expected variables inside ERA5-Land output (your inner data_0.nc)
NEEDED_VARS = ["t2m", "d2m", "tp", "sp", "u10", "v10", "ssrd"]


# -----------------------
# Helpers
# -----------------------
def is_zip_file(path: Path) -> bool:
    """Detect zip by header, not by extension."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


def extract_inner_nc(path: Path) -> Path | None:
    """
    If path is a zip (even if named .nc), extract the first .nc inside
    (usually data_0.nc) into CACHE_DIR and return extracted file path.
    """
    if not is_zip_file(path):
        return path  # already a real netcdf

    out_nc = CACHE_DIR / f"{path.stem}__inner.nc"
    if out_nc.exists() and out_nc.stat().st_size > 0:
        return out_nc

    try:
        with zipfile.ZipFile(path, "r") as z:
            members = [m for m in z.namelist() if m.lower().endswith(".nc")]
            if not members:
                print(f"[WARN] {path.name} is zip but contains no .nc. Skipping.")
                return None

            # prefer data_0.nc if present
            target = "data_0.nc" if "data_0.nc" in members else sorted(members)[0]

            tmp = CACHE_DIR / f"{path.stem}__tmp.nc"
            with z.open(target) as src, open(tmp, "wb") as dst:
                dst.write(src.read())

            if tmp.stat().st_size == 0:
                print(f"[WARN] Extracted empty inner nc from {path.name}. Skipping.")
                tmp.unlink(missing_ok=True)
                return None

            tmp.replace(out_nc)
            return out_nc

    except zipfile.BadZipFile:
        print(f"[WARN] Bad zip file: {path}. Skipping.")
        return None
    except Exception as e:
        print(f"[WARN] Failed extracting {path.name}: {e}. Skipping.")
        return None


def list_city_month_files(city: str) -> list[Path]:
    """
    Find all files for a city, regardless of being .zip / .nc / fake .nc zip.
    """
    # match HK_YYYY_MM.*
    files = sorted(DATA_DIR.glob(f"{city}_*.nc")) + sorted(DATA_DIR.glob(f"{city}_*.zip"))
    return sorted(set(files))


def open_one_dataset(path: Path) -> xr.Dataset:
    """
    Open one ERA5-Land monthly file robustly.
    - Extract inner nc if needed
    - Use h5netcdf (works well for NetCDF4/HDF5)
    - Rename valid_time -> time
    """
    inner = extract_inner_nc(path)
    if inner is None:
        raise OSError(f"Could not extract inner NetCDF from {path.name}")

    ds = xr.open_dataset(inner, engine="h5netcdf")

    # handle valid_time
    if "time" not in ds.coords and "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    elif "time" not in ds.coords and "time" in ds.variables:
        # sometimes time might exist as variable, rare
        ds = ds.set_coords("time")

    return ds


def to_daily_df_from_month(ds: xr.Dataset) -> pd.DataFrame:
    """
    Convert one month hourly ds -> daily df for all grid points.
    """
    # confirm variables exist
    missing = [v for v in NEEDED_VARS if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Missing vars {missing}. Found: {list(ds.data_vars)}")

    # Convert units first (still hourly)
    ds = ds.assign(
        t2m_c=ds["t2m"] - 273.15,
        d2m_c=ds["d2m"] - 273.15,
        tp_mm=ds["tp"] * 1000.0,  # m -> mm
    )

    # daily aggregation (xarray way)
    daily = xr.Dataset(
        data_vars=dict(
            t2m_mean=ds["t2m_c"].resample(time="1D").mean(),
            d2m_mean=ds["d2m_c"].resample(time="1D").mean(),
            tp_sum_mm=ds["tp_mm"].resample(time="1D").sum(),
            sp_mean=ds["sp"].resample(time="1D").mean(),
            u10_mean=ds["u10"].resample(time="1D").mean(),
            v10_mean=ds["v10"].resample(time="1D").mean(),
            ssrd_sum=ds["ssrd"].resample(time="1D").sum(),
        )
    )

    df = daily.to_dataframe().reset_index()

    # rename coords to match your pipeline
    if "latitude" in df.columns:
        df.rename(columns={"latitude": "lat"}, inplace=True)
    if "longitude" in df.columns:
        df.rename(columns={"longitude": "lon"}, inplace=True)

    # RH from T and Td (Magnus)
    a, b = 17.625, 243.04
    es = np.exp((a * df["t2m_mean"]) / (b + df["t2m_mean"]))
    e = np.exp((a * df["d2m_mean"]) / (b + df["d2m_mean"]))
    df["rh2m"] = (100.0 * (e / es)).clip(0, 100)

    df.drop(columns=["d2m_mean"], inplace=True)

    # keep clean columns
    keep = ["time", "lat", "lon", "t2m_mean", "rh2m", "tp_sum_mm", "sp_mean", "u10_mean", "v10_mean", "ssrd_sum"]
    df = df[keep].dropna()

    return df


# -----------------------
# MAIN
# -----------------------
def main():
    warnings.filterwarnings("ignore")

    for city in CITIES:
        files = list_city_month_files(city)
        if not files:
            print(f"\nLoading: {city}")
            print(f"[WARN] No files found for {city} in {DATA_DIR}. Skipping city.")
            continue

        print(f"\nLoading: {city}")
        all_daily = []
        tmin, tmax = None, None
        bad = 0

        for f in files:
            try:
                ds = open_one_dataset(f)

                # track span
                _min = pd.to_datetime(ds["time"].min().values)
                _max = pd.to_datetime(ds["time"].max().values)
                tmin = _min if (tmin is None or _min < tmin) else tmin
                tmax = _max if (tmax is None or _max > tmax) else tmax

                df = to_daily_df_from_month(ds)
                all_daily.append(df)

                ds.close()
            except Exception as e:
                bad += 1
                print(f"[WARN] {city}: skip file {f.name} ({e})")

        if not all_daily:
            print(f"[WARN] {city}: no usable files after filtering. Skipping.")
            continue

        out_df = pd.concat(all_daily, ignore_index=True)

        # de-dupe in case overlap
        out_df = out_df.drop_duplicates(subset=["time", "lat", "lon"])

        out_df["city"] = city

        if tmin is not None and tmax is not None:
            print(f"{city} time span: {tmin.date()} -> {tmax.date()}")

        if bad:
            print(f"[WARN] {city}: skipped {bad} bad/unreadable file(s).")

        out = OUT_DIR / f"daily_points_{city}.parquet"
        out_df.to_parquet(out, index=False)
        print("Saved:", out, "rows:", len(out_df), "cols:", out_df.shape[1])

    print("\nDone. (Cities without data were skipped.)")


if __name__ == "__main__":
    main()
