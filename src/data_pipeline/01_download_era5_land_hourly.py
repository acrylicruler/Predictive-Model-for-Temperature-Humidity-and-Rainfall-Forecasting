import cdsapi
from pathlib import Path
import calendar

# -----------------------
# CONFIG
# -----------------------
OUT_DIR = Path("data/era5_land_hourly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# bbox = (min_lon, min_lat, max_lon, max_lat)
CITIES = {
    "HK":  (113.80, 22.15, 114.45, 22.60),
    "BKK": (100.30, 13.45, 100.95, 14.10),
    "HCM": (106.30, 10.55, 106.95, 11.20),
    "KL":  (101.45,  2.90, 101.95,  3.35),
}

VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]

YEARS = list(range(2000, 2025))  # 2000-2024
MONTHS = [f"{m:02d}" for m in range(1, 13)]
HOURS = [f"{h:02d}:00" for h in range(24)]

def days_in_month(year, month_str):
    n = calendar.monthrange(year, int(month_str))[1]
    return [f"{d:02d}" for d in range(1, n + 1)]

def cds_area(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    # CDS format: [N, W, S, E]
    return [max_lat, min_lon, min_lat, max_lon]

def main():
    c = cdsapi.Client()

    for city, bbox in CITIES.items():
        area = cds_area(bbox)

        for y in YEARS:
            for m in MONTHS:

                out = OUT_DIR / f"{city}_{y}_{m}.zip"

                out_nc = OUT_DIR / f"{city}_{y}_{m}.nc"

                if out.exists() or out_nc.exists():
                    print("Skip:", out if out.exists() else out_nc)
                    continue

                print("Downloading:", city, y, m)
                c.retrieve(
                    "reanalysis-era5-land",
                    {
                        "variable": VARS,
                        "year": str(y),
                        "month": m,
                        "day": days_in_month(y, m),
                        "time": HOURS,
                        "area": area,
                        "format": "netcdf",
                    },
                    str(out),
                )
                print("Saved:", out)

if __name__ == "__main__":
    main()
