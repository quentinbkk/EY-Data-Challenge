"""
extract_soilgrids.py
--------------------
Extract SoilGrids 250m v2.0 soil properties for every unique site in the
training and validation datasets.

Data source: ISRIC SoilGrids 250m, served as Cloud-Optimized GeoTIFFs via
WebDAV.  The VRT index files reference all tiles, so rasterio can read
individual pixels without downloading the full global rasters.

Properties extracted
---------------------
  phh2o  – soil pH (H2O)              raw ×10  → divide by 10
  cec    – cation exchange capacity    raw ×10  → divide by 10 (cmolc/kg)
  clay   – clay content                raw ×10  → divide by 10 (g/kg)
  soc    – soil organic carbon         raw ×10  → divide by 10 (g/kg)

Depths: 0-5cm and 5-15cm (surface layers most relevant for runoff chemistry).

Output
------
  soilgrids_cache.parquet  – one row per unique (Latitude, Longitude) site;
                             columns named  sg_{property}_{depth}
                             e.g. sg_phh2o_0_5cm, sg_cec_5_15cm, ...

Re-running this script is safe: it loads the cache and only fetches sites
that are not yet present.

Usage
-----
    python extract_soilgrids.py                         # use default CSV paths
    python extract_soilgrids.py --cache my_cache.parquet
"""

import os
import argparse
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

# ── GDAL tuning for HTTP COG reads ───────────────────────────────────────────
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL   = "https://files.isric.org/soilgrids/latest/data"
CACHE_FILE = "soilgrids_cache.parquet"

# (property, depth-label, unit-scale)
LAYERS = [
    ("phh2o", "0-5cm",  0.1),
    ("phh2o", "5-15cm", 0.1),
    ("cec",   "0-5cm",  0.1),
    ("cec",   "5-15cm", 0.1),
    ("clay",  "0-5cm",  0.1),
    ("clay",  "5-15cm", 0.1),
    ("soc",   "0-5cm",  0.1),
    ("soc",   "5-15cm", 0.1),
    # New layers
    ("sand",  "0-5cm",  0.1),
    ("sand",  "5-15cm", 0.1),
    ("silt",  "0-5cm",  0.1),
    ("silt",  "5-15cm", 0.1),
    ("bdod",  "0-5cm",  0.01),   # bulk density: raw ×100 → cg/cm³, scale to g/cm³
    ("bdod",  "5-15cm", 0.01),
    ("nitrogen", "0-5cm",  0.01),  # raw ×100 → g/kg
    ("nitrogen", "5-15cm", 0.01),
]

TRAIN_WQ_CSV  = "../Data/water_quality_training_dataset.csv"
SUBMISSION_CSV = "../Data/submission_template.csv"


def col_name(prop: str, depth: str) -> str:
    return f"sg_{prop}_{depth.replace('-', '_')}"


def sample_layer(vrt_url: str, xs, ys, scale: float, nodata) -> list:
    """Sample one COG layer at all (x, y) Homolosine coordinates."""
    vals = []
    with rasterio.open(vrt_url) as src:
        nodata = nodata if nodata is not None else src.nodata
        for x, y in zip(xs, ys):
            try:
                row_i, col_i = src.index(x, y)
                raw = src.read(1, window=rasterio.windows.Window(col_i, row_i, 1, 1))[0, 0]
                vals.append(float("nan") if (nodata is not None and raw == nodata) else raw * scale)
            except Exception:
                vals.append(float("nan"))
    return vals


def extract(train_csv: str = TRAIN_WQ_CSV,
            submission_csv: str = SUBMISSION_CSV,
            cache_file: str = CACHE_FILE) -> pd.DataFrame:
    """
    Main extraction routine.  Returns a DataFrame with one row per unique site
    and one column per SoilGrids layer.  Writes/updates cache_file.
    """
    water_quality   = pd.read_csv(train_csv)
    submission_tmpl = pd.read_csv(submission_csv)

    all_coords = pd.concat([
        water_quality[["Latitude", "Longitude"]],
        submission_tmpl[["Latitude", "Longitude"]],
    ]).drop_duplicates().reset_index(drop=True)

    print(f"Total unique sites: {len(all_coords)}")

    # Load cache
    if os.path.exists(cache_file):
        cached      = pd.read_parquet(cache_file)
        cached_keys = set(zip(cached["Latitude"].round(6), cached["Longitude"].round(6)))
        print(f"  Cache loaded: {len(cached)} entries.")
    else:
        cached      = None
        cached_keys = set()

    needed_keys = set(zip(all_coords["Latitude"].round(6), all_coords["Longitude"].round(6)))
    to_fetch    = all_coords[
        all_coords.apply(
            lambda r: (round(r["Latitude"], 6), round(r["Longitude"], 6)) not in cached_keys,
            axis=1,
        )
    ].reset_index(drop=True)

    if to_fetch.empty:
        print("  All sites already cached — nothing to fetch.")
        return cached

    print(f"  Fetching {len(to_fetch)} new sites from SoilGrids COG files...")

    # Project to SoilGrids Homolosine
    transformer = Transformer.from_crs("EPSG:4326", "ESRI:54052", always_xy=True)
    xs, ys = transformer.transform(to_fetch["Longitude"].values, to_fetch["Latitude"].values)

    new_rows = to_fetch.copy()
    for prop, depth, scale in LAYERS:
        key     = col_name(prop, depth)
        vrt_url = f"/vsicurl/{BASE_URL}/{prop}/{prop}_{depth}_mean.vrt"
        print(f"  {key} ...", end=" ", flush=True)
        try:
            vals = sample_layer(vrt_url, xs, ys, scale, nodata=None)
        except Exception as e:
            print(f"\n    ERROR opening {vrt_url}: {e}")
            vals = [float("nan")] * len(to_fetch)
        new_rows[key] = vals
        n_nan = sum(np.isnan(v) for v in vals)
        print(f"mean={np.nanmean(vals):.2f}  NaN={n_nan}")

    # Merge with existing cache and save
    if cached is not None and not cached.empty:
        soil_df = pd.concat([cached, new_rows], ignore_index=True)
    else:
        soil_df = new_rows

    soil_df.to_parquet(cache_file, index=False)
    print(f"\nCache saved: {len(soil_df)} total sites → {cache_file}")
    return soil_df


def main():
    parser = argparse.ArgumentParser(description="Extract SoilGrids features for all water quality sites.")
    parser.add_argument("--train",      default=TRAIN_WQ_CSV,   help="Training water quality CSV")
    parser.add_argument("--submission", default=SUBMISSION_CSV, help="Submission template CSV")
    parser.add_argument("--cache",      default=CACHE_FILE,     help="Output parquet cache path")
    args = parser.parse_args()

    soil_df = extract(args.train, args.submission, args.cache)

    sg_cols = [c for c in soil_df.columns if c.startswith("sg_")]
    print(f"\nExtracted {len(sg_cols)} SoilGrids columns:")
    print(soil_df[sg_cols].describe().round(2))

    # Merge into training_merged.csv and validation_merged.csv
    merge_cols = ["Latitude", "Longitude"] + sg_cols
    for csv_path in ["../Data/training_merged.csv", "../Data/validation_merged.csv"]:
        df = pd.read_csv(csv_path)
        # Drop old sg_ cols to avoid duplicates on re-run
        df = df.drop(columns=[c for c in df.columns if c.startswith("sg_")], errors="ignore")
        df = df.merge(soil_df[merge_cols], on=["Latitude", "Longitude"], how="left")
        df.to_csv(csv_path, index=False)
        nan_count = df[sg_cols].isna().sum().sum()
        print(f"  Merged into {csv_path}  (total NaN across sg cols: {nan_count})")


if __name__ == "__main__":
    main()
