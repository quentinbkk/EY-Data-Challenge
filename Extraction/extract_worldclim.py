#!/usr/bin/env python3
"""
extract_worldclim.py
====================
Downloads WorldClim 2.1 BIO variables (30-year climatological normals,
1970–2000) at 2.5 arc-minute resolution and extracts values for every
unique site, appending them directly to Data/training_merged.csv and
Data/validation_merged.csv.

Features added
--------------
  wc_bio1   – Mean Annual Temperature (°C × 10 → converted to °C)
  wc_bio4   – Temperature Seasonality (std of monthly T × 100)
  wc_bio12  – Annual Precipitation (mm/year)
  wc_bio15  – Precipitation Seasonality (CV of monthly precip)

Why useful
----------
These are 30-year (1970–2000) climatological normals — far more stable
than the 5-year monthly TerraClimate values currently used.

  wc_bio12  captures the LONG-RUN dilution effect: wetter sites have
            lower long-run TA/EC regardless of month-to-month variation.
  wc_bio15  captures rainfall concentration: highly seasonal sites (dry
            season + wet season) concentrate ions differently than
            aseasonal sites.
  wc_bio1   long-run temperature → weathering kinetics at the site.
  wc_bio4   temperature swings → freeze/thaw cycles → mineral breakdown.

Data source
-----------
WorldClim 2.1 (Fick & Hijmans 2017, doi:10.1002/joc.5086)
GeoTIFF rasters downloaded from:
  https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio_{i}.tif

Usage (from project root)
--------------------------
  source venv/bin/activate
  python Extraction/extract_worldclim.py          # download + extract + update CSVs
  python Extraction/extract_worldclim.py --force  # re-download even if cached
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rasterio

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
ROOT        = SCRIPT_DIR.parent
DATA_DIR    = ROOT / "Data"
RASTER_DIR  = SCRIPT_DIR / "worldclim_rasters"
TRAIN_CSV   = DATA_DIR / "training_merged.csv"
VAL_CSV     = DATA_DIR / "validation_merged.csv"

# BIO variables to download: {index: column_name}
BIO_VARS = {
    1:  "wc_bio1",   # Mean Annual Temperature
    4:  "wc_bio4",   # Temperature Seasonality
    12: "wc_bio12",  # Annual Precipitation
    15: "wc_bio15",  # Precipitation Seasonality
}

WC_ZIP_URL  = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip"

# ── Download ──────────────────────────────────────────────────────────────────

def download_rasters(force: bool = False) -> dict[int, Path]:
    """
    Download wc2.1_2.5m_bio.zip (all 19 bio variables) if not already present,
    extract the TIF files we need, and return {bio_idx: local_path}.
    """
    import zipfile

    RASTER_DIR.mkdir(exist_ok=True)
    zip_path = RASTER_DIR / "wc2.1_2.5m_bio.zip"

    # Check if all required TIFs are already extracted
    needed = {i: RASTER_DIR / f"wc2.1_2.5m_bio_{i}.tif" for i in BIO_VARS}
    if not force and all(p.exists() for p in needed.values()):
        print("  All BIO rasters already extracted.")
        return needed

    # Download zip if missing or forced
    if force or not zip_path.exists():
        print(f"  Downloading {WC_ZIP_URL} ...", end="", flush=True)
        r = requests.get(WC_ZIP_URL, stream=True, timeout=300)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        size_mb = zip_path.stat().st_size / 1e6
        print(f" {size_mb:.0f} MB")
    else:
        print(f"  Zip already present ({zip_path.name}).")

    # Extract only the TIFs we need
    print("  Extracting required BIO TIFs ...")
    with zipfile.ZipFile(zip_path) as zf:
        for bio_idx, dest in needed.items():
            if dest.exists() and not force:
                print(f"    bio{bio_idx}: already extracted")
                continue
            member = f"wc2.1_2.5m_bio_{bio_idx}.tif"
            zf.extract(member, RASTER_DIR)
            size_mb = dest.stat().st_size / 1e6
            print(f"    bio{bio_idx}: extracted ({size_mb:.1f} MB)")

    return needed


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_values(raster_path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Extract raster values at (lat, lon) coordinates using nearest-neighbour
    sampling (appropriate for 2.5 arc-minute resolution ≈ 4.6 km).
    Returns array of float values, NaN where outside raster extent.
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        # rasterio index() maps geographic coords → pixel row/col
        rows, cols = rasterio.transform.rowcol(
            src.transform, lons, lats
        )
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        height, width = src.height, src.width

        # Clamp to valid pixel range
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        values = np.full(len(lats), np.nan)

        if valid.any():
            data = src.read(1)
            raw = data[rows[valid], cols[valid]].astype(float)
            if nodata is not None:
                raw[raw == nodata] = np.nan
            values[valid] = raw

    return values


def build_worldclim_features(
    df: pd.DataFrame,
    raster_paths: dict[int, Path],
) -> pd.DataFrame:
    """
    For each BIO variable, extract values at df's (Latitude, Longitude) coords.
    Returns a DataFrame with one column per BIO variable.
    """
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    result = {}

    for bio_idx, col_name in BIO_VARS.items():
        vals = extract_values(raster_paths[bio_idx], lats, lons)
        # WorldClim v2.1 stores all bio variables as floats in their natural units
        # (bio1 = °C directly — no scaling needed, unlike v1.4 which used °C × 10)
        result[col_name] = vals
        n_valid = np.sum(~np.isnan(vals))
        print(f"  {col_name}: min={np.nanmin(vals):.1f}  "
              f"mean={np.nanmean(vals):.1f}  "
              f"max={np.nanmax(vals):.1f}  "
              f"({n_valid}/{len(vals)} valid)")

    return pd.DataFrame(result, index=df.index)


def update_csv(csv_path: Path, features_df: pd.DataFrame) -> None:
    df = pd.read_csv(csv_path)

    # Drop any existing WorldClim columns
    existing = [c for c in df.columns if c.startswith("wc_")]
    if existing:
        df.drop(columns=existing, inplace=True)
        print(f"  Dropped {len(existing)} existing wc_ columns.")

    for col in features_df.columns:
        df[col] = features_df[col].values

    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.name} ({len(df):,} rows, +{len(features_df.columns)} wc_ cols)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    print("Downloading WorldClim 2.1 BIO rasters (2.5 arc-min, ~4.6 km) ...")
    raster_paths = download_rasters(force=force)

    print("\nLoading site coordinates ...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    print("\nExtracting WorldClim values — TRAINING:")
    train_wc = build_worldclim_features(train_df, raster_paths)

    print("\nExtracting WorldClim values — VALIDATION:")
    val_wc = build_worldclim_features(val_df, raster_paths)

    print("\nUpdating CSVs ...")
    print("── TRAINING ──")
    update_csv(TRAIN_CSV, train_wc)
    print("── VALIDATION ──")
    update_csv(VAL_CSV, val_wc)

    print("\nWorldClim extraction complete.")

    # Summary comparison
    print("\nKey feature comparison — train vs val:")
    for col in [f"wc_{c}" for c in ["bio1", "bio12", "bio15", "bio4"]]:
        tm = train_wc[col].mean()
        vm = val_wc[col].mean()
        flag = " ◄" if abs(vm - tm) / (abs(tm) + 1e-9) > 0.2 else ""
        print(f"  {col:12s}: train={tm:.1f}  val={vm:.1f}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract WorldClim 2.1 BIO variables per site."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download rasters even if already present locally."
    )
    args = parser.parse_args()
    main(force=args.force)
