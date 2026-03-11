#!/usr/bin/env python3
"""
extract_elevation.py
====================
Queries the OpenTopoData API (https://www.opentopodata.org) using the
SRTM30m dataset to extract elevation (metres) for every unique site and
appends the feature directly to Data/training_merged.csv and
Data/validation_merged.csv.

Features added
--------------
  elevation_m   – site elevation in metres above sea level (SRTM30m)

Why useful
----------
Elevation is a proxy for:
  - Long-run temperature (lapse rate ≈ 6.5°C / 1000m) → weathering kinetics
  - Catchment relief / slope → faster runoff = more dilution = lower TA/EC
  - Rainfall regime (orographic enhancement at higher elevations)

Eastern Cape validation sites range from near-coastal to inland Karoo
plateau (~1200m) — a large range that drives meaningful TA/EC variation.

Usage (from project root)
--------------------------
  source venv/bin/activate
  python Extraction/extract_elevation.py          # fetch + update CSVs
  python Extraction/extract_elevation.py --force  # re-fetch even if cached
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent
DATA_DIR   = ROOT / "Data"
CACHE_FILE = SCRIPT_DIR / "elevation_cache.json"
TRAIN_CSV  = DATA_DIR / "training_merged.csv"
VAL_CSV    = DATA_DIR / "validation_merged.csv"

OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"
BATCH_SIZE = 100   # API allows up to 100 locations per request
SLEEP_S    = 1.1   # rate limit: 1 req/s for free tier

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 6)},{round(lon, 6)}"


def query_batch(latlons: list[tuple[float, float]]) -> dict[str, float | None]:
    """Query up to BATCH_SIZE (lat, lon) pairs and return {key: elevation}."""
    locations = "|".join(f"{lat},{lon}" for lat, lon in latlons)
    try:
        r = requests.get(
            OPENTOPODATA_URL,
            params={"locations": locations},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        results = {}
        for item in data.get("results", []):
            loc = item["location"]
            key = _cache_key(loc["lat"], loc["lng"])
            results[key] = item.get("elevation")
        return results
    except Exception as e:
        print(f"    WARNING: batch query failed — {e}")
        return {_cache_key(lat, lon): None for lat, lon in latlons}


def fetch_elevations(
    sites_df: pd.DataFrame,
    cache: dict,
    force: bool = False,
) -> dict:
    """
    Fetch elevation for all unique (Latitude, Longitude) in sites_df.
    Results are merged into cache in-place and returned.
    """
    unique_sites = (
        sites_df[["Latitude", "Longitude"]]
        .drop_duplicates()
        .values.tolist()
    )

    to_fetch = [
        (lat, lon)
        for lat, lon in unique_sites
        if (not (pd.isna(lat) or pd.isna(lon)))
        and (force or _cache_key(lat, lon) not in cache)
    ]

    if not to_fetch:
        print(f"  All {len(unique_sites)} sites already cached.")
        return cache

    print(f"  Fetching elevation for {len(to_fetch)} sites "
          f"({len(unique_sites) - len(to_fetch)} cached) ...")

    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i : i + BATCH_SIZE]
        batch_results = query_batch(batch)
        cache.update(batch_results)
        n_done = min(i + BATCH_SIZE, len(to_fetch))
        sample = next(iter(batch_results.values()))
        print(f"    [{n_done:3d}/{len(to_fetch)}] last elevation = {sample}")
        if n_done < len(to_fetch):
            time.sleep(SLEEP_S)

    return cache


def build_elevation_series(df: pd.DataFrame, cache: dict) -> pd.Series:
    """Return a Series of elevation_m aligned to df's index."""
    return df.apply(
        lambda row: cache.get(_cache_key(row["Latitude"], row["Longitude"])),
        axis=1,
    ).astype(float)


def update_csv(csv_path: Path, elevation_series: pd.Series) -> None:
    df = pd.read_csv(csv_path)

    # Drop existing column if present
    if "elevation_m" in df.columns:
        df.drop(columns=["elevation_m"], inplace=True)

    df["elevation_m"] = elevation_series.values
    df.to_csv(csv_path, index=False)
    missing = df["elevation_m"].isna().sum()
    print(f"  Saved {csv_path.name} ({len(df):,} rows) — "
          f"elevation_m: {missing} missing")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    print("Loading site coordinates from merged CSVs ...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    all_sites = pd.concat(
        [train_df[["Latitude", "Longitude"]], val_df[["Latitude", "Longitude"]]],
        ignore_index=True,
    )

    cache = load_cache(CACHE_FILE)
    print(f"Cache: {len(cache)} entries loaded from {CACHE_FILE.name}")

    cache = fetch_elevations(all_sites, cache, force=force)
    save_cache(cache, CACHE_FILE)
    print(f"Cache saved ({len(cache)} entries).")

    print("\nUpdating CSVs ...")
    print("── TRAINING ──")
    update_csv(TRAIN_CSV, build_elevation_series(train_df, cache))
    print("── VALIDATION ──")
    update_csv(VAL_CSV, build_elevation_series(val_df, cache))

    print("\nElevation extraction complete.")

    # Quick sanity check
    val_updated = pd.read_csv(VAL_CSV)
    print(f"\nValidation elevation stats:")
    print(f"  min={val_updated['elevation_m'].min():.0f}m  "
          f"mean={val_updated['elevation_m'].mean():.0f}m  "
          f"max={val_updated['elevation_m'].max():.0f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SRTM30m elevation per site.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if elevation is already cached.")
    args = parser.parse_args()
    main(force=args.force)
