"""
extract_hydrosheds.py — Extract HydroSHEDS upstream catchment area and stream order
for each unique site in training_merged.csv and validation_merged.csv.

Source: HydroRIVERS v1.0 Africa (hydrosheds.org)
  - UP_AREA  : upstream catchment area (km²)
  - ORD_STRA : Strahler stream order

Outputs:
  - Extraction/hydrosheds_cache.parquet  (per-site cache)
  - Data/training_merged.csv             (with upland_skm, strahler_order columns)
  - Data/validation_merged.csv           (same)
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "Data"
EXTRACT_DIR = ROOT / "Extraction"
CACHE_FILE  = EXTRACT_DIR / "hydrosheds_cache.parquet"
SHP_CACHE   = EXTRACT_DIR / "hydrorivers_af"        # extracted shapefile folder

HYDRORIVERS_URL = (
    "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_af_shp.zip"
)

SEARCH_RADIUS_DEG = 0.1   # ~11 km — wide enough to always find a reach
FALLBACK_RADIUS_DEG = 0.5 # wider fallback for remote sites


# ── Step 1: Download & cache HydroRIVERS Africa shapefile ────────────────────

def download_hydrorivers():
    """Download and extract HydroRIVERS Africa shapefile if not already cached."""
    shp_files = list(SHP_CACHE.rglob("*.shp")) if SHP_CACHE.exists() else []
    if shp_files:
        print(f"HydroRIVERS shapefile already cached at {shp_files[0]}")
        return shp_files[0]

    print(f"Downloading HydroRIVERS Africa from hydrosheds.org (~50 MB)...")
    r = requests.get(HYDRORIVERS_URL, stream=True, timeout=120)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    data = bytearray()
    downloaded = 0
    for chunk in r.iter_content(chunk_size=1024 * 256):
        data.extend(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r  {pct:.0f}%  ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)", end="", flush=True)
    print()

    SHP_CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(SHP_CACHE)

    shp_files = list(SHP_CACHE.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp file found after extracting zip.")
    print(f"Extracted to {SHP_CACHE}")
    return shp_files[0]


# ── Step 2: Load shapefile and build spatial index ────────────────────────────

def load_rivers(shp_path):
    print(f"Loading HydroRIVERS shapefile: {shp_path}")
    rivers = gpd.read_file(shp_path, columns=["UPLAND_SKM", "ORD_STRA", "geometry"])
    # Ensure WGS84
    if rivers.crs is None or rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs(epsg=4326)
    print(f"  Loaded {len(rivers):,} river reaches")
    return rivers


# ── Step 3: Nearest-reach lookup per site ────────────────────────────────────

def query_site(lat, lon, rivers_sindex, rivers, radius=SEARCH_RADIUS_DEG):
    """Return (upland_skm, strahler_order) for the nearest river reach."""
    pt = Point(lon, lat)
    bbox = (lon - radius, lat - radius, lon + radius, lat + radius)
    candidates_idx = list(rivers_sindex.intersection(bbox))

    if not candidates_idx:
        if radius < FALLBACK_RADIUS_DEG:
            return query_site(lat, lon, rivers_sindex, rivers, FALLBACK_RADIUS_DEG)
        return np.nan, np.nan

    candidates = rivers.iloc[candidates_idx].to_crs(epsg=32735)  # UTM zone 35S (South Africa)
    pt_proj = gpd.GeoSeries([pt], crs=4326).to_crs(epsg=32735).iloc[0]
    dists = candidates.geometry.distance(pt_proj)
    nearest = rivers.iloc[candidates_idx[dists.values.argmin()]]
    return float(nearest["UPLAND_SKM"]), int(nearest["ORD_STRA"])


def extract_all_sites(sites, rivers):
    print("Building spatial index...")
    sindex = rivers.sindex

    results = []
    n = len(sites)
    for i, (_, row) in enumerate(sites.iterrows()):
        up_area, stra_ord = query_site(row.Latitude, row.Longitude, sindex, rivers)
        results.append({
            "Latitude":       row.Latitude,
            "Longitude":      row.Longitude,
            "upland_skm":     up_area,
            "strahler_order": stra_ord,
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} sites done")

    return pd.DataFrame(results)


# ── Step 4: Merge into training/validation CSVs ───────────────────────────────

def merge_into_csv(csv_path, hydro_df):
    df = pd.read_csv(csv_path)

    # Drop old columns if re-running
    for col in ["upland_skm", "strahler_order"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(
        hydro_df[["Latitude", "Longitude", "upland_skm", "strahler_order"]],
        on=["Latitude", "Longitude"],
        how="left",
    )
    df.to_csv(csv_path, index=False)
    print(f"  Merged into {csv_path.name}  "
          f"(upland_skm NaN: {df.upland_skm.isna().sum()}, "
          f"strahler NaN: {df.strahler_order.isna().sum()})")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Collect all unique sites across train + val
    train = pd.read_csv(DATA_DIR / "training_merged.csv")
    val   = pd.read_csv(DATA_DIR / "validation_merged.csv")
    sites = (
        pd.concat([train[["Latitude", "Longitude"]], val[["Latitude", "Longitude"]]])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"Total unique sites: {len(sites)}")

    # Load from cache if available
    if CACHE_FILE.exists():
        print(f"Loading cached HydroSHEDS attributes from {CACHE_FILE}")
        hydro_df = pd.read_parquet(CACHE_FILE)
        # Check if all current sites are covered
        cached_keys = set(zip(hydro_df.Latitude, hydro_df.Longitude))
        missing = sites[
            ~sites.apply(lambda r: (r.Latitude, r.Longitude) in cached_keys, axis=1)
        ]
        if len(missing) == 0:
            print("  All sites already cached.")
        else:
            print(f"  {len(missing)} new sites — querying...")
            shp_path = download_hydrorivers()
            rivers   = load_rivers(shp_path)
            new_rows = extract_all_sites(missing, rivers)
            hydro_df = pd.concat([hydro_df, new_rows], ignore_index=True)
            hydro_df.to_parquet(CACHE_FILE, index=False)
    else:
        shp_path = download_hydrorivers()
        rivers   = load_rivers(shp_path)
        hydro_df = extract_all_sites(sites, rivers)
        hydro_df.to_parquet(CACHE_FILE, index=False)
        print(f"Saved cache → {CACHE_FILE}")

    print("\nHydroSHEDS stats:")
    print(hydro_df[["upland_skm", "strahler_order"]].describe().round(1))

    # Merge into CSVs
    print("\nMerging into training_merged.csv...")
    merge_into_csv(DATA_DIR / "training_merged.csv", hydro_df)
    print("Merging into validation_merged.csv...")
    merge_into_csv(DATA_DIR / "validation_merged.csv", hydro_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
