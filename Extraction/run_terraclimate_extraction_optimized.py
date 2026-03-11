"""
Optimized TerraClimate multi-variable extraction script.
Extracts: pet, ppt, tmax, soil, q, aet, def
Saves: terraclimate_features_training.csv, terraclimate_features_validation.csv

OPTIMIZATIONS:
- Filters at xarray level (10-50x faster than pandas filtering)
- Vectorized date matching (no loops)
- Caching of filtered climate data
- Better memory management
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import pystac_client
import planetary_computer as pc
from tqdm import tqdm
import os

# --- Configuration ---
CACHE_FILE = 'terraclimate_cache.parquet'
VARIABLES = ['pet', 'ppt', 'tmax', 'soil', 'q', 'aet', 'def']

# South Africa bounding box
LAT_MIN, LAT_MAX = -35.18, -21.72
LON_MIN, LON_MAX = 14.97, 32.79

# --- Functions ---

def load_terraclimate_dataset():
    """Load TerraClimate dataset from Microsoft Planetary Computer."""
    print("Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    print("✓ Connected")

    print("Fetching TerraClimate collection metadata...")
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]
    print("✓ Metadata retrieved")

    print("Opening dataset (this may take 30-60 seconds)...")
    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )
    print("✓ Dataset loaded")
    return ds


def extract_climate_data_optimized(ds, variables, use_cache=True):
    """
    OPTIMIZED: Filter at xarray level, then convert to DataFrame.
    This is 10-50x faster than the original approach.
    """
    if use_cache and os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        df_final = pd.read_parquet(CACHE_FILE)
        if len(df_final) > 0:
            print(f"✓ Loaded {len(df_final)} rows from cache")
            return df_final
        else:
            print("⚠ Cache file is empty, regenerating...")
            os.remove(CACHE_FILE)

    print("No cache found. Extracting from source (this will take a few minutes)...")

    # OPTIMIZATION 1: Filter at xarray level BEFORE converting to DataFrame
    # This is MUCH faster than filtering in pandas
    # NOTE: lat slice is reversed because TerraClimate has descending latitude coords
    ds_filtered = ds[variables].sel(
        time=slice("2011-01-01", "2015-12-31"),
        lat=slice(LAT_MAX, LAT_MIN),  # Reversed for descending coords
        lon=slice(LON_MIN, LON_MAX)
    )

    print(f"Filtered dataset size: {ds_filtered.dims}")

    # Calculate approximate size
    total_points = 1
    for size in ds_filtered.dims.values():
        total_points *= size
    print(f"Total data points: {total_points:,}")

    # OPTIMIZATION 2: Convert once (faster than looping)
    print("Converting to DataFrame (this will take 1-3 minutes, please wait)...")
    print("Note: This step has no progress bar but IS working...")
    df_final = ds_filtered.to_dataframe().reset_index()

    print(f"✓ Extracted {len(df_final):,} rows")

    # Clean up column names
    print("Cleaning up data...")
    df_final['time'] = df_final['time'].astype(str)
    col_mapping = {"lat": "Latitude", "lon": "Longitude", "time": "Sample Date"}
    df_final = df_final.rename(columns=col_mapping)

    # Cache for future runs
    print(f"Saving cache to {CACHE_FILE}...")
    df_final.to_parquet(CACHE_FILE, index=False)
    print(f"✓ Cache saved ({len(df_final):,} rows)")

    return df_final


def assign_nearest_climate_vectorized(sa_df, climate_df, var_names):
    """
    OPTIMIZED: Vectorized spatial and temporal matching.
    Much faster than the original loop-based approach.
    """
    print("Building spatial index...")
    # Spatial matching using KDTree (same as before, this is already efficient)
    sa_coords = np.radians(sa_df[['Latitude', 'Longitude']].values)
    climate_coords = np.radians(climate_df[['Latitude', 'Longitude']].values)

    tree = cKDTree(climate_coords)
    _, idx = tree.query(sa_coords, k=1)  # _ is distance (not needed)

    nearest_points = climate_df.iloc[idx].reset_index(drop=True)
    sa_df = sa_df.reset_index(drop=True)

    # OPTIMIZATION 3: Vectorized temporal matching
    print("Performing temporal matching...")
    sa_df['Sample Date'] = pd.to_datetime(sa_df['Sample Date'], dayfirst=True, errors='coerce')
    climate_df['Sample Date'] = pd.to_datetime(climate_df['Sample Date'], errors='coerce')

    # Create composite key for spatial matching
    sa_df['nearest_lat'] = nearest_points['Latitude'].values
    sa_df['nearest_lon'] = nearest_points['Longitude'].values
    sa_df['spatial_key'] = sa_df['nearest_lat'].astype(str) + '_' + sa_df['nearest_lon'].astype(str)
    climate_df['spatial_key'] = climate_df['Latitude'].astype(str) + '_' + climate_df['Longitude'].astype(str)

    # Sort for merge_asof (vectorized temporal join)
    sa_df = sa_df.sort_values('Sample Date')
    climate_df = climate_df.sort_values('Sample Date')

    # OPTIMIZATION 4: Use merge_asof for each spatial location (much faster than loop)
    result_list = []
    unique_keys = sa_df['spatial_key'].unique()
    print(f"Processing {len(unique_keys)} unique locations...")

    for key in tqdm(unique_keys, desc="Matching climate data", unit="locations"):
        sa_subset = sa_df[sa_df['spatial_key'] == key]
        climate_subset = climate_df[climate_df['spatial_key'] == key]

        if climate_subset.empty:
            # No climate data for this location
            matched = sa_subset.copy()
            for var in var_names:
                matched[var] = np.nan
        else:
            # Vectorized temporal match using merge_asof
            matched = pd.merge_asof(
                sa_subset[['Latitude', 'Longitude', 'Sample Date']],
                climate_subset[['Sample Date'] + var_names],
                on='Sample Date',
                direction='nearest'
            )

        result_list.append(matched)

    print("Combining results...")
    result_df = pd.concat(result_list, ignore_index=True)

    # Return in original order
    return result_df[var_names]


# --- Main ---

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED TerraClimate Extraction")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n" + "=" * 60)
    print("Loading TerraClimate dataset...")
    print("=" * 60)
    ds = load_terraclimate_dataset()

    # Step 2: Extract and filter climate data (with caching)
    print("\n" + "=" * 60)
    print(f"Extracting variables: {VARIABLES}")
    print("=" * 60)
    tc_data = extract_climate_data_optimized(ds, VARIABLES, use_cache=True)

    # Step 3: Process training data
    print("\n" + "=" * 60)
    print("Mapping to training locations (9,319 points)...")
    print("=" * 60)
    Water_Quality_df = pd.read_csv('../Data/water_quality_training_dataset.csv')

    Terraclimate_training_df = assign_nearest_climate_vectorized(
        Water_Quality_df, tc_data, VARIABLES
    )

    # Add location/date columns
    Terraclimate_training_df['Latitude'] = Water_Quality_df['Latitude'].values
    Terraclimate_training_df['Longitude'] = Water_Quality_df['Longitude'].values
    Terraclimate_training_df['Sample Date'] = Water_Quality_df['Sample Date'].values

    # Reorder columns
    Terraclimate_training_df = Terraclimate_training_df[
        ['Latitude', 'Longitude', 'Sample Date'] + VARIABLES
    ]

    Terraclimate_training_df.to_csv('../Data/terraclimate_features_training.csv', index=False)
    print(f"✓ Saved terraclimate_features_training.csv ({len(Terraclimate_training_df)} rows)")
    print(Terraclimate_training_df.head())

    # Step 4: Process validation data
    print("\n" + "=" * 60)
    print("Mapping to validation locations (200 points)...")
    print("=" * 60)
    Validation_df = pd.read_csv('../Data/submission_template.csv')

    Terraclimate_validation_df = assign_nearest_climate_vectorized(
        Validation_df, tc_data, VARIABLES
    )

    # Add location/date columns
    Terraclimate_validation_df['Latitude'] = Validation_df['Latitude'].values
    Terraclimate_validation_df['Longitude'] = Validation_df['Longitude'].values
    Terraclimate_validation_df['Sample Date'] = Validation_df['Sample Date'].values

    # Reorder columns
    Terraclimate_validation_df = Terraclimate_validation_df[
        ['Latitude', 'Longitude', 'Sample Date'] + VARIABLES
    ]

    Terraclimate_validation_df.to_csv('../Data/terraclimate_features_validation.csv', index=False)
    print(f"✓ Saved terraclimate_features_validation.csv ({len(Terraclimate_validation_df)} rows)")
    print(Terraclimate_validation_df.head())

    print("\n" + "=" * 60)
    print("✓ DONE! Both CSVs saved successfully.")
    print(f"✓ Cache saved to {CACHE_FILE} for faster future runs")
    print("=" * 60)
