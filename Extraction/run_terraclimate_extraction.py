"""
TerraClimate multi-variable extraction script.
Extracts: pet, ppt, tmax, soil, q, aet, def
Saves: terraclimate_features_training.csv, terraclimate_features_validation.csv
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

# --- Functions ---

def load_terraclimate_dataset():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

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
    return ds


def filterg_multi(ds, variables):
    """Extract multiple TerraClimate variables in one loop over time steps."""
    ds_time = ds[variables].sel(time=slice("2011-01-01", "2015-12-31"))

    df_append = []
    for i in tqdm(range(len(ds_time.time)), desc="Filtering time steps"):
        df_slice = ds_time.isel(time=i).to_dataframe().reset_index()
        df_filter = df_slice[
            (df_slice['lat'] > -35.18) & (df_slice['lat'] < -21.72) &
            (df_slice['lon'] > 14.97) & (df_slice['lon'] < 32.79)
        ]
        df_append.append(df_filter)

    df_final = pd.concat(df_append, ignore_index=True)
    print(f"Filtering for {variables} completed")

    df_final['time'] = df_final['time'].astype(str)
    col_mapping = {"lat": "Latitude", "lon": "Longitude", "time": "Sample Date"}
    df_final = df_final.rename(columns=col_mapping)
    return df_final


def assign_nearest_climate_multi(sa_df, climate_df, var_names):
    """Map nearest climate variable values for multiple variables in one pass."""
    sa_coords = np.radians(sa_df[['Latitude', 'Longitude']].values)
    climate_coords = np.radians(climate_df[['Latitude', 'Longitude']].values)

    tree = cKDTree(climate_coords)
    dist, idx = tree.query(sa_coords, k=1)

    nearest_points = climate_df.iloc[idx].reset_index(drop=True)

    sa_df = sa_df.reset_index(drop=True)
    sa_df[['nearest_lat', 'nearest_lon']] = nearest_points[['Latitude', 'Longitude']]

    sa_df['Sample Date'] = pd.to_datetime(sa_df['Sample Date'], dayfirst=True, errors='coerce')
    climate_df['Sample Date'] = pd.to_datetime(climate_df['Sample Date'], dayfirst=True, errors='coerce')

    results = {var: [] for var in var_names}

    for i in tqdm(range(len(sa_df)), desc="Mapping climate values"):
        sample_date = sa_df.loc[i, 'Sample Date']
        nearest_lat = sa_df.loc[i, 'nearest_lat']
        nearest_lon = sa_df.loc[i, 'nearest_lon']

        subset = climate_df[
            (climate_df['Latitude'] == nearest_lat) &
            (climate_df['Longitude'] == nearest_lon)
        ]

        if subset.empty:
            for var in var_names:
                results[var].append(np.nan)
            continue

        nearest_idx = (subset['Sample Date'] - sample_date).abs().idxmin()
        for var in var_names:
            results[var].append(subset.loc[nearest_idx, var])

    return pd.DataFrame(results)


# --- Main ---

if __name__ == "__main__":
    variables = ['pet', 'ppt', 'tmax', 'soil', 'q', 'aet', 'def']

    print("=" * 60)
    print("Loading TerraClimate dataset...")
    print("=" * 60)
    ds = load_terraclimate_dataset()

    print("\n" + "=" * 60)
    print(f"Extracting variables: {variables}")
    print("=" * 60)
    tc_data = filterg_multi(ds, variables)

    # --- Training data ---
    print("\n" + "=" * 60)
    print("Mapping to training locations (9,319 points)...")
    print("=" * 60)
    Water_Quality_df = pd.read_csv('../Data/water_quality_training_dataset.csv')
    Terraclimate_training_df = assign_nearest_climate_multi(Water_Quality_df, tc_data, variables)
    Terraclimate_training_df['Latitude'] = Water_Quality_df['Latitude']
    Terraclimate_training_df['Longitude'] = Water_Quality_df['Longitude']
    Terraclimate_training_df['Sample Date'] = Water_Quality_df['Sample Date']
    Terraclimate_training_df = Terraclimate_training_df[['Latitude', 'Longitude', 'Sample Date'] + variables]
    Terraclimate_training_df.to_csv('../Data/terraclimate_features_training.csv', index=False)
    print(f"Saved terraclimate_features_training.csv ({len(Terraclimate_training_df)} rows)")
    print(Terraclimate_training_df.head())

    # --- Validation data ---
    print("\n" + "=" * 60)
    print("Mapping to validation locations (200 points)...")
    print("=" * 60)
    Validation_df = pd.read_csv('../Data/submission_template.csv')
    Terraclimate_validation_df = assign_nearest_climate_multi(Validation_df, tc_data, variables)
    Terraclimate_validation_df['Latitude'] = Validation_df['Latitude']
    Terraclimate_validation_df['Longitude'] = Validation_df['Longitude']
    Terraclimate_validation_df['Sample Date'] = Validation_df['Sample Date']
    Terraclimate_validation_df = Terraclimate_validation_df[['Latitude', 'Longitude', 'Sample Date'] + variables]
    Terraclimate_validation_df.to_csv('../Data/terraclimate_features_validation.csv', index=False)
    print(f"Saved terraclimate_features_validation.csv ({len(Terraclimate_validation_df)} rows)")
    print(Terraclimate_validation_df.head())

    print("\n" + "=" * 60)
    print("DONE! Both CSVs saved successfully.")
    print("=" * 60)
