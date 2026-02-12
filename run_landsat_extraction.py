"""
Landsat extraction script - extracts all 6 bands (including red & blue).
Saves: landsat_features_training.csv, landsat_features_validation.csv
Includes batch saving every 100 rows to avoid losing progress on failure.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from tqdm import tqdm
import os

def compute_Landsat_values(row):
    lat = row['Latitude']
    lon = row['Longitude']
    date = pd.to_datetime(row['Sample Date'], dayfirst=True, errors='coerce')

    bbox_size = 0.00089831
    bbox = [
        lon - bbox_size / 2,
        lat - bbox_size / 2,
        lon + bbox_size / 2,
        lat + bbox_size / 2
    ]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime="2011-01-01/2015-12-31",
        query={"eo:cloud_cover": {"lt": 10}},
    )

    items = search.item_collection()

    if not items:
        return pd.Series({
            "nir": np.nan, "green": np.nan, "red": np.nan, "blue": np.nan,
            "swir16": np.nan, "swir22": np.nan
        })

    try:
        sample_date_utc = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
        items = sorted(
            items,
            key=lambda x: abs(pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc)
        )
        selected_item = pc.sign(items[0])

        bands_of_interest = ["blue", "green", "red", "nir08", "swir16", "swir22"]
        data = stac_load([selected_item], bands=bands_of_interest, bbox=bbox).isel(time=0)

        blue = data["blue"].astype("float")
        green = data["green"].astype("float")
        red = data["red"].astype("float")
        nir = data["nir08"].astype("float")
        swir16 = data["swir16"].astype("float")
        swir22 = data["swir22"].astype("float")

        results = {}
        for name, band in [("blue", blue), ("green", green), ("red", red),
                           ("nir", nir), ("swir16", swir16), ("swir22", swir22)]:
            val = float(band.median(skipna=True).values)
            results[name] = val if val != 0 else np.nan

        return pd.Series(results)

    except Exception as e:
        return pd.Series({
            "nir": np.nan, "green": np.nan, "red": np.nan, "blue": np.nan,
            "swir16": np.nan, "swir22": np.nan
        })


def extract_with_checkpoints(df, output_path, checkpoint_every=100):
    """Extract Landsat features with periodic checkpoint saves."""
    checkpoint_path = output_path + ".checkpoint.csv"

    # Resume from checkpoint if it exists
    start_idx = 0
    if os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        start_idx = len(existing)
        print(f"Resuming from checkpoint at row {start_idx}")
        results_list = [existing]
    else:
        results_list = []

    batch = []
    for i in tqdm(range(start_idx, len(df)), desc="Extracting Landsat bands",
                  initial=start_idx, total=len(df)):
        row = df.iloc[i]
        result = compute_Landsat_values(row)
        batch.append(result)

        # Save checkpoint periodically
        if len(batch) >= checkpoint_every:
            batch_df = pd.DataFrame(batch)
            results_list.append(batch_df)
            combined = pd.concat(results_list, ignore_index=True)
            combined.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved at row {i + 1}/{len(df)}")
            batch = []

    # Save remaining
    if batch:
        batch_df = pd.DataFrame(batch)
        results_list.append(batch_df)

    all_results = pd.concat(results_list, ignore_index=True)
    return all_results


def compute_indices(df):
    """Compute spectral indices from extracted bands."""
    eps = 1e-10
    df['NDMI'] = (df['nir'] - df['swir16']) / (df['nir'] + df['swir16'] + eps)
    df['MNDWI'] = (df['green'] - df['swir22']) / (df['green'] + df['swir22'] + eps)
    df['NDVI'] = (df['nir'] - df['red']) / (df['nir'] + df['red'] + eps)
    df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + eps)
    df['BSI'] = ((df['swir22'] + df['red']) - (df['nir'] + df['blue'])) / \
                ((df['swir22'] + df['red']) + (df['nir'] + df['blue']) + eps)
    return df


if __name__ == "__main__":
    # --- Training data ---
    print("=" * 60)
    print("Extracting Landsat features for TRAINING data (9,319 points)")
    print("This will take approximately 6-7 hours.")
    print("=" * 60)

    Water_Quality_df = pd.read_csv('water_quality_training_dataset.csv')
    train_features = extract_with_checkpoints(
        Water_Quality_df, 'landsat_features_training.csv', checkpoint_every=100
    )
    train_features = compute_indices(train_features)

    # Add location/date columns
    train_features['Latitude'] = Water_Quality_df['Latitude']
    train_features['Longitude'] = Water_Quality_df['Longitude']
    train_features['Sample Date'] = Water_Quality_df['Sample Date']
    train_features = train_features[['Latitude', 'Longitude', 'Sample Date',
                                      'blue', 'green', 'red', 'nir', 'swir16', 'swir22',
                                      'NDMI', 'MNDWI', 'NDVI', 'NDTI', 'BSI']]
    train_features.to_csv('landsat_features_training.csv', index=False)
    print(f"\nSaved landsat_features_training.csv ({len(train_features)} rows)")
    print(train_features.head())

    # Clean up checkpoint
    if os.path.exists('landsat_features_training.csv.checkpoint.csv'):
        os.remove('landsat_features_training.csv.checkpoint.csv')

    # --- Validation data ---
    print("\n" + "=" * 60)
    print("Extracting Landsat features for VALIDATION data (200 points)")
    print("=" * 60)

    Validation_df = pd.read_csv('submission_template.csv')
    val_features = extract_with_checkpoints(
        Validation_df, 'landsat_features_validation.csv', checkpoint_every=50
    )
    val_features = compute_indices(val_features)

    val_features['Latitude'] = Validation_df['Latitude']
    val_features['Longitude'] = Validation_df['Longitude']
    val_features['Sample Date'] = Validation_df['Sample Date']
    val_features = val_features[['Latitude', 'Longitude', 'Sample Date',
                                  'blue', 'green', 'red', 'nir', 'swir16', 'swir22',
                                  'NDMI', 'MNDWI', 'NDVI', 'NDTI', 'BSI']]
    val_features.to_csv('landsat_features_validation.csv', index=False)
    print(f"\nSaved landsat_features_validation.csv ({len(val_features)} rows)")
    print(val_features.head())

    # Clean up checkpoint
    if os.path.exists('landsat_features_validation.csv.checkpoint.csv'):
        os.remove('landsat_features_validation.csv.checkpoint.csv')

    print("\n" + "=" * 60)
    print("DONE! Both Landsat CSVs saved with blue, green, red, nir, swir16, swir22 + indices.")
    print("=" * 60)
