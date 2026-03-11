#!/usr/bin/env python3
"""
extract_glim.py
===============
Queries the Macrostrat REST API (https://macrostrat.org) to retrieve
bedrock lithology for every unique site and appends GLIM-equivalent
features directly to Data/training_merged.csv and Data/validation_merged.csv.

Macrostrat is a free, open geology database with global coverage.
It is used here as a programmatic equivalent to the GLIM shapefile
(Hartmann & Moosdorf 2012, doi:10.1029/2012GC004370).

Features added
--------------
  macrostrat_name          – formation/group name (string)
  macrostrat_lith          – high-level lithology type (string)
  macrostrat_descrip       – detailed description (string)
  macrostrat_age_ma        – approximate age in Ma (float)
  glim_class               – GLIM-equivalent class code (ss/sc/su/mt/pa/pb/vb/ev/sm)
  glim_ss                  – siliciclastic (quartzite, sandstone, shale)
  glim_sc                  – carbonate (limestone, dolomite) → high TA/EC
  glim_su                  – unconsolidated sediments (alluvial)
  glim_mt                  – metamorphic
  glim_pa                  – acid plutonic (granite)
  glim_pb                  – basic plutonic (gabbro, Bushveld Complex)
  glim_vb                  – basic volcanic (basalt, Karoo dolerites)
  glim_va                  – acid volcanic
  glim_ev                  – evaporite
  glim_sm                  – mixed sedimentary (Karoo Supergroup)
  glim_is_carbonate        – alias: glim_sc
  glim_is_siliciclastic    – alias: glim_ss  (Eastern Cape = Cape Supergroup)
  glim_is_crystalline      – binary: pa | pb | mt
  glim_weathering_idx      – ion weathering intensity proxy (0–1)

Usage (from project root)
--------------------------
  source venv/bin/activate
  python Extraction/extract_glim.py          # fetch + update CSVs
  python Extraction/extract_glim.py --force  # re-fetch even if cache exists
"""

import time
import json
import argparse
from pathlib import Path

import requests
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent
DATA_DIR   = ROOT / 'Data'
CACHE_FILE = SCRIPT_DIR / 'macrostrat_cache.json'
TRAIN_CSV  = DATA_DIR / 'training_merged.csv'
VAL_CSV    = DATA_DIR / 'validation_merged.csv'

MACROSTRAT_URL = 'https://macrostrat.org/api/geologic_units/map'
REQUEST_DELAY  = 0.35   # seconds between API calls (polite rate limiting)

# ── GLIM-equivalent classes present in South Africa ───────────────────────────
GLIM_CLASSES = ['su', 'ss', 'sm', 'sc', 'ev', 'mt', 'pa', 'pb', 'va', 'vb', 'pi', 'vi', 'py']

# Keyword → GLIM class mapping (applied to lower-cased combined text)
LITH_KEYWORDS: list[tuple[str, str]] = [
    # Carbonate — highest priority (primary TA/EC driver)
    ('limestone',       'sc'),
    ('dolomite',        'sc'),
    ('carbonate',       'sc'),
    ('calcareous',      'sc'),
    ('calc-',           'sc'),
    ('marble',          'sc'),   # metamorphosed carbonate
    # Evaporite
    ('evaporite',       'ev'),
    ('gypsum',          'ev'),
    ('halite',          'ev'),
    ('salt',            'ev'),
    # Siliciclastic — Cape Supergroup (Eastern Cape)
    ('quartzite',       'ss'),
    ('sandstone',       'ss'),
    ('siltstone',       'ss'),
    ('shale',           'ss'),
    ('mudrock',         'ss'),
    ('mudstone',        'ss'),
    ('slate',           'ss'),
    ('phyllite',        'ss'),
    ('siliciclastic',   'ss'),
    ('wacke',           'ss'),
    ('arkose',          'ss'),
    # Mixed sedimentary
    ('conglomerate',    'sm'),
    ('mixed sediment',  'sm'),
    ('diamictite',      'sm'),
    ('tillite',         'sm'),
    # Unconsolidated
    ('unconsolidated',  'su'),
    ('alluvial',        'su'),
    ('quaternary',      'su'),
    ('gravel',          'su'),
    ('sand, gravel',    'su'),
    ('colluvial',       'su'),
    # Basic volcanic (Karoo dolerites, basalts)
    ('basalt',          'vb'),
    ('dolerite',        'vb'),
    ('diabase',         'vb'),
    ('basic volcanic',  'vb'),
    ('mafic',           'vb'),
    ('komatiite',       'vb'),
    # Acid volcanic
    ('rhyolite',        'va'),
    ('andesite',        'va'),
    ('volcanic',        'va'),
    ('pyroclastic',     'va'),
    ('tuff',            'va'),
    ('lava',            'va'),
    # Basic plutonic (Bushveld Complex)
    ('gabbro',          'pb'),
    ('norite',          'pb'),
    ('peridotite',      'pb'),
    ('pyroxenite',      'pb'),
    ('ultramafic',      'pb'),
    ('basic plutonic',  'pb'),
    ('intrusive igneous', 'pb'),
    # Acid plutonic
    ('granite',         'pa'),
    ('granodiorite',    'pa'),
    ('syenite',         'pa'),
    ('tonalite',        'pa'),
    ('acid plutonic',   'pa'),
    ('felsic',          'pa'),
    ('quartz monzonite','pa'),
    # Metamorphic
    ('gneiss',          'mt'),
    ('schist',          'mt'),
    ('amphibolite',     'mt'),
    ('granulite',       'mt'),
    ('migmatite',       'mt'),
    ('metamorphic',     'mt'),
    # Pyroclastic (catch-all if not yet matched)
    ('volcaniclastic',  'py'),
]

# Weathering intensity index (higher = more ions released → higher TA/EC)
WEATHERING_IDX: dict[str, float] = {
    'sc': 1.0,   # carbonate: rapid dissolution
    'ev': 0.9,   # evaporite: very soluble
    'sm': 0.6,   # mixed sedimentary
    'su': 0.5,   # unconsolidated
    'py': 0.4,
    'ss': 0.35,  # siliciclastic: resistant quartz (Eastern Cape → low TA)
    'vi': 0.35,
    'vb': 0.35,
    'va': 0.3,
    'pi': 0.3,
    'pb': 0.3,
    'mt': 0.25,
    'pa': 0.25,  # granite: very resistant
}


# ── Macrostrat API query ───────────────────────────────────────────────────────

def query_macrostrat(lat: float, lon: float) -> dict:
    """Return the best available geological unit for a point."""
    # Try medium scale first (most detailed for SA continental geology)
    for scale in ('medium', None):
        params = {'lat': lat, 'lng': lon}
        if scale:
            params['scale'] = scale
        try:
            r = requests.get(MACROSTRAT_URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get('success', {}).get('data', [])
            # Filter out water-body entries
            data = [d for d in data if d.get('descrip', '').lower() != 'water'
                    and d.get('name', '').lower() != 'water']
            if data:
                u = data[0]
                return {
                    'name':    u.get('name', ''),
                    'lith':    u.get('lith', ''),
                    'descrip': u.get('descrip', ''),
                    't_age':   u.get('t_age'),
                    'b_age':   u.get('b_age'),
                    'scale':   scale or 'auto',
                }
        except Exception as e:
            print(f'    API error (lat={lat}, lon={lon}, scale={scale}): {e}')
    return {}


# ── Lithology classification ───────────────────────────────────────────────────

def classify_glim(name: str, lith: str, descrip: str) -> str:
    """
    Map Macrostrat text fields to a GLIM-equivalent class code.
    Applies keyword rules in priority order (first match wins).
    """
    combined = f'{name} {lith} {descrip}'.lower()
    for keyword, glim_code in LITH_KEYWORDS:
        if keyword in combined:
            return glim_code
    # Fallback based on broad Macrostrat lith categories
    if 'igneous' in combined or 'pluton' in combined:
        return 'pa'
    if 'sedimentary' in combined:
        return 'sm'
    return 'sm'   # unknown → mixed sedimentary (neutral prior)


# ── Cache helpers ──────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: Path) -> None:
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)


# ── Main extraction ────────────────────────────────────────────────────────────

def extract_sites(sites_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Query Macrostrat for each unique site and return a DataFrame with
    raw API fields + derived GLIM columns.
    """
    cache = {} if force else load_cache(CACHE_FILE)

    results = []
    total   = len(sites_df)

    for i, (_, row) in enumerate(sites_df.iterrows(), 1):
        lat, lon = round(float(row['Latitude']), 6), round(float(row['Longitude']), 6)
        key = f'{lat},{lon}'

        if key in cache:
            api_result = cache[key]
        else:
            print(f'  [{i:3d}/{total}] Querying ({lat:.4f}, {lon:.4f}) ...', end=' ')
            api_result = query_macrostrat(lat, lon)
            cache[key] = api_result
            save_cache(cache, CACHE_FILE)
            time.sleep(REQUEST_DELAY)
            name = api_result.get('name', 'no data')[:50]
            print(f'{name}')

        glim_class = classify_glim(
            api_result.get('name', ''),
            api_result.get('lith', ''),
            api_result.get('descrip', ''),
        )

        age_mid = None
        if api_result.get('t_age') and api_result.get('b_age'):
            age_mid = (api_result['t_age'] + api_result['b_age']) / 2

        results.append({
            'Latitude':            lat,
            'Longitude':           lon,
            'macrostrat_name':     api_result.get('name', ''),
            'macrostrat_lith':     api_result.get('lith', ''),
            'macrostrat_descrip':  api_result.get('descrip', ''),
            'macrostrat_age_ma':   age_mid,
            'glim_class':          glim_class,
        })

    return pd.DataFrame(results)


def build_glim_features(site_results: pd.DataFrame) -> pd.DataFrame:
    """Add dummy columns, composite flags, and weathering index."""
    df = site_results.copy()
    for cls in GLIM_CLASSES:
        df[f'glim_{cls}'] = (df['glim_class'] == cls).astype(int)
    df['glim_is_carbonate']     = df['glim_sc']
    df['glim_is_siliciclastic'] = df['glim_ss']
    df['glim_is_crystalline']   = (df['glim_pa'] + df['glim_pb'] + df['glim_mt']).clip(0, 1)
    df['glim_weathering_idx']   = df['glim_class'].map(WEATHERING_IDX).fillna(0.4)
    return df


def update_csv(csv_path: Path, site_features: pd.DataFrame) -> None:
    """Merge GLIM features into the merged CSV and overwrite it."""
    df = pd.read_csv(csv_path)

    # Drop any existing GLIM / macrostrat columns
    drop_cols = [c for c in df.columns
                 if c.startswith('glim_') or c.startswith('macrostrat_')]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f'  Dropped {len(drop_cols)} existing columns.')

    # Merge on site coordinates (round to match)
    df['_lat_r'] = df['Latitude'].round(6)
    df['_lon_r'] = df['Longitude'].round(6)
    site_features = site_features.rename(
        columns={'Latitude': '_lat_r', 'Longitude': '_lon_r'}
    )
    df = df.merge(site_features, on=['_lat_r', '_lon_r'], how='left')
    df.drop(columns=['_lat_r', '_lon_r'], inplace=True)

    new_cols = [c for c in df.columns
                if c.startswith('glim_') or c.startswith('macrostrat_')]
    df.to_csv(csv_path, index=False)
    print(f'  Saved {csv_path.name} ({len(df):,} rows, +{len(new_cols)} GLIM cols)')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fetch Macrostrat geology and append GLIM features to merged CSVs.'
    )
    parser.add_argument('--force', action='store_true',
                        help='Re-query all sites even if already cached.')
    args = parser.parse_args()

    # Collect all unique sites from both splits
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    all_sites = (
        pd.concat([
            train_df[['Latitude', 'Longitude']],
            val_df[['Latitude', 'Longitude']],
        ])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f'Fetching geology for {len(all_sites)} unique sites via Macrostrat ...')
    if not args.force and CACHE_FILE.exists():
        cached_n = len(load_cache(CACHE_FILE))
        print(f'  ({cached_n} already cached in {CACHE_FILE.name})')

    site_raw  = extract_sites(all_sites, force=args.force)
    site_feat = build_glim_features(site_raw)

    print('\nGLIM class distribution across all sites:')
    print(site_feat['glim_class'].value_counts().to_string())
    print()
    print('Macrostrat coverage (sample):')
    sample_cols = ['Latitude', 'Longitude', 'macrostrat_name', 'glim_class', 'glim_weathering_idx']
    print(site_feat[sample_cols].to_string(index=False, max_rows=12))

    # Write to both CSVs
    print('\nUpdating CSVs ...')
    for label, csv_path in [('training', TRAIN_CSV), ('validation', VAL_CSV)]:
        print(f'\n── {label.upper()} ──────────────────────────────')
        update_csv(csv_path, site_feat.copy())

    print('\nGLIM/Macrostrat extraction complete.')
    print(f'Cache saved to: {CACHE_FILE}')


if __name__ == '__main__':
    main()
