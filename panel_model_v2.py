"""
panel_model_v2.py — Panel model + SoilGrids + target-encoded macrostrat_name.

Schema:
    y_it = f_between(time-invariant_i, X_bar_i)   [Ridge on site-level aggregates]
         + f_within(X_it - X_bar_i, month)         [XGB on within-site deviations]

Run this script directly to smoke-test on synthetic data.
Use train() / predict() to integrate into a real pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# ── Feature lists (edit here to change what goes where) ──────────────────────

# Columns that never change within a site
TIME_INVARIANT_COLS = [
    "elevation_m",
    "wc_bio1", "wc_bio4", "wc_bio12", "wc_bio15",
    "macrostrat_age_ma",
    "is_karoo_supergroup", "is_cape_supergroup",
    "is_beaufort_group", "is_ecca_group", "is_dwyka_group", "is_karoo_dolerite",
    # SoilGrids 250m v2 — surface layers
    "sg_phh2o_0_5cm", "sg_phh2o_5_15cm",
    "sg_cec_0_5cm",   "sg_cec_5_15cm",
    "sg_clay_0_5cm",  "sg_clay_5_15cm",
    "sg_soc_0_5cm",   "sg_soc_5_15cm",
    "sg_sand_0_5cm",  "sg_sand_5_15cm",
    "sg_silt_0_5cm",  "sg_silt_5_15cm",
    "sg_bdod_0_5cm",  "sg_bdod_5_15cm",
    "sg_nitrogen_0_5cm", "sg_nitrogen_5_15cm",
    # Target-encoded macrostrat (added per-target at fit time)
    "macrostrat_te_Total Alkalinity",
    "macrostrat_te_Electrical Conductance",
    "macrostrat_te_Dissolved Reactive Phosphorus",

    # HydroRivers data
    "upland_skm",
    "log_upland_skm",
    "strahler_order",
]

# Columns that vary within a site (will be decomposed into mean + deviation)
TIME_VARIANT_COLS = [
    "swir22", "NDMI", "MNDWI", "thermal", "NDWI", "NDVI", "BSI",
    "red_turbidity", "evi", "lswi", "blue_red_ratio", "nir_red_ratio",
    "pet", "ppt", "tmax", "soil", "q", "aet", "def",
    "aridity", "seasonal_wetness", "water_stress",
    "runoff_coeff", "baseflow_index", "carbonate_dissolution",
]

TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]

SITE_COLS = ["Latitude", "Longitude"]


# ── Target encoding ───────────────────────────────────────────────────────────

def target_encode_macrostrat(train: pd.DataFrame, val: pd.DataFrame,
                              targets: list, n_splits: int = 5) -> tuple:
    """
    Out-of-fold target encoding of macrostrat_name to avoid leakage.
    Adds columns macrostrat_te_<target> to both train and val.
    Safe to call even if macrostrat_name is missing (columns are left as NaN).
    """
    from sklearn.model_selection import KFold

    if "macrostrat_name" not in train.columns:
        return train, val

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train = train.copy()
    val   = val.copy()

    for target in targets:
        col = f"macrostrat_te_{target}"
        train[col] = np.nan

        for tr_idx, oof_idx in kf.split(train):
            means = train.iloc[tr_idx].groupby("macrostrat_name")[target].mean()
            train.loc[train.index[oof_idx], col] = (
                train.iloc[oof_idx]["macrostrat_name"].map(means)
            )

        global_mean = train[target].mean()
        train[col] = train[col].fillna(global_mean)

        full_means = train.groupby("macrostrat_name")[target].mean()
        val[col] = val["macrostrat_name"].map(full_means).fillna(global_mean)

    return train, val


# ── Core helpers ──────────────────────────────────────────────────────────────

def _site_key(df: pd.DataFrame) -> pd.Series:
    return df["Latitude"].astype(str) + "_" + df["Longitude"].astype(str)


def _select_present(cols: list, df: pd.DataFrame) -> list:
    """Return only columns that exist in df."""
    return [c for c in cols if c in df.columns]


def compute_site_means(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Compute per-site mean for each column in `cols`.
    Returns a DataFrame indexed by site key with columns like 'col__site_mean'.
    """
    present = _select_present(cols, df)
    df = df.copy()
    df["__site__"] = _site_key(df)
    means = df.groupby("__site__")[present].mean()
    means.columns = [f"{c}__site_mean" for c in present]
    return means


def add_within_deviations(df: pd.DataFrame, site_means: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Add deviation columns (X_it - X_bar_i) for each column in `cols`.
    `site_means` must be a DataFrame as returned by compute_site_means().
    """
    present = _select_present(cols, df)
    df = df.copy()
    df["__site__"] = _site_key(df)
    df = df.join(site_means, on="__site__")
    for c in present:
        mean_col = f"{c}__site_mean"
        if mean_col in df.columns:
            df[f"{c}__dev"] = df[c] - df[mean_col]
    return df


def build_between_features(df: pd.DataFrame, site_means: pd.DataFrame) -> np.ndarray:
    """
    Features for the between model: time-invariant cols + site means of time-variant cols.
    Requires site_means to be joined to df already (via add_within_deviations).
    """
    invariant = _select_present(TIME_INVARIANT_COLS, df)
    site_mean_cols = [c for c in df.columns if c.endswith("__site_mean")]
    month_dummies = [c for c in df.columns if c.startswith("month_")]
    all_between = invariant + site_mean_cols + month_dummies
    return df[all_between].values.astype(float), all_between


def build_within_features(df: pd.DataFrame) -> np.ndarray:
    """
    Features for the within model: deviation cols + month dummies.
    """
    dev_cols = [c for c in df.columns if c.endswith("__dev")]
    month_dummies = [c for c in df.columns if c.startswith("month_")]
    all_within = dev_cols + month_dummies
    return df[all_within].values.astype(float), all_within


# ── Model class ───────────────────────────────────────────────────────────────

class PanelWaterQualityModel:
    """
    Per-target panel decomposition model.

    Attributes set after fit():
        site_means_   : pd.DataFrame  — training site means (reused at predict time)
        between_scaler_: StandardScaler
        within_scaler_ : StandardScaler
        between_model_ : Ridge
        within_model_  : XGBRegressor
    """

    def __init__(
        self,
        target: str,
        ridge_alpha: float = 500.0,
        xgb_params: dict | None = None,
        between_weight: float = 0.5,
    ):
        self.target = target
        self.ridge_alpha = ridge_alpha
        self.between_weight = between_weight
        self.xgb_params = xgb_params or dict(
            n_estimators=400, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=3.0, reg_lambda=3.0,
            random_state=42, n_jobs=-1,
        )

    def _prepare(self, df: pd.DataFrame, site_means: pd.DataFrame | None = None):
        """Shared feature construction for fit and predict."""
        # Add month dummies
        df = df.copy()
        if "month" in df.columns:
            month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True).astype(float)
            df = pd.concat([df, month_dummies], axis=1)

        # Compute or reuse site means for time-variant cols
        if site_means is None:
            site_means = compute_site_means(df, TIME_VARIANT_COLS)

        df = add_within_deviations(df, site_means, TIME_VARIANT_COLS)
        return df, site_means

    def fit(self, df: pd.DataFrame):
        df, self.site_means_ = self._prepare(df)

        X_between, self.between_cols_ = build_between_features(df, self.site_means_)
        X_within, self.within_cols_ = build_within_features(df)
        y = df[self.target].values

        # Impute NaNs (fit imputers on training data)
        self.between_imputer_ = SimpleImputer(strategy='median').fit(X_between)
        self.within_imputer_  = SimpleImputer(strategy='median').fit(X_within)
        X_between = self.between_imputer_.transform(X_between)
        X_within  = self.within_imputer_.transform(X_within)

        # Between model (Ridge on full data)
        self.between_scaler_ = StandardScaler()
        X_b_scaled = self.between_scaler_.fit_transform(X_between)
        self.between_model_ = Ridge(alpha=self.ridge_alpha)
        self.between_model_.fit(X_b_scaled, y)
        y_between_hat = self.between_model_.predict(X_b_scaled)

        # Within model (XGB on residuals)
        self.within_scaler_ = StandardScaler()
        X_w_scaled = self.within_scaler_.fit_transform(X_within)
        residuals = y - self.between_weight * y_between_hat
        self.within_model_ = XGBRegressor(**self.xgb_params)
        self.within_model_.fit(X_w_scaled, residuals)

        return self

    def predict(self, df: pd.DataFrame, site_means: pd.DataFrame | None = None) -> np.ndarray:
        """
        Predict on new data.
        For known sites: reuse training site_means_ (pass nothing).
        For new sites:   pass site_means computed from the new site's observations.
        If site_means=None, falls back to training site_means_ (unknown sites get NaN-filled → median).
        """
        # Merge training site means with any new ones
        known = self.site_means_
        if site_means is not None:
            known = pd.concat([known, site_means[~site_means.index.isin(known.index)]])

        df, _ = self._prepare(df, site_means=known)

        X_between, _ = build_between_features(df, known)
        X_within, _ = build_within_features(df)

        X_between = self.between_imputer_.transform(X_between)
        X_within  = self.within_imputer_.transform(X_within)

        y_between = self.between_model_.predict(
            self.between_scaler_.transform(X_between)
        )
        y_within = self.within_model_.predict(
            self.within_scaler_.transform(X_within)
        )
        return self.between_weight * y_between + y_within


# ── Spatial cross-validation ──────────────────────────────────────────────────

def spatial_cv(df: pd.DataFrame, target: str, n_splits: int = 5, **model_kwargs) -> float:
    """
    Leave-one-group-of-sites-out CV. All observations of a site are held out together.
    Returns mean R² across folds.
    """
    df = df.copy()
    df["__site__"] = _site_key(df)
    groups = df["__site__"].values

    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = PanelWaterQualityModel(target=target, **model_kwargs)
        model.fit(train_df)

        # Compute site means for test sites from their own observations (X only, no y)
        test_site_means = compute_site_means(test_df, TIME_VARIANT_COLS)
        y_pred = model.predict(test_df, site_means=test_site_means)
        y_true = test_df[target].values

        score = r2_score(y_true, y_pred)
        scores.append(score)
        print(f"  Fold {fold+1}: R²={score:.4f}  (n_test={len(test_idx)})")

    mean_score = np.mean(scores)
    print(f"  → Mean R² = {mean_score:.4f}")
    return mean_score


# ── Smoke test on synthetic data ──────────────────────────────────────────────

def _make_synthetic_data(n_sites: int = 30, obs_per_site: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    Generate synthetic panel data that mimics the real structure.
    Ground truth: y = 2*elevation + 0.5*ppt_bar + 0.3*ppt_dev + noise
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sites):
        lat = rng.uniform(-34, -25)
        lon = rng.uniform(17, 30)
        # Time-invariant site properties
        elevation = rng.uniform(100, 2000)
        wc_bio12 = rng.uniform(200, 800)  # annual precip baseline
        age_ma = rng.uniform(100, 2500)
        is_karoo = int(rng.random() < 0.4)
        is_cape = int(rng.random() < 0.2)
        is_beaufort = int(is_karoo and rng.random() < 0.6)
        is_ecca = int(is_karoo and not is_beaufort and rng.random() < 0.3)
        is_dwyka = 0
        is_dolerite = 0

        # Site mean for time-variant features
        ppt_site_mean = rng.uniform(20, 150)

        for t in range(obs_per_site):
            month = rng.integers(1, 13)
            ppt = ppt_site_mean + rng.normal(0, 20)
            tmax = rng.uniform(15, 35)
            ndvi = rng.uniform(-0.2, 0.6)
            pet = rng.uniform(50, 200)

            # Ground truth signal (known structure for testing)
            y = (
                50
                + 0.05 * elevation
                + 0.2 * wc_bio12
                + 0.3 * (ppt - ppt_site_mean)   # within-site effect
                - 10 * is_cape
                + 15 * is_karoo
                + rng.normal(0, 10)
            )

            rows.append(dict(
                Latitude=lat, Longitude=lon,
                month=month,
                # time-invariant
                elevation_m=elevation, wc_bio1=tmax, wc_bio4=500.0,
                wc_bio12=wc_bio12, wc_bio15=60.0,
                macrostrat_age_ma=age_ma,
                macrostrat_name=rng.choice(["Beaufort Group","Ecca Group","Cape Supergroup","Dolerite","Other"]),
                is_karoo_supergroup=is_karoo, is_cape_supergroup=is_cape,
                is_beaufort_group=is_beaufort, is_ecca_group=is_ecca,
                is_dwyka_group=is_dwyka, is_karoo_dolerite=is_dolerite,
                # SoilGrids (synthetic)
                sg_phh2o_0_5cm=rng.uniform(5,8),  sg_phh2o_5_15cm=rng.uniform(5,8),
                sg_cec_0_5cm=rng.uniform(5,30),   sg_cec_5_15cm=rng.uniform(5,30),
                sg_clay_0_5cm=rng.uniform(50,400), sg_clay_5_15cm=rng.uniform(50,400),
                sg_soc_0_5cm=rng.uniform(5,80),   sg_soc_5_15cm=rng.uniform(5,40),
                # time-variant
                ppt=ppt, tmax=tmax, NDVI=ndvi, pet=pet,
                swir22=rng.uniform(5000, 15000),
                NDMI=rng.uniform(-0.3, 0.3), MNDWI=rng.uniform(-0.4, 0.2),
                thermal=rng.uniform(40000, 50000), NDWI=rng.uniform(-0.3, 0.2),
                BSI=rng.uniform(-0.1, 0.2), red_turbidity=rng.uniform(-0.1, 0.1),
                evi=rng.uniform(-0.1, 0.5), lswi=rng.uniform(-0.3, 0.3),
                blue_red_ratio=rng.uniform(0.5, 1.5), nir_red_ratio=rng.uniform(0.5, 2.5),
                soil=rng.uniform(0, 50), q=rng.uniform(0, 100),
                aet=rng.uniform(10, 150), **{"def": rng.uniform(0, 200)},
                aridity=rng.uniform(0.1, 3.0), seasonal_wetness=rng.uniform(0, 1),
                water_stress=rng.uniform(0, 1), runoff_coeff=rng.uniform(0, 0.5),
                baseflow_index=rng.uniform(0, 1), carbonate_dissolution=rng.uniform(0, 2),
                **{"Total Alkalinity": y,
                   "Electrical Conductance": y * 4 + rng.normal(0, 20),
                   "Dissolved Reactive Phosphorus": max(0, y * 0.1 + rng.normal(0, 5))},
            ))

    return pd.DataFrame(rows)


def smoke_test():
    print("=" * 60)
    print("SMOKE TEST — synthetic panel data")
    print("=" * 60)

    df = _make_synthetic_data(n_sites=40, obs_per_site=20)
    print(f"Data: {df.shape[0]} rows, {df['Latitude'].nunique()} sites\n")

    # Target-encode macrostrat (train=val for smoke test — just checks it runs)
    df, _ = target_encode_macrostrat(df, df.iloc[:0].copy(), TARGETS)

    for target in TARGETS:
        print(f"--- {target} ---")
        score = spatial_cv(df, target=target, n_splits=5)
        print()

    print("Smoke test passed — schema works end-to-end.")


if __name__ == "__main__":
    smoke_test()
