"""
src/data/preprocessing.py
--------------------------
Preprocessing pipeline for Blue Zone Explorer.

Steps:
    1. Load interim datasets
    2. Recode NUTS codes to 2021 vintage
    3. Restrict to time window 2013-2024
    4. Time series imputation (interpolate -> country mean -> ffill/bfill)
    5. Drop unreliable regions (BA, XK, AL)
    6. Spatial imputation (NUTS1 -> NUTS0 -> global mean)
    7. Compute mean + slope features
    8. Standardise with StandardScaler
    9. Save to data/processed/ and data/artifacts/

Feature notes:
    - 13 features: 12 computed as mean + slope, 1 as mean only (social_support)
    - social_support treated as cultural constant (single 2015 observation)
    - life_expectancy kept out of PCA entirely, used for validation only

NUTS vintage notes:
    - Finnish NUTS 2016 codes (FI13, FI18, FI1A) recoded to NUTS 2021
      equivalents (FI1D, FI1C, FI1B) for geographic alignment

Run via:
    python -m src.data.preprocessing
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

# ── Config ────────────────────────────────────────────────────────────────────

YEAR_START = 2013
YEAR_END   = 2024
WINDOW     = list(range(YEAR_START, YEAR_END + 1))

# All PCA features — order determines column order in feature matrix
PCA_FEATURES = [
    "unemployment_rate",   # economic stress         — NUTS2, full time series
    "poverty_risk",        # material security       — NUTS2, full time series
    "internet_usage",      # modernisation proxy     — NUTS2, full time series
    "gdp_per_capita",      # wealth                  — NUTS2, full time series
    "social_trust",        # community cohesion      — country, full time series
    "life_satisfaction",   # subjective wellbeing    — country, full time series
    "aerobic_activity",    # physical activity       — country, 2 points (2014, 2019)
    "heavy_drinking",      # lifestyle behaviour     — country, 2 points (2014, 2019)
    "smoking",             # lifestyle behaviour     — country, 2 points (2014, 2019)
    "fruit_veggies",       # diet quality            — country, 2 points (2014, 2019)
    "social_contact",      # social isolation proxy  — country, 2 points (2013, 2015)
    "family_contact",      # Blue Zone principle     — country, 2 points (2015, 2022)
    "obesity_rate",        # lifestyle outcome       — country, 4 points
    "social_support",      # cultural constant       — country, 1 point (2015)
]

# Features with only 1 time point — compute mean only, no slope
# Treated as cultural constants that change slowly at regional level
MEAN_ONLY_FEATURES = ["social_support"]

# Validation — never enters PCA, used to validate score after training
VALIDATION_FEATURES = ["life_expectancy"]

# Regions dropped due to too many missing values across multiple datasets
DROP_REGIONS = ["BA", "BA0", "BA00", "XK", "XK0", "XK00", "AL", "AL0", "AL01", "AL02", "AL03"]

# Reference dataset — most complete NUTS2 coverage, defines spatial index
REFERENCE_DATASET = "unemployment_rate"

# NUTS vintage harmonisation — recode NUTS 2016 Finnish codes to NUTS 2021
# Source: Eurostat NUTS correspondence tables + Wikipedia
NUTS_RECODE = {
    "FI13": "FI1D",
    "FI18": "FI1C",
    "FI1A": "FI1B",
}


# ── Step 1 — Load ─────────────────────────────────────────────────────────────

def load_interim(names: list[str]) -> dict[str, pd.DataFrame]:
    """Load interim parquet files into a dict keyed by dataset name."""
    datasets = {}
    for name in names:
        path = INTERIM_DIR / f"{name}.parquet"
        if path.exists():
            datasets[name] = pd.read_parquet(path)
            log.info(f"  ✅ {name}: {datasets[name].shape}")
        else:
            log.warning(f"  ⚠️  {name}: not found at {path}")
    return datasets


# ── Step 2 — NUTS recode ──────────────────────────────────────────────────────

def recode_nuts(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Harmonise NUTS codes across datasets to NUTS 2021 vintage.
    Some datasets contain both old (NUTS 2016) and new (NUTS 2021) codes
    for the same region. In that case the old code row is dropped.
    """
    result = {}
    for name, df in datasets.items():
        for old, new in NUTS_RECODE.items():
            if old in df.index:
                if new in df.index:
                    # Both old and new exist — drop old, keep new
                    df = df.drop(index=old)
                    log.info(f"  {name:<22} dropped {old} ({new} already exists)")
                else:
                    # Only old exists — rename to new
                    df = df.rename(index={old: new})
                    log.info(f"  {name:<22} recoded {old} → {new}")
        result[name] = df
    return result


# ── Step 3 — Window ───────────────────────────────────────────────────────────

def apply_window(datasets: dict[str, pd.DataFrame],
                 window: list[int]) -> dict[str, pd.DataFrame]:
    """Restrict all datasets to years within the time window."""
    windowed = {}
    for name, df in datasets.items():
        cols = [y for y in window if y in df.columns]
        windowed[name] = df[cols].copy()
        log.info(f"  {name:<22} {df.shape} → {windowed[name].shape}")
    return windowed


# ── Step 4 — Time series imputation ──────────────────────────────────────────

def impute_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values within each region's time series:
    1. Linear interpolation for gaps <= 2 consecutive years
    2. Country mean (first 2 chars of geo code) for larger gaps
       Note: uses mean of available NUTS2 peers within the country,
       not the NUTS0 country row, for internal consistency
    3. Forward fill then backward fill for edge gaps
    """
    # Pass 1 — interpolate isolated gaps
    df = df.interpolate(axis=1, limit=2, limit_direction="both")

    # Pass 2 — country mean for larger gaps
    df["country"] = df.index.str[:2]
    for year in df.columns[:-1]:
        country_mean = df.groupby("country")[year].transform("mean")
        df[year]     = df[year].fillna(country_mean)
    df = df.drop(columns=["country"])

    # Pass 3 — edge gaps (e.g. UK trailing years post-Brexit)
    df = df.ffill(axis=1).bfill(axis=1)

    return df


# ── Step 5 — Drop unreliable regions ─────────────────────────────────────────

def drop_unreliable(datasets: dict[str, pd.DataFrame],
                    regions: list[str]) -> dict[str, pd.DataFrame]:
    """Drop regions with too many missing values across multiple datasets."""
    result = {}
    for name, df in datasets.items():
        before = len(df)
        result[name] = df[~df.index.isin(regions)]
        after  = len(result[name])
        if before != after:
            log.info(f"  {name:<22} dropped {before - after} regions")
        else:
            result[name] = df
    return result


# ── Step 6 — Spatial imputation ───────────────────────────────────────────────

def impute_missing_regions(df: pd.DataFrame,
                           reference_index: pd.Index) -> pd.DataFrame:
    """
    Expand dataset to match reference_index by imputing missing regions:
    1. NUTS1 mean (first 3 chars of geo code)
    2. NUTS0 mean (first 2 chars)
    3. Global mean
    """
    df = df.reindex(reference_index)

    df["nuts1"] = df.index.str[:3]
    df["nuts0"] = df.index.str[:2]
    feature_cols = [c for c in df.columns if c not in ["nuts1", "nuts0"]]

    for col in feature_cols:
        nuts1_mean = df.groupby("nuts1")[col].transform("mean")
        df[col]    = df[col].fillna(nuts1_mean)
        nuts0_mean = df.groupby("nuts0")[col].transform("mean")
        df[col]    = df[col].fillna(nuts0_mean)
        df[col]    = df[col].fillna(df[col].mean())

    return df.drop(columns=["nuts1", "nuts0"])


# ── Step 7 — Compute features ─────────────────────────────────────────────────

def compute_features(df: pd.DataFrame,
                     mean_only: bool = False) -> pd.DataFrame:
    """
    Compute mean and OLS slope per region across available time points.

    Parameters
    ----------
    df : pd.DataFrame
        Wide dataframe with years as columns.
    mean_only : bool
        If True, compute mean only (no slope). Used for cultural constants
        with a single time point (social_support).

    Returns
    -------
    pd.DataFrame
        Columns: mean (always), slope (unless mean_only=True).

    Notes
    -----
    For datasets with only 2 time points (aerobic_activity, heavy_drinking,
    smoking, fruit_veggies, social_contact, family_contact), slope is
    computed as rise/run between the two observations. Methodologically
    valid but less robust than a multi-year OLS trend.
    Documented as a limitation in the methodology section.
    """
    years = np.array(df.columns, dtype=float)
    means = df.mean(axis=1)

    if mean_only:
        return pd.DataFrame({"mean": means}, index=df.index)

    slopes = df.apply(
        lambda row: linregress(years, row.values)[0], axis=1
    )
    return pd.DataFrame({"mean": means, "slope": slopes}, index=df.index)


# ── Full imputation pipeline ──────────────────────────────────────────────────

def run_imputation(windowed: dict[str, pd.DataFrame],
                   pca_features: list[str],
                   drop: list[str],
                   reference_dataset: str) -> tuple[dict, pd.Index]:
    """
    Full imputation pipeline:
    1. Time series imputation per dataset
    2. Drop unreliable regions
    3. Spatial imputation to align all datasets to reference index
    """
    imputed = {}

    log.info("── Step 4: Time series imputation ──")
    for name in pca_features:
        before        = windowed[name].isnull().sum().sum()
        imputed[name] = impute_time_series(windowed[name].copy())
        after         = imputed[name].isnull().sum().sum()
        log.info(f"  {name:<22} {before:>5} missing → {after:>5} missing")

    log.info("── Step 5: Drop unreliable regions ──")
    cleaned = drop_unreliable(
        {name: imputed[name] for name in pca_features}, drop
    )
    for name in pca_features:
        imputed[name] = cleaned[name]

    reference_index = imputed[reference_dataset].index
    log.info(
        f"── Step 6: Spatial imputation "
        f"(reference: '{reference_dataset}', {len(reference_index)} regions) ──"
    )
    for name in pca_features:
        before        = len(imputed[name])
        imputed[name] = impute_missing_regions(imputed[name], reference_index)
        after         = len(imputed[name])
        missing       = imputed[name].isnull().sum().sum()
        log.info(
            f"  {name:<22} {before:>4} → {after:>4} regions | {missing} missing"
        )

    return imputed, reference_index


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    all_names = PCA_FEATURES + VALIDATION_FEATURES

    # ── Step 1 — Load ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("🌍 Blue Zone Explorer — Preprocessing")
    log.info("=" * 60)
    log.info("── Step 1: Load interim datasets ──")
    interim = load_interim(all_names)

    # ── Step 2 — NUTS recode ──────────────────────────────────────────────────
    log.info("── Step 2: NUTS vintage harmonisation ──")
    interim = recode_nuts(interim)

    # ── Step 3 — Window ───────────────────────────────────────────────────────
    log.info(f"── Step 3: Apply time window {YEAR_START}–{YEAR_END} ──")
    windowed = apply_window(interim, WINDOW)

    # ── Steps 4–6 — Imputation ────────────────────────────────────────────────
    imputed, reference_index = run_imputation(
        windowed          = windowed,
        pca_features      = PCA_FEATURES,
        drop              = DROP_REGIONS,
        reference_dataset = REFERENCE_DATASET,
    )

    # ── Step 7 — Compute features ─────────────────────────────────────────────
    log.info("── Step 7: Compute mean + slope features ──")
    features = {}
    for name in PCA_FEATURES:
        mean_only      = name in MEAN_ONLY_FEATURES
        feat           = compute_features(imputed[name], mean_only=mean_only)
        feat.columns   = [f"{name}_{c}" for c in feat.columns]
        features[name] = feat
        tag = "mean only" if mean_only else "mean + slope"
        log.info(f"  ✅ {name:<22} {feat.shape}  [{tag}]")

    feature_matrix = pd.concat(features.values(), axis=1, join="inner")
    log.info(
        f"  Feature matrix: {feature_matrix.shape} | "
        f"missing: {feature_matrix.isnull().sum().sum()}"
    )

    # ── Validation feature ────────────────────────────────────────────────────
    log.info("── Validation: life expectancy ──")
    le_imputed  = impute_time_series(windowed["life_expectancy"].copy())
    le_imputed  = impute_missing_regions(le_imputed, reference_index)
    le_features = compute_features(le_imputed)
    le_features.columns = ["life_expectancy_mean", "life_expectancy_slope"]
    log.info(
        f"  life_expectancy: {le_features.shape} | "
        f"missing: {le_features.isnull().sum().sum()}"
    )

    # ── Step 8 — Standardise ──────────────────────────────────────────────────
    log.info("── Step 8: Standardise ──")
    scaler = StandardScaler()
    feature_matrix_scaled = pd.DataFrame(
        scaler.fit_transform(feature_matrix),
        index   = feature_matrix.index,
        columns = feature_matrix.columns,
    )

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    log.info("  ✅ Feature matrix scaled and scaler saved")

    # ── Step 9 — Save ─────────────────────────────────────────────────────────
    log.info("── Step 9: Save outputs ──")

    feature_matrix.to_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    log.info(f"  ✅ feature_matrix.parquet              {feature_matrix.shape}")

    feature_matrix_scaled.to_parquet(
        PROCESSED_DIR / "feature_matrix_scaled.parquet"
    )
    log.info(
        f"  ✅ feature_matrix_scaled.parquet       {feature_matrix_scaled.shape}"
    )

    le_features.to_parquet(
        PROCESSED_DIR / "life_expectancy_validation.parquet"
    )
    log.info(
        f"  ✅ life_expectancy_validation.parquet  {le_features.shape}"
    )

    # Save imputed time series for year-by-year scoring
    for name in PCA_FEATURES:
        df = imputed[name].copy()
        df.columns = [str(c) for c in df.columns]
        df.to_parquet(PROCESSED_DIR / f"{name}_imputed.parquet")
    log.info(f"  ✅ imputed time series ({len(PCA_FEATURES)} datasets)")

    log.info("=" * 60)
    log.info("🎉 Preprocessing complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
