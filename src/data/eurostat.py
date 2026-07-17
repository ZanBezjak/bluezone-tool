"""
src/data/eurostat.py
--------------------
Reusable functions for downloading, parsing and saving Eurostat datasets.
Called by src/data/__main__.py via: python -m src.data
"""

import gzip
import logging
import shutil
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/"
    "data/{code}?format=TSV&compressed=true"
)

SUPRANATIONAL = [
    "EU", "EU27_2020", "EU28",
    "EA", "EA19", "EA20", "EA21",
    "EEA", "EEA31",
]

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "aerobic_activity": {
        "code":    "hlth_ehis_pe2e",
        "filters": {"unit": "PC", "sex": "T", "age": "TOTAL",
                    "isced11": "TOTAL", "duration": "MN_GE150"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% meeting 150+ min/week. 2014 and 2019 only.",
    },
    "family_contact": {
        "code":    "ilc_scp09",
        "filters": {"unit": "PC", "sex": "T", "age": "Y_GE16",
                    "isced11": "TOTAL", "frequenc": "WEEK",
                    "pers_cat": "FAM_REL"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% getting together with family at least weekly. 2015 only.",
    },
    "gdp_per_capita": {
        "code":    "nama_10r_2gdp",
        "filters": {"unit": "PPS_HAB_EU27_2020"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "GDP per capita in PPS. NUTS0-2 confirmed.",
    },
    "heavy_drinking": {
        "code":    "hlth_ehis_al3e",
        "filters": {"unit": "PC", "sex": "T", "age": "TOTAL",
                    "isced11": "TOTAL", "frequenc": "NVR_NM12"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% drinking heavily at least once/week. 2014 and 2019 only.",
    },
    "internet_usage": {
        "code":    "isoc_r_iuse_i",
        "filters": {"unit": "PC_IND", "indic_is": "I_IDAY"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% regularly using internet. NUTS2 confirmed.",
    },
    "life_expectancy": {
        "code":    "demo_r_mlifexp",
        "filters": {"unit": "YR", "sex": "T",
                    "age": "Y_LT1"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "Life expectancy at birth. No total sex — use F and M separately then average.",
    },
    "life_satisfaction": {
        "code":    "ilc_pw01",
        "filters": {"unit": "RTG", "sex": "T",
                    "age": "Y_GE16", 'isced11': 'TOTAL'},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "Meaning of life",
    },
    "obesity_rate": {
        "code":    "sdg_02_10",
        "filters": {"unit": "PC", "bmi": "BMI_GE30"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% population BMI>=30.",
    },
    "poverty_risk": {
        "code":    "ilc_li41",
        "filters": {"unit": "PC"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "No sub-dimensions — just filter unit. NUTS2 confirmed.",
    },
    "smoking": {
        "code":    "hlth_ehis_sk3e",
        "filters": {"unit": "PC", "sex": "T", "age": "TOTAL",
                    "isced11": "TOTAL", "smoking": "TOTAL"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% smokers total (light + heavy). 2014 and 2019 only.",
    },
    "social_contact": {
        "code":    "ilc_scp17",
        "filters": {"unit": "PC", "sex": "T", "age": "Y_GE16",
                    "isced11": "TOTAL"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% with no one to discuss personal matters. 2015 only.",
    },
    "social_support": {
        "code":    "ilc_scp15",
        "filters": {"unit": "PC", "sex": "T", "age": "Y_GE16",
                    "isced11": "TOTAL"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "% with no one to ask for help. 2015 only.",
    },
    "social_trust": {
        "code":    "ilc_pw03",
        "filters": {"unit": "RTG", "sex": "T", "age": "Y_GE16",
                    "isced11": "TOTAL", "statinfo": "AVG",
                    "domain": "OTH"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "Mean trust rating. 2013, 2018, 2022 only.",
    },
    "unemployment_rate": {
        "code":    "lfst_r_lfu3rt",
        "filters": {"unit": "PC", "sex": "T", "age": "Y_GE15",
                    "isced11": "TOTAL"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   "Total unemployment rate. NUTS2 confirmed.",
    },
    "fruit_veggies": {
        "code":    "hlth_ehis_fv3e",
        "filters": {"unit": "PC", "sex": "T", "age": "TOTAL",
                    "isced11": "TOTAL", "n_portion": "GE5"},
        "level":   "all",
        "keep_aggregates": False,
        "notes":   ">5 portions per day. 2014, 2019 only.",
    },
}

# Reverse lookup: Eurostat code -> friendly name
eurostat_datasets = {config["code"]: name for name, config in DATASETS.items()}


# ── Download & Save ──────────────────────────────────────────────────────────────────

def download_and_save(code: str, label: str, force: bool = False) -> tuple[Path, bool]:
    """
    Download a Eurostat dataset by code and cache as TSV in RAW_DIR.
    Skips download if file already exists.
    Returns (path, was_downloaded).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DIR / f"{label}.tsv"

    if save_path.exists() and not force:
        log.info(f"⏭️ SKIP {label} — already cached")
        return save_path, False

    url = BASE_URL.format(code=code)
    log.info(f"Downloading {label} ({code})...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    content = r.content
    if content[:2] != b"\x1f\x8b":
        raise ValueError(
            f"[{code}] response is not gzip — got {content[:30]!r}"
        )

    text = gzip.decompress(content).decode("utf-8")
    save_path.write_text(text, encoding="utf-8")
    log.info(f"💾 Saved {label} -> {save_path}")
    return save_path, True


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the Eurostat first column (comma-separated dimensions + geo)
    into individual columns. Renames geo TIME_PERIOD to geo.
    """
    first_col = df.columns[0]
    dims = first_col.split(",")
    df[dims] = df[first_col].str.split(",", expand=True)
    df = df.drop(columns=[first_col])
    df.columns = [str(c).strip() for c in df.columns]

    geo_col = next(
        (c for c in df.columns if "geo" in c.lower()),
        dims[-1]
    )
    df = df.rename(columns={geo_col: "geo"})
    df["geo"] = df["geo"].str.strip()
    return df


# ── Transform ─────────────────────────────────────────────────────────────────

def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep geo + year columns only. Strip value flags (e.g. '12.3 p' -> 12.3).
    Cast year column names to int. Set geo as index.
    """
    year_cols = [c for c in df.columns if str(c).strip().lstrip("-").isdigit()]
    df = df[["geo"] + year_cols].copy()
    df.columns = ["geo"] + [int(c) for c in year_cols]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r"([\d.]+)", expand=False),
            errors="coerce",
        )
    return df.set_index("geo").sort_index()


def filter_geo(df: pd.DataFrame, level: str, keep_aggregates: bool) -> pd.DataFrame:
    """Filter rows by NUTS level, excluding supranational aggregates."""
    if not keep_aggregates:
        df = df[~df.index.isin(SUPRANATIONAL)]

    valid = (
        df.index.str.match(r"^[A-Z]{2}") &
        ~df.index.str.match(r"^EA\d+$") &
        ~df.index.str.match(r"^EU")
    )
    df = df[valid]

    if level == "nuts2":
        return df[df.index.str.len() == 4]
    elif level == "country":
        return df[df.index.str.len() == 2]
    elif level == "all":
        return df[df.index.str.len().isin([2, 3, 4])]
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_dataset(name: str, config: dict, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Download, filter, reshape and save one dataset to data/interim/."""
    log.info(f"⏳ Processing {name} ({config['code']})")

    save_path, _ = download_and_save(config["code"], name, force=force)
    df = pd.read_csv(save_path, sep="\t", na_values=[":", ": "])

    df = parse(df)

    for col, val in config["filters"].items():
        if col in df.columns:
            df = df[df[col] == val]
        else:
            log.warning(f" ⚠️ '{col}' not found in {name} — skipping filter")

    df = to_wide(df)
    df = filter_geo(df, config["level"], config["keep_aggregates"])

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / f"{name}.parquet"
    df.to_parquet(path)

    level_counts = df.index.str.len().value_counts().sort_index().to_dict()
    missing_pct  = (
        df.isnull().sum().sum() / df.size * 100
        if df.size > 0 else float("nan")
    )

    log.info(
        f" ✅ + {df.shape} | "
        f"nuts0={level_counts.get(2, 0)} "
        f"nuts1={level_counts.get(3, 0)} "
        f"nuts2={level_counts.get(4, 0)} | "
        f"years={min(df.columns)}-{max(df.columns)} | "
        f"missing={missing_pct:.1f}%"
    )

    return df, {
        "dataset":     name,
        "code":        config["code"],
        "level":       config["level"],
        "nuts0":       level_counts.get(2, 0),
        "nuts1":       level_counts.get(3, 0),
        "nuts2":       level_counts.get(4, 0),
        "year_min":    int(min(df.columns)),
        "year_max":    int(max(df.columns)),
        "n_years":     len(df.columns),
        "shape":       str(df.shape),
        "missing_pct": round(missing_pct, 1),
        "notes":       config.get("notes", ""),
    }


def clear_interim():
    """Wipe and recreate data/interim/."""
    shutil.rmtree(INTERIM_DIR, ignore_errors=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Cleared data/interim/")


def main(force: bool = False):
    """Run the full data collection pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("=" * 60)
    log.info("🌍 Blue Zone Explorer — Data Collection")
    log.info("=" * 60)

    report = []
    for name, config in DATASETS.items():
        _, stats = process_dataset(name, config, force=force)
        report.append(stats)

    report_df = pd.DataFrame(report)
    report_path = INTERIM_DIR / "collection_report.csv"
    report_df.to_csv(report_path, index=False)

    log.info("=" * 60)
    log.info("\n" + report_df.to_string(index=False))
    log.info(f"📊 Report saved to {report_path}")
    log.info("🎉 All datasets collected successfully")
