#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import unicodedata
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from tqdm.auto import tqdm

from dedication_classifier import classify  # local import

log = logging.getLogger("pipeline")

# ===== DEFAULT SETTINGS (can be overridden by CLI) =============================
INPUT_PATH = Path("input")
OUTPUT_PATH = Path("output")
CACHE_PATH = Path("cache")

CSV_PATH = INPUT_PATH / "STRAD_ITA_20251010.csv"
DUGS_CSV = INPUT_PATH / "dug.txt"

SAVE_LLM_EVERY = 1000
VERBOSE = False

# Base columns (old CSVs may only have these)
BASE_LLM_COLS = ["label", "category_id", "category", "gender"]
# New full set, including who did the classification
LLM_COLS = BASE_LLM_COLS + ["source"]
# ==============================================================================


# --------------------------------------------------------------------------------------
# DUG (street type) handling
# --------------------------------------------------------------------------------------
def load_dug_prefixes_from_txt(path: Path) -> List[str]:
    """
    Load DUG prefixes from a plain text file, one prefix per line.
    Returns a de-duplicated list sorted by descending length
    (so the longest match wins).
    """
    if not path.exists():
        raise FileNotFoundError(f"DUGs file not found: {path.resolve()}")

    lines = path.read_text(encoding="utf-8").splitlines()

    dugs: List[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        dugs.append(s)

    return sorted(set(dugs), key=lambda x: (-len(x), x))


def _str_strip(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Odonym pipeline → LLM classifications → SQLite database."
    )
    p.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name to use for the classifier.",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to input CSV.")
    p.add_argument("--dugs", type=Path, default=DUGS_CSV, help="Path to DUGs list.")
    p.add_argument(
        "--output-dir", type=Path, default=OUTPUT_PATH, help="Output directory."
    )
    p.add_argument(
        "--cache-dir", type=Path, default=CACHE_PATH, help="Cache directory."
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def standardize_label(value: Union[str, pd.Series, Dict[str, Any], None]):
    """
    Standardize dedication labels.

    Handles:
      - Unicode normalization
      - Apostrophe variants → ASCII '
      - Whitespace normalization
      - Trailing vowel + apostrophe → grave accent
      - Acute vowels → grave vowels (Italian normalization)
        e.g. É → È
    """

    _APOS_VARIANTS_RE = re.compile(r"[’`´ʹʻʽ‛]")
    _WS_RE = re.compile(r"\s+")
    _TRAILING_VOWEL_APOS_RE = re.compile(r"(?iu)([AEIOU])'$")

    _GRAVE_FROM_APOS = {
        "A": "À", "E": "È", "I": "Ì", "O": "Ò", "U": "Ù",
        "a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù",
    }

    # Acute → grave normalization
    _ACUTE_TO_GRAVE = str.maketrans({
        "Á": "À", "É": "È", "Í": "Ì", "Ó": "Ò", "Ú": "Ù",
        "á": "à", "é": "è", "í": "ì", "ó": "ò", "ú": "ù",
    })

    def _standardize_one(label: str) -> str:
        if label is None:
            return ""

        s = str(label)

        # Unicode normalization
        s = unicodedata.normalize("NFKC", s)

        # Normalize apostrophes
        s = _APOS_VARIANTS_RE.sub("'", s)

        # Normalize whitespace
        s = s.strip()
        s = _WS_RE.sub(" ", s)

        if not s:
            return ""

        # Convert trailing vowel + apostrophe
        toks = s.split(" ")
        out = []
        for t in toks:
            m = _TRAILING_VOWEL_APOS_RE.search(t)
            if m:
                v = m.group(1)
                accented = _GRAVE_FROM_APOS.get(v, v)
                t = _TRAILING_VOWEL_APOS_RE.sub(accented, t)
            out.append(t)

        s = " ".join(out)

        # Convert acute vowels to grave vowels
        s = s.translate(_ACUTE_TO_GRAVE)

        # Final NFC normalization
        s = unicodedata.normalize("NFC", s)

        return s

    if value is None:
        return ""

    if isinstance(value, str):
        return _standardize_one(value)

    if isinstance(value, pd.Series):
        def _apply_series(x):
            if x is None or pd.isna(x):
                return None
            y = _standardize_one(x)
            return y if y.strip() else None
        return value.apply(_apply_series)

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            kk = _standardize_one(k)
            if not kk:
                continue
            if kk not in out:
                out[kk] = v
        return out

    return _standardize_one(str(value))


def expand_abbreviated_labels(
    series: pd.Series,
    *,
    min_full_tokens: int = 2,
) -> tuple[pd.Series, Dict[str, str]]:
    """
    Expand abbreviations like:
      - "A Manzoni", "A. Manzoni", "A.Manzoni"
      - "G V Vico", "G. V. Vico", "G.V.Vico"
      - "A. De Gasperi"
    into the most frequent matching full-form label already present in `series`.

    Matching rule (no surname particles):
      Abbr = initials + tail_tokens
      Full matches if:
        - first letters of first len(initials) tokens match initials sequence
        - last len(tail_tokens) tokens match tail_tokens exactly
      Among matches, pick the most frequent full label.
    Returns: (expanded_series, mapping_used)
    """
    s = series.dropna().astype(str).str.strip()
    vc = s.value_counts()

    _WS_RE = re.compile(r"\s+")
    _CLEAN_RE = re.compile(r"[^\w\s'\-\.]")  # keep dots for initials, keep ' and -

    def _normalize(label: str) -> str:
        x = (label or "").strip()
        # "A.Manzoni" -> "A. Manzoni" (space after initial dot when stuck)
        x = re.sub(r"(?i)\b([A-Z])\.(?=[A-ZÀ-ÖØ-öø-ÿ])", r"\1. ", x)
        x = _CLEAN_RE.sub(" ", x)
        x = _WS_RE.sub(" ", x).strip()
        return x

    def _toks_upper(label: str) -> list[str]:
        x = _normalize(label)
        if not x:
            return []
        toks = x.split()
        out: list[str] = []
        for t in toks:
            tt = t[:-1] if t.endswith(".") else t  # strip trailing dot: "A." -> "A"
            out.append(tt.upper())
        return out

    def _is_initial(tok: str) -> bool:
        return len(tok) == 1 and tok.isalpha()

    def _abbr_key(label: str):
        toks = _toks_upper(label)
        if len(toks) < 2:
            return None

        initials: list[str] = []
        i = 0
        while i < len(toks) and _is_initial(toks[i]):
            initials.append(toks[i])
            i += 1

        if not initials or i >= len(toks):
            return None

        tail = tuple(toks[i:])
        if not tail:
            return None

        return (tuple(initials), tail)

    # Precompute candidate full labels
    full_labels: list[tuple[str, list[str], int]] = []
    for lab in vc.index:
        ftoks = _toks_upper(lab)
        if len(ftoks) < min_full_tokens:
            continue
        # reject labels that are basically initials-only before the last token
        if len(ftoks) >= 2 and all(_is_initial(t) for t in ftoks[:-1]):
            continue
        full_labels.append((lab, ftoks, int(vc.get(lab, 0))))

    mapping: Dict[str, str] = {}
    for lab_abbr in vc.index:
        key = _abbr_key(lab_abbr)
        if not key:
            continue

        initials, tail = key
        n_i = len(initials)
        n_t = len(tail)

        best: tuple[int, str] | None = None  # (count, full_label)

        for lab_full, ftoks, cnt in full_labels:
            if len(ftoks) < n_i + n_t:
                continue

            # initials sequence must match first letters of first n_i tokens
            ok_init = True
            for j in range(n_i):
                if not ftoks[j] or ftoks[j][0] != initials[j]:
                    ok_init = False
                    break
            if not ok_init:
                continue

            # tail tokens must match the end exactly
            if tuple(ftoks[-n_t:]) != tail:
                continue

            if (best is None) or (cnt > best[0]):
                best = (cnt, lab_full)

        if best:
            mapping[lab_abbr] = best[1]

    if not mapping:
        return series, {}

    expanded = series.replace(mapping)
    return expanded, mapping


# -----------------------------------------------------------------------------
# Cache handling (root-level mapping: { "<label>": {...}, ... })
# -----------------------------------------------------------------------------
def _load_cache_data(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return blob if isinstance(blob, dict) else {}


def _save_cache_data(path: Path, data: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False))


def _load_existing_llm(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load existing classifications from CSV.

    Requires only BASE_LLM_COLS so old CSVs without 'source' still work.
    'source' is treated as optional and may be None.
    """
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return {}

    required = set(BASE_LLM_COLS)
    if not required.issubset(set(df.columns)):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        label = r["label"]
        cat_val = r.get("category_id")
        cat_id = None
        if isinstance(cat_val, str) and cat_val.strip() != "":
            try:
                cat_id = int(cat_val)
            except Exception:
                cat_id = None
        out[label] = {
            "category_id": cat_id,
            "category": r.get("category"),
            "gender": r.get("gender"),
            "source": r.get("source") or None,  # may be missing in older CSVs
        }
    return out


# -----------------------------------------------------------------------------
# Filename tag helpers
# -----------------------------------------------------------------------------
_TAG_SAFE_RE_1 = re.compile(r"\s+")
_TAG_SAFE_RE_2 = re.compile(r"[^a-z0-9_.-]")


def _model_tag(model: str | None) -> str:
    """
    Return a filesystem-safe tag for outputs/caches.
    """
    tag = (model or "").strip()
    if "/" in tag:
        tag = tag.split("/")[-1]
    tag = tag.lower()
    tag = _TAG_SAFE_RE_1.sub("_", tag)
    tag = _TAG_SAFE_RE_2.sub("-", tag)
    tag = tag.strip("._-")
    return tag


# -----------------------------------------------------------------------------
# SQL backend
# -----------------------------------------------------------------------------
def _df_with_nones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaNs with Python None so sqlite3 gets proper NULLs.
    """
    return df.where(pd.notna(df), None)


def build_sql_db(
    db_path: Path,
    prov: pd.DataFrame,
    mun: pd.DataFrame,
    streets: pd.DataFrame,
    entities: pd.DataFrame,
) -> None:
    """
    Create / overwrite an SQLite database with provinces, municipalities,
    streets and entities tables.
    Normalized (no redundant IDs):
      - provinces:      sigla_prov (PK)
      - municipalities: istat_comune (PK), FK -> provinces(sigla_prov)
      - entities:       entity_id (PK), label UNIQUE
      - streets:        progressivo_nazionale (PK, from PROGRESSIVO_NAZIONALE),
                        FK -> municipalities(istat_comune),
                        FK -> entities(entity_id)
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.executescript(
        """
        DROP TABLE IF EXISTS streets;
        DROP TABLE IF EXISTS municipalities;
        DROP TABLE IF EXISTS provinces;
        DROP TABLE IF EXISTS entities;
        """
    )

    # Provinces: sigla_prov is the primary key
    cur.execute(
        """
        CREATE TABLE provinces (
            sigla_prov   TEXT PRIMARY KEY,
            provincia    TEXT,
            cod_comune   TEXT
        );
        """
    )

    # Municipalities: istat_comune is the primary key, refers to provinces(sigla_prov)
    cur.execute(
        """
        CREATE TABLE municipalities (
            istat_comune  TEXT PRIMARY KEY,
            comune        TEXT,
            sigla_prov    TEXT,
            FOREIGN KEY (sigla_prov) REFERENCES provinces(sigla_prov)
        );
        """
    )

    # Entities: entity_id primary key, label unique
    cur.execute(
        """
        CREATE TABLE entities (
            entity_id        TEXT PRIMARY KEY,
            label            TEXT UNIQUE,
            gender       TEXT,
            category_id  INTEGER,
            category     TEXT
        );
        """
    )

    # Streets: progressivo_nazionale = PROGRESSIVO_NAZIONALE, link to istat_comune and entities(entity_id)
    cur.execute(
        """
        CREATE TABLE streets (
            progressivo_nazionale INTEGER PRIMARY KEY,
            odonimo               TEXT,
            istat_comune          TEXT,
            dedication_entity_id  TEXT,
            FOREIGN KEY (istat_comune)         REFERENCES municipalities(istat_comune),
            FOREIGN KEY (dedication_entity_id) REFERENCES entities(entity_id)
        );
        """
    )

    # Insert provinces
    prov_rec_df = _df_with_nones(
        prov[["sigla_prov", "provincia", "cod_comune"]]
    )
    prov_records = list(prov_rec_df.itertuples(index=False, name=None))
    cur.executemany(
        "INSERT INTO provinces (sigla_prov, provincia, cod_comune) VALUES (?, ?, ?);",
        prov_records,
    )

    # Insert municipalities
    mun_rec_df = _df_with_nones(
        mun[["istat_comune", "comune", "sigla_prov"]]
    )
    mun_records = list(mun_rec_df.itertuples(index=False, name=None))
    cur.executemany(
        """
        INSERT INTO municipalities (
            istat_comune, comune, sigla_prov
        ) VALUES (?, ?, ?);
        """,
        mun_records,
    )

    # Insert entities with hash entity_id
    ent_df = _df_with_nones(
        entities[["entity_id", "label", "gender", "category_id", "category"]]
    )
    ent_records = list(ent_df.itertuples(index=False, name=None))
    cur.executemany(
        """
        INSERT INTO entities (
            entity_id, label, gender, category_id, category
        ) VALUES (?, ?, ?, ?, ?);
        """,
        ent_records,
    )

    # Insert streets: only normalized fields
    street_df = _df_with_nones(
        streets[
            [
                "progressivo_nazionale",
                "ODONIMO",
                "istat_comune",
                "dedication_entity_id",
            ]
        ]
    )

    # Plain Python tuples so sqlite3 sees native types
    street_rows = list(street_df.itertuples(index=False, name=None))
    # Each row = (progressivo_nazionale, ODONIMO, istat_comune, dedication_entity_id)

    cur.executemany(
        """
        INSERT INTO streets (
            progressivo_nazionale,
            odonimo,
            istat_comune,
            dedication_entity_id
        ) VALUES (?, ?, ?, ?);
        """,
        street_rows,
    )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    # Silence noisy libs (in case classify uses them)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    model_tag = _model_tag(args.model)

    OUT_LLM_CLASSES = args.output_dir / f"{model_tag}_classifications.csv"
    LLM_CACHE_FILE = args.cache_dir / f"{model_tag}_cache.json"
    OUT_DB = args.output_dir / f"{model_tag}_streets.sqlite"

    odonym_prefixes = load_dug_prefixes_from_txt(args.dugs)
    log.info("Loaded %d odonym prefixes from %s", len(odonym_prefixes), args.dugs.name)

    # Build a regex to strip DUG at the start (case-insensitive)
    if odonym_prefixes:
        dug_pattern = "|".join(
            sorted(
                {re.escape(d.strip()) for d in odonym_prefixes if str(d).strip()},
                key=len,
                reverse=True,
            )
        )
        dug_regex = rf"^\s*(?:{dug_pattern})\s*(?P<core>.*)$"
    else:
        dug_pattern = ""
        dug_regex = r"^(?!)"  # matches nothing

    log.info("[1/3] Reading CSV (utf-8-sig): %s", args.csv)
    df = pd.read_csv(
        args.csv,
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
        dtype={
            "CODICE_ISTAT": str,
        },
    )
    log.info("Loaded rows=%d cols=%d", len(df), len(df.columns))

    # DUG filter: keep only ODONIMO that start with one of the DUGs
    if "ODONIMO" not in df.columns:
        raise SystemExit("ERROR: 'ODONIMO' column missing.")

    odonimo_series = df["ODONIMO"].astype(str)
    ded_core_series_full = odonimo_series.str.extract(
        dug_regex, flags=re.IGNORECASE
    )["core"]

    before = len(df)

    # core when DUG is known (NaN when unknown)
    core_known = ded_core_series_full

    # fallback when DUG is unknown: drop first token
    core_fallback = (
        odonimo_series.astype(str)
        .str.strip()
        .str.replace(r"^\S+\s*", "", regex=True)
        .str.strip()
    )

    mask_known = core_known.notna()
    mask_unknown = ~mask_known

    known_count = int(mask_known.sum())
    unknown_count = int(mask_unknown.sum())

    # final core: known DUG -> stripped core, unknown DUG -> first-word-dropped core
    ded_core_series_full = core_known.fillna(core_fallback)

    log.info("Known DUG rows: %d / %d", known_count, before)
    log.info("Unknown DUG rows (kept, first word dropped): %d / %d", unknown_count, before)

    # -------------------------------------------------------------------------
    # Report UNKNOWN street types (first word) + counts
    # -------------------------------------------------------------------------
    unknown = df.loc[mask_unknown, "ODONIMO"].astype(str).str.strip()

    first_token = (
        unknown
        .str.extract(r"^([^\s]+)", expand=False)
        .fillna("")
        .str.upper()
    )

    nonempty = first_token != ""
    unknown = unknown.loc[nonempty]
    first_token = first_token.loc[nonempty]

    unknown_counts = (
        first_token.value_counts()
        .rename_axis("unknown_type")
        .reset_index(name="count")
    )

    if unknown_count > 0:
        EXAMPLES_PER_TYPE = 3
        examples = (
            pd.DataFrame({"ODONIMO": unknown, "type": first_token})
            .groupby("type")["ODONIMO"]
            .apply(lambda s: list(s.head(EXAMPLES_PER_TYPE)))
            .reset_index(name="examples")
        )

        TOP_EXAMPLE_TYPES = 10
        log.info("Examples for top unknown DUG types:")
        for _, r in (
            examples.merge(
                unknown_counts.head(TOP_EXAMPLE_TYPES).rename(columns={"unknown_type": "type"}),
                on="type",
                how="inner",
            )
            .sort_values("count", ascending=False)
            .iterrows()
        ):
            log.info("  %s (%d): %s", r["type"], int(r["count"]), r["examples"])

    # Provinces (simple aggregation: first value per SIGLA)
    for col in ("SIGLA", "PROVINCIA", "CODICE_COMUNE"):
        if col not in df.columns:
            df[col] = None
    prov = df[["SIGLA", "PROVINCIA", "CODICE_COMUNE"]].copy()
    prov.columns = ["sigla_prov", "provincia", "cod_comune"]
    prov = prov.groupby("sigla_prov", dropna=False, as_index=False).agg(
        provincia=("provincia", "first"),
        cod_comune=("cod_comune", "first"),
    )
    prov["sigla_prov"] = prov["sigla_prov"].astype(str).str.strip()
    prov["provincia"] = prov["provincia"].astype(str).str.strip()

    # Municipalities (simple aggregation: first value per CODICE_ISTAT)
    for col in ("CODICE_ISTAT", "COMUNE", "SIGLA"):
        if col not in df.columns:
            df[col] = None
    mun = df[["CODICE_ISTAT", "COMUNE", "SIGLA"]].copy()
    mun.columns = ["istat_comune", "comune", "sigla_prov"]
    mun["istat_comune"] = mun["istat_comune"].astype(str).str.strip()
    mun = mun.groupby("istat_comune", dropna=False, as_index=False).agg(
        comune=("comune", "first"),
        sigla_prov=("sigla_prov", "first"),
    )
    mun["comune"] = mun["comune"].astype(str).str.strip()
    mun["sigla_prov"] = mun["sigla_prov"].astype(str).str.strip()

    # Streets: only keep what's needed and use PROGRESSIVO_NAZIONALE directly
    street_cols = [
        "PROGRESSIVO_NAZIONALE",
        "ODONIMO",
        "CODICE_ISTAT",
    ]

    # require PROGRESSIVO_NAZIONALE to exist in the input
    if "PROGRESSIVO_NAZIONALE" not in df.columns:
        raise SystemExit("ERROR: 'PROGRESSIVO_NAZIONALE' column missing.")

    streets = df[[c for c in street_cols if c in df.columns]].copy()
    for c in street_cols:
        if c not in streets.columns:
            streets[c] = None

    # progressivo_nazionale = PROGRESSIVO_NAZIONALE as INTEGER
    streets["progressivo_nazionale"] = pd.to_numeric(
        streets["PROGRESSIVO_NAZIONALE"], errors="coerce"
    )

    # Convert to native Python int where possible, else keep None
    streets["progressivo_nazionale"] = streets["progressivo_nazionale"].apply(
        lambda v: int(v) if pd.notna(v) else None
    )

    # Normalize CODICE_ISTAT and expose as istat_comune for FK to municipalities
    streets["CODICE_ISTAT"] = (
        streets["CODICE_ISTAT"].astype(str).str.strip().replace({"": None})
    )
    # Keep same cleaned values for istat_comune (avoid "None" string)
    streets["istat_comune"] = streets["CODICE_ISTAT"]

    # Dedications & labels (strip DUG via same regex)
    log.info("[2/3] Extracting dedications & running LLM classifications…")
    streets["dedication_label"] = (
        ded_core_series_full
        .reindex(streets.index)
        .fillna("")
    )

    # Normalize empty dedication labels to NULL (so FK doesn't fail)
    streets["dedication_label"] = streets["dedication_label"].apply(
        lambda x: x if isinstance(x, str) and x.strip() != "" else None
    )

    # Standardize name spellings (apostrophes, accents, whitespace)
    streets["dedication_label"] = standardize_label(streets["dedication_label"])

    before_labels = streets["dedication_label"].copy()

    streets["dedication_label"], mapping = expand_abbreviated_labels(streets["dedication_label"])
    if mapping:
        changed_rows = (streets["dedication_label"].fillna("") != before_labels.fillna("")).sum()
        log.info(
            "Expanded %d abbreviated dedication labels (changed rows=%d).",
            len(mapping),
            int(changed_rows),
        )

    # Safety redundant standardization
    streets["dedication_label"] = standardize_label(streets["dedication_label"])

    # -------------------------------------------------------------------------
    # Report most frequent unique dedication labels BEFORE classification
    # -------------------------------------------------------------------------
    TOP_FREQ = 20  # change as you like

    ded_series = streets["dedication_label"].dropna().astype(str).str.strip()
    ded_counts = ded_series.value_counts()

    log.info("Top %d most frequent dedication labels (unique cores):", TOP_FREQ)
    for lab, cnt in ded_counts.head(TOP_FREQ).items():
        log.info("  %7d  %s", int(cnt), lab)

    # Unique labels for LLM (exclude NULL/empty)
    labels_orig = sorted(
        {
            v.strip()
            for v in streets["dedication_label"].dropna().astype(str)
            if v.strip()
        }
    )

    # Load caches
    existing = _load_existing_llm(OUT_LLM_CLASSES)
    cache_data: Dict[str, Any] = _load_cache_data(LLM_CACHE_FILE)

    # Standardize cache keys so lookups match standardized labels
    # Prevents cache misses if older runs used different accent/apostrophe logic.
    existing = standardize_label(existing)
    cache_data = standardize_label(cache_data)

    llm_rows: List[Dict[str, Any]] = []
    newly_classified = 0

    for lab in tqdm(labels_orig, desc="Classifying labels", unit="label"):
        if lab in existing:
            # Use existing classification (may or may not have 'source')
            row = {"label": lab}
            row.update(existing[lab])
            llm_rows.append(row)
            continue

        cached = cache_data.get(lab)
        if isinstance(cached, dict):
            # JSON cache (contains timestamp + classification + maybe source)
            res = cached
        else:
            # Fresh classification
            try:
                res = classify(lab, model=args.model)
            except Exception as e:
                log.warning("Classifier failed for %r: %s", lab, e)
                res = {}

            cache_data[lab] = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                **res,
            }
            newly_classified += 1

        llm_rows.append(
            {
                "label": lab,
                "category_id": res.get("category_id"),
                "category": res.get("category"),
                "gender": res.get("gender"),
                "source": res.get("source"),  # <- who did it: heuristic / llm / None
            }
        )

        if SAVE_LLM_EVERY and newly_classified and newly_classified % SAVE_LLM_EVERY == 0:
            _save_cache_data(LLM_CACHE_FILE, cache_data)

    if newly_classified > 0 or not LLM_CACHE_FILE.exists():
        _save_cache_data(LLM_CACHE_FILE, cache_data)

    # Build CSV with explicit columns (including 'source')
    df_local = pd.DataFrame(llm_rows, columns=LLM_COLS)
    _atomic_write_csv(df_local, OUT_LLM_CLASSES)
    log.info(
        "Wrote LLM classifications to %s (%d total, %d new)",
        OUT_LLM_CLASSES,
        len(df_local),
        newly_classified,
    )

    # Index LLM results
    llm_by_label: Dict[str, Dict[str, Any]] = {}
    for _, r in df_local.iterrows():
        cat_val = r.get("category_id")
        cat_id = None
        if pd.notna(cat_val) and str(cat_val).strip() != "":
            try:
                cat_id = int(cat_val)
            except Exception:
                cat_id = None
        llm_by_label[r["label"]] = {
            "category_id": cat_id,
            "category": r.get("category"),
            "gender": r.get("gender"),
            "source": r.get("source"),
        }

    # Entities (LLM-only metadata)
    enriched_rows: List[Dict[str, Any]] = []
    for lab in labels_orig:
        llm_meta = llm_by_label.get(lab, {})
        enriched_rows.append(
            {
                "label": lab,
                "gender": llm_meta.get("gender"),
                "category_id": llm_meta.get("category_id"),
                "category": llm_meta.get("category"),
                # you could also store "source" here if you add a column in DB later
            }
        )

    entities = pd.DataFrame(enriched_rows)

    # Ensure one row per exact label text (no normalization)
    entities = entities.drop_duplicates(subset="label", keep="first")

    # Hash-based entity_id (stable across rebuilds) from the exact label string
    entities["entity_id"] = entities["label"].map(
        lambda x: "ent_" + md5(str(x).encode("utf-8")).hexdigest()
    )

    # Map dedication_label -> entities.entity_id
    label_to_id: Dict[str, str] = {
        row["label"]: row["entity_id"]
        for _, row in entities.iterrows()
    }
    streets["dedication_entity_id"] = streets["dedication_label"].map(
        lambda lab: label_to_id.get(lab) if isinstance(lab, str) else None
    )

    # Final streets dataframe: no redundant columns
    streets = streets[
        ["progressivo_nazionale", "ODONIMO", "istat_comune", "dedication_entity_id"]
    ]

    log.info("[3/3] Building SQLite database: %s", OUT_DB)
    build_sql_db(OUT_DB, prov=prov, mun=mun, streets=streets, entities=entities)

    log.info(
        "[Done] LLM_CSV=%s  DB=%s  Model=%s",
        OUT_LLM_CLASSES.name,
        OUT_DB,
        args.model,
    )


if __name__ == "__main__":
    main()