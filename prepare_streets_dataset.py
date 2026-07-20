#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re
import unicodedata

# Hardcoded paths
CSV_PATH = Path("raw/STRAD_ITA_20251010.csv")
LOOKUP_CSV_PATH = Path("raw/Elenco-comuni-italiani.csv")
OUT_DIR = Path("input")
OUT_FILE = OUT_DIR / CSV_PATH.name  # save with same name under input/

# Join keys and lookup column mapping
CSV_KEY = "CODICE_COMUNE"
LOOKUP_KEY = "Codice Catastale del comune"


def norm_header(s: str) -> str:
    """Normalize header names so we can match them robustly."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("\r\n", "\n").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    # IMPORTANT: do NOT strip parentheses content, because your CSV uses it
    # meaningfully (e.g. "Denominazione (Italiana e straniera)").
    # If you ever change this, you must also adjust LOOKUP_TARGETS.
    # s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    return s


# Map from *normalized* header -> final column name
# We let norm_header do the work so there are no typos.
LOOKUP_TARGETS = {
    norm_header("Denominazione in italiano"): (
        "COMUNE"      # or "Denominazione (Italiana e straniera)" if you prefer that one
    ),
    norm_header(
        "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)"
    ): "PROVINCIA",
    norm_header("Sigla automobilistica"): "SIGLA",
}


def norm_odonimo(s: str | None) -> str:
    """
    Normalize ODONIMO:
      - NFKC normalization
      - normalize fancy apostrophes to '
      - collapse whitespace
      - strip leading/trailing spaces
    (keeps original casing, no other changes)
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main() -> None:
    # --- Streets CSV (known UTF-8) ---
    csv_df = pd.read_csv(
        CSV_PATH,
        sep=";",
        encoding="utf-8",
        dtype=str,
        index_col=False,
    )
    csv_df.columns = [str(c).strip() for c in csv_df.columns]
    # Drop any Unnamed columns from CSV immediately and remember cleaned order
    csv_df = csv_df.loc[:, ~csv_df.columns.str.startswith("Unnamed")]
    base_cols = list(csv_df.columns)  # cleaned original CSV column order

    # Normalize ODONIMO if present
    if "ODONIMO" in csv_df.columns:
        csv_df["ODONIMO"] = csv_df["ODONIMO"].map(norm_odonimo)

    # --- Municipality lookup CSV ---
    # We want: join key + the 3 targets (COMUNE, PROVINCIA, SIGLA)
    needed_norm = {norm_header(LOOKUP_KEY), *LOOKUP_TARGETS.keys()}

    lookup_df = pd.read_csv(
        LOOKUP_CSV_PATH,
        sep=";",
        encoding="cp1252",
        dtype=str,
        keep_default_na=False,
    )

    # Keep only needed columns by normalized name
    cols_to_keep = [c for c in lookup_df.columns if norm_header(c) in needed_norm]
    lookup_df = lookup_df[cols_to_keep]

    # Standardize headers: drop unnamed/empty, then rename targets and the key
    lookup_df = lookup_df.loc[
        :, [c for c in lookup_df.columns if str(c).strip() and not str(c).startswith("Unnamed")]
    ]
    lookup_df = lookup_df.rename(
        columns=lambda c: LOOKUP_TARGETS.get(norm_header(c), c)
    )

    # Ensure the join key column is named exactly as LOOKUP_KEY
    key_candidates = [c for c in lookup_df.columns if norm_header(c) == norm_header(LOOKUP_KEY)]
    if not key_candidates:
        raise KeyError(
            f"Join key '{LOOKUP_KEY}' not found (after normalization) in {LOOKUP_CSV_PATH.name}."
        )
    if key_candidates[0] != LOOKUP_KEY:
        lookup_df = lookup_df.rename(columns={key_candidates[0]: LOOKUP_KEY})

    # Debug (optional)
    # print("After renaming, lookup_df.columns:", lookup_df.columns.tolist())

    # Clean keys
    csv_df[CSV_KEY] = csv_df[CSV_KEY].astype(str).str.strip().str.upper()
    lookup_df[LOOKUP_KEY] = lookup_df[LOOKUP_KEY].astype(str).str.strip().str.upper()

    # Build a safe list of lookup columns for merge (in case some are missing)
    merge_cols = [LOOKUP_KEY] + [
        c for c in ("COMUNE", "PROVINCIA", "SIGLA") if c in lookup_df.columns
    ]

    # Left join with only needed lookup cols
    merged = csv_df.merge(
        lookup_df[merge_cols],
        left_on=CSV_KEY,
        right_on=LOOKUP_KEY,
        how="left",
    )

    # Drop the lookup key and any stray unnamed/empty-named columns
    merged = merged.drop(columns=[LOOKUP_KEY], errors="ignore")
    merged = merged.loc[
        :, [c for c in merged.columns if str(c).strip() and not str(c).startswith("Unnamed")]
    ]

    # Reorder safely:
    # 1) keep original CSV columns that still exist (in their original order)
    # 2) insert new fields immediately to the RIGHT of the join key column
    base_cols_in_merged = [c for c in base_cols if c in merged.columns]

    # Only lookup-derived fields that weren't already in the CSV
    new_fields = [
        c
        for c in ("COMUNE", "PROVINCIA", "SIGLA")
        if c in merged.columns and c not in base_cols_in_merged
    ]

    if CSV_KEY in base_cols_in_merged:
        insert_at = base_cols_in_merged.index(CSV_KEY) + 1
        new_order = (
            base_cols_in_merged[:insert_at]
            + new_fields
            + base_cols_in_merged[insert_at:]
        )
    else:
        # Fallback: if the key isn't present, append the new fields at the end
        new_order = base_cols_in_merged + new_fields

    # Add any remaining columns not yet included (preserving current order)
    remainder = [c for c in merged.columns if c not in new_order]
    merged = merged[new_order + remainder]

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()