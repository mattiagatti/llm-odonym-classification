#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sqlite3
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# plotting (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FuncFormatter

# Optional map deps
try:
    import geopandas as gpd
    import folium
    from folium.features import GeoJsonPopup, GeoJsonTooltip
    import branca.colormap as bcm
    from branca.element import MacroElement, Template
except Exception:  # noqa: BLE001
    gpd = None
    folium = None
    GeoJsonTooltip = None
    GeoJsonPopup = None
    bcm = None
    MacroElement = None
    Template = None

# Optional metrics deps
try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:  # noqa: BLE001
    classification_report = None
    confusion_matrix = None


# =========================
# Config and constants
# =========================

VALID_GENDERS = {"F", "M", "U", "FM"}
PERSON_CATEGORY_IDS = {1, 2}
CATEGORY_OTHER = 0

METRIC_FEMALE_SHARE = "female_share"

DEFAULT_COLORMAP = "plasma"
DEFAULT_TILES = None  # folium default


@dataclass(frozen=True)
class Paths:
    # roots
    input_dir: Path = Path("input")
    out_dir: Path = Path("output")
    report_dir: Path = out_dir / "reports"
    maps_dir: Path = out_dir / "maps"

    # core inputs
    db_path: Path = out_dir / "llama-3.1-70b-instruct_streets.sqlite"
    classifications_csv: Path = out_dir / "llama-3.1-70b-instruct_classifications.csv"
    gold_csv_path: Path = input_dir / "manual_labels"
    municipal_lookup_csv: Path = Path("raw") / "Elenco-comuni-italiani.csv"

    # geometry inputs
    municipalities_shp: Path = input_dir / "italy_admin_boundaries/Com01012025_g/Com01012025_g_WGS84.shp"
    provinces_shp: Path = input_dir / "italy_admin_boundaries/ProvCM01012025_g/ProvCM01012025_g_WGS84.shp"
    regions_shp: Path = input_dir / "italy_admin_boundaries/Reg01012025_g/Reg01012025_g_WGS84.shp"

    # main exports
    out_streets_csv: Path = report_dir / "streets_with_classification.csv"

    # maps (HTML)
    muni_dir = maps_dir / "municipal_csv"
    map_html_3levels: Path = maps_dir / "gender_map.html"
    map_html_muni: Path = maps_dir / "municipal_gender_map.html"
    map_html_prov: Path = maps_dir / "province_gender_map.html"
    map_html_region: Path = maps_dir / "region_gender_map.html"

    # maps (PDF)
    map_pdf_muni: Path = maps_dir / "municipal_gender_map.pdf"
    map_pdf_prov: Path = maps_dir / "province_gender_map.pdf"
    map_pdf_region: Path = maps_dir / "region_gender_map.pdf"

    # plots
    prov_ecdf_pdf: Path = report_dir / "province_female_share_ecdf.pdf"
    gender_bar_pdf: Path = report_dir / "gender_dedications_bar.pdf"

    # evaluation outputs
    eval_confusion_pdf: Path = report_dir / "confusion_matrix.pdf"
    eval_diag_csv: Path = report_dir / "diagnostics.csv"
    eval_report_csv: Path = report_dir / "report.csv"
    eval_diag_female_mis: Path = report_dir / "diagnostics_female_misclassified.csv"
    eval_diag_male_mis: Path = report_dir / "diagnostics_male_misclassified.csv"
    eval_diag_undef_as_f: Path = report_dir / "diagnostics_undefined_misclassified_as_female.csv"
    eval_diag_undef_as_m: Path = report_dir / "diagnostics_undefined_misclassified_as_male.csv"


@dataclass(frozen=True)
class GeoSchema:
    muni_id_field: str = "PRO_COM_T"     # ISTAT text code in municipalities shapefile
    prov_sigla_field: str = "SIGLA"      # province sigla in provinces shapefile
    prov_name_field: str = "DEN_PROV"    # province name field (optional)
    region_code_field: str = "COD_REG"   # region code in regions shapefile
    region_name_field: str = "DEN_REG"   # region name in regions shapefile


@dataclass(frozen=True)
class Settings:
    map_start_zoom: int = 6
    join_on_model: str = "street_id"  # joins to GOLD PROGRESSIVO_NAZIONALE
    provinces_filter: tuple[str, ...] = ()  # names in DB "p.provincia", case-insensitive
    simplify_tol_muni: float = 0.0
    simplify_tol_prov: float = 0.0
    simplify_tol_region: float = 0.0

    # 3-layer map behavior
    enable_zoom_autoswitch: bool = False  # set True if you want auto layer switching by zoom


# =========================
# Small helpers
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_text(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip()


def norm_upper(x) -> str:
    return norm_text(x).upper()


def norm_gender(x) -> str | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().upper()
    return s if s in VALID_GENDERS else None


def safe_union_all(geoseries):
    try:
        return geoseries.union_all()
    except Exception:
        return geoseries.unary_union


def compute_center(gdf) -> list[float]:
    try:
        union_geom = safe_union_all(gdf.geometry)
        c = union_geom.centroid
        return [float(c.y), float(c.x)]
    except Exception:
        return [41.8719, 12.5674]


def simplify_to_wgs84(gdf, tolerance: float) -> "gpd.GeoDataFrame":
    if gpd is None or gdf is None or gdf.empty:
        return gdf
    if tolerance is None or tolerance <= 0:
        return gdf
    out = gdf
    try:
        out = out.to_crs(3857)
        out["geometry"] = out.geometry.simplify(tolerance, preserve_topology=True)
        out = out.to_crs(4326)
    except Exception:
        return gdf
    return out


def set_geom_precision(gdf, grid_size: float = 1e-5):
    if gpd is None or gdf is None or gdf.empty:
        return gdf
    try:
        from shapely.set_precision import set_precision
        gdf["geometry"] = gdf.geometry.apply(lambda geom: set_precision(geom, grid_size=grid_size))
    except Exception:
        pass
    return gdf


def compute_weighted_unweighted_averages(
    df: pd.DataFrame,
    level_name: str,
) -> None:
    """
    Prints weighted and unweighted averages of female_share
    for a given aggregation level (province or region).
    Standard deviation is reported only for the unweighted mean.
    """
    if df is None or df.empty:
        print(f"[INFO] No data for {level_name} averages.")
        return

    # Denominator F+M
    denom = (
        pd.to_numeric(df.get("female_count"), errors="coerce").fillna(0).astype(float)
        + pd.to_numeric(df.get("male_count"), errors="coerce").fillna(0).astype(float)
    )

    ratio = pd.to_numeric(df.get(METRIC_FEMALE_SHARE), errors="coerce").astype(float)

    valid = (denom > 0) & ratio.notna()
    if not valid.any():
        print(f"[INFO] No valid ratios for {level_name}.")
        return

    r = ratio[valid]

    # Unweighted statistics
    unweighted_mean = float(r.mean())
    unweighted_sd = float(r.std(ddof=1))  # sample SD

    # Weighted mean (national ratio)
    weighted = float((r * denom[valid]).sum() / denom[valid].sum())

    print(f"\nAverage {level_name} female ratio (F / (F + M)):")
    print(f"- unweighted: {unweighted_mean * 100:.2f}% (SD {unweighted_sd * 100:.2f}%)")
    print(f"- weighted:   {weighted * 100:.2f}%")


def female_share_from_counts(female: pd.Series, male: pd.Series) -> pd.Series:
    denom = (female.fillna(0).astype(float) + male.fillna(0).astype(float)).replace(0, np.nan)
    return female.astype(float).divide(denom)


def add_tooltip_sections(df_like: pd.DataFrame) -> pd.DataFrame:
    df_like["section_street_counts"] = ""
    df_like["section_gender_counts"] = ""
    df_like["female_ratio_pct"] = df_like[METRIC_FEMALE_SHARE].apply(
        lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "0.00% (No data)"
    )
    return df_like


def fill_count_columns(df_like: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df_like.columns:
            df_like[c] = 0
        df_like[c] = pd.to_numeric(df_like[c], errors="coerce").fillna(0).astype(int)
    return df_like


COUNT_COLS = [
    "person_single_count",
    "person_collective_count",
    "other_count",
    "female_count",
    "male_count",
    "mixed_gender_count",
    "unknown_gender_count",
    "total_person_count",
    "total_streets_count",
]


# =========================
# SQLite extraction
# =========================

def load_streets_with_classification(
    db_path: Path,
    provinces_filter: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Returns a DataFrame with at least:
      province, province_sigla, municipality, istat,
      street_id, street, entity, category_id, category, gender
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    sql = """
    SELECT
      p.provincia              AS province,
      m.comune                 AS municipality,
      m.istat_comune           AS istat,
      m.sigla_prov             AS province_sigla,
      s.progressivo_nazionale  AS street_id,
      s.odonimo                AS street,
      e.entity_id              AS entity_id,
      e.label                  AS entity,
      e.category_id            AS category_id,
      e.category               AS category,
      e.gender                 AS gender
    FROM streets s
    LEFT JOIN municipalities m
           ON s.istat_comune = m.istat_comune
    LEFT JOIN provinces p
           ON m.sigla_prov = p.sigla_prov
    LEFT JOIN entities e
           ON s.dedication_entity_id = e.entity_id
    """

    params: list[str] = []
    prov_list = [p.strip().upper() for p in provinces_filter if p.strip()]
    if prov_list:
        placeholders = ",".join("?" for _ in prov_list)
        sql += f"\nWHERE UPPER(p.provincia) IN ({placeholders})"
        params = prov_list

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn, params=params or None, dtype={"istat": str})
    finally:
        conn.close()

    df["province"] = df["province"].map(norm_text)
    df["municipality"] = df["municipality"].map(norm_text)
    df["istat"] = df["istat"].astype(str).map(norm_text)

    if "province_sigla" not in df.columns:
        df["province_sigla"] = ""
    df["province_sigla"] = df["province_sigla"].map(norm_upper)

    df["street_id"] = pd.to_numeric(df.get("street_id"), errors="coerce").astype("Int64")
    df["category_id"] = pd.to_numeric(df.get("category_id"), errors="coerce").fillna(0).astype("Int64")
    df["gender"] = df["gender"].map(norm_gender)

    return df


# =========================
# Aggregations
# =========================

def build_city_counts(df_streets: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["province", "province_sigla", "municipality", "istat"]

    person = df_streets[df_streets["category_id"].isin(PERSON_CATEGORY_IDS)].copy()

    if person.empty:
        grouped = pd.DataFrame(columns=index_cols)
    else:
        grouped = (
            person.groupby(index_cols, dropna=False)
            .agg(
                person_single_count=("category_id", lambda s: (s == 1).sum()),
                person_collective_count=("category_id", lambda s: (s == 2).sum()),
                female_count=("gender", lambda s: (s == "F").sum()),
                male_count=("gender", lambda s: (s == "M").sum()),
                mixed_gender_count=("gender", lambda s: (s == "FM").sum()),
                unknown_gender_count=("gender", lambda s: ((~s.isin(["F", "M", "FM"])) | s.isna()).sum()),
            )
            .reset_index()
        )

    other = (
        df_streets[df_streets["category_id"] == CATEGORY_OTHER]
        .groupby(index_cols, dropna=False)
        .size()
        .rename("other_count")
        .reset_index()
    )

    if grouped.empty and other.empty:
        return pd.DataFrame(columns=index_cols + COUNT_COLS + [METRIC_FEMALE_SHARE])

    if (not grouped.empty) and (not other.empty):
        combined = grouped.merge(other, on=index_cols, how="outer")
    else:
        combined = grouped.copy() if not grouped.empty else other.copy()

    for col in [
        "person_single_count",
        "person_collective_count",
        "female_count",
        "male_count",
        "mixed_gender_count",
        "unknown_gender_count",
        "other_count",
    ]:
        if col not in combined.columns:
            combined[col] = 0
        combined[col] = combined[col].fillna(0).astype(int)

    combined["total_person_count"] = (combined["person_single_count"] + combined["person_collective_count"]).astype(int)
    combined["total_streets_count"] = (combined["total_person_count"] + combined["other_count"]).astype(int)
    combined[METRIC_FEMALE_SHARE] = female_share_from_counts(combined["female_count"], combined["male_count"])

    return combined


def build_province_counts(df_city: pd.DataFrame) -> pd.DataFrame:
    if df_city is None or df_city.empty:
        return pd.DataFrame(columns=["province", "province_sigla"] + COUNT_COLS + [METRIC_FEMALE_SHARE])

    group_cols = ["province", "province_sigla"]
    agg_cols = [c for c in COUNT_COLS if c != "total_streets_count"] + ["total_streets_count"]

    df_prov = df_city.groupby(group_cols, dropna=False)[agg_cols].sum().reset_index()
    df_prov["province_sigla"] = df_prov["province_sigla"].map(norm_upper)
    df_prov[METRIC_FEMALE_SHARE] = female_share_from_counts(df_prov["female_count"], df_prov["male_count"])
    return df_prov


def build_region_counts_from_provinces(
    df_prov: pd.DataFrame,
    provinces_gdf: "gpd.GeoDataFrame",
    regions_gdf: "gpd.GeoDataFrame",
    geo: GeoSchema,
) -> pd.DataFrame:
    if df_prov is None or df_prov.empty or provinces_gdf is None or provinces_gdf.empty or regions_gdf is None or regions_gdf.empty:
        return pd.DataFrame(columns=["region", "region_code", *COUNT_COLS, METRIC_FEMALE_SHARE])

    prov = df_prov.copy()
    prov["province_sigla"] = prov["province_sigla"].map(norm_upper)

    pcols = [geo.prov_sigla_field, geo.region_code_field]
    prov2 = prov.merge(
        provinces_gdf[pcols].assign(**{
            geo.prov_sigla_field: provinces_gdf[geo.prov_sigla_field].astype(str).str.strip().str.upper()
        }),
        left_on="province_sigla",
        right_on=geo.prov_sigla_field,
        how="left",
    )

    rcols = [geo.region_code_field, geo.region_name_field]
    prov3 = prov2.merge(regions_gdf[rcols], on=geo.region_code_field, how="left")

    prov3["region"] = prov3[geo.region_name_field].fillna("Unknown Region")
    prov3["region_code"] = prov3[geo.region_code_field]

    group_cols = ["region_code", "region"]
    agg_cols = [c for c in COUNT_COLS if c in prov3.columns]

    df_region = prov3.groupby(group_cols)[agg_cols].sum().reset_index()
    df_region[METRIC_FEMALE_SHARE] = female_share_from_counts(df_region["female_count"], df_region["male_count"])
    return df_region


def print_top_regions(df_region: pd.DataFrame) -> None:
    if df_region is None or df_region.empty:
        print("[INFO] No region stats to print.")
        return

    tmp = df_region.copy()
    tmp["fm_denom"] = tmp["female_count"].fillna(0).astype(int) + tmp["male_count"].fillna(0).astype(int)
    tmp = tmp[tmp["fm_denom"] > 0].copy()
    tmp = tmp.sort_values(METRIC_FEMALE_SHARE, ascending=False)

    print("\nRegions sorted by female_share (F / (F+M)):")
    for _, row in tmp.iterrows():
        f = int(row.get("female_count", 0))
        m = int(row.get("male_count", 0))
        denom = f + m
        perc = (f / denom) * 100 if denom else 0.0
        print(f"- {row['region']} ({row['region_code']}): {perc:.2f}% female ({f}/{denom} F+M)")


# =========================
# Lookup (fallback names)
# =========================

def load_municipality_lookup(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[INFO] Municipality lookup file not found: {path}")
        return None

    try:
        df = pd.read_csv(path, sep=";", encoding="cp1252", dtype={"Codice Comune formato alfanumerico": str})
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read municipality lookup CSV: {exc}")
        return None

    istat_col = "Codice Comune formato alfanumerico"
    comune_col = "Denominazione (Italiana e straniera)"
    provincia_col = "Denominazione dell'Unità territoriale sovracomunale \n(valida a fini statistici)"
    sigla_col = "Sigla automobilistica"

    missing = [c for c in (istat_col, comune_col, provincia_col, sigla_col) if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns {missing} in {path}. Adjust load_municipality_lookup().")
        return None

    out = pd.DataFrame()
    out["istat_code"] = df[istat_col].astype(str).str.strip().str.zfill(6)
    out["municipality_lkp"] = df[comune_col].astype(str).str.strip()
    out["province_lkp"] = df[provincia_col].astype(str).str.strip()
    out["sigla_lkp"] = df[sigla_col].astype(str).str.strip().str.upper()

    out = out.dropna(subset=["istat_code"]).drop_duplicates(subset=["istat_code"])
    return out


# =========================
# Geo preparation
# =========================

def prepare_municipality_gdf(
    df_city: pd.DataFrame,
    paths: Paths,
    geo: GeoSchema,
    settings: Settings,
) -> tuple["gpd.GeoDataFrame", list[float], str] | tuple[None, None, None]:
    if folium is None or gpd is None:
        print("[INFO] GeoPandas or Folium not installed. Skipping municipality maps.")
        return None, None, None
    if not paths.municipalities_shp.exists():
        print(f"[INFO] Geo file not found: {paths.municipalities_shp}. Skipping municipality maps.")
        return None, None, None
    if df_city is None or df_city.empty or "istat" not in df_city.columns:
        print("[INFO] No city counts to map. Skipping municipality maps.")
        return None, None, None

    gdf = gpd.read_file(paths.municipalities_shp)
    gdf = gdf[[geo.muni_id_field, "geometry"]].copy()
    gdf["_istat"] = gdf[geo.muni_id_field].astype(str)

    dfk = df_city.copy()
    dfk["istat"] = dfk["istat"].astype(str)
    merged = gdf.merge(dfk, left_on="_istat", right_on="istat", how="left")

    lookup = load_municipality_lookup(paths.municipal_lookup_csv)
    if lookup is not None and not lookup.empty:
        merged = merged.merge(lookup, left_on="_istat", right_on="istat_code", how="left")

        for col in ["municipality", "province", "province_sigla"]:
            if col not in merged.columns:
                merged[col] = pd.NA

        merged["municipality"] = merged["municipality"].where(
            merged["municipality"].notna() & (merged["municipality"].astype(str).str.strip() != ""),
            merged["municipality_lkp"],
        )
        merged["province"] = merged["province"].where(
            merged["province"].notna() & (merged["province"].astype(str).str.strip() != ""),
            merged["province_lkp"],
        )
        merged["province_sigla"] = merged["province_sigla"].where(
            merged["province_sigla"].notna() & (merged["province_sigla"].astype(str).str.strip() != ""),
            merged["sigla_lkp"],
        )

        merged["municipality"] = merged["municipality"].astype(str).str.strip()
        merged["province"] = merged["province"].astype(str).str.strip()
        merged["province_sigla"] = merged["province_sigla"].astype(str).str.strip().str.upper()

        merged["province"] = np.where(
            merged["province_sigla"].str.strip() != "",
            merged["province"] + " (" + merged["province_sigla"] + ")",
            merged["province"],
        )

        merged = merged.drop(columns=[c for c in ["istat_code", "municipality_lkp", "province_lkp", "sigla_lkp"] if c in merged.columns])

    merged = fill_count_columns(merged, COUNT_COLS)
    if METRIC_FEMALE_SHARE not in merged.columns:
        merged[METRIC_FEMALE_SHARE] = np.nan
    merged[METRIC_FEMALE_SHARE] = pd.to_numeric(merged[METRIC_FEMALE_SHARE], errors="coerce").astype(float)

    merged.loc[merged["total_person_count"] == 0, METRIC_FEMALE_SHARE] = np.nan
    merged = add_tooltip_sections(merged)

    merged = merged.to_crs("EPSG:4326") if merged.crs is not None else gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged = simplify_to_wgs84(merged, settings.simplify_tol_muni)
    merged = set_geom_precision(merged)

    center = compute_center(merged)
    return merged, center, METRIC_FEMALE_SHARE


def prepare_province_gdf(
    df_prov: pd.DataFrame,
    paths: Paths,
    geo: GeoSchema,
    settings: Settings,
) -> tuple["gpd.GeoDataFrame", list[float], str] | tuple[None, None, None]:
    if gpd is None or folium is None:
        print("[INFO] GeoPandas or Folium not installed. Skipping province maps.")
        return None, None, None
    if df_prov is None or df_prov.empty:
        print("[INFO] No province counts to map. Skipping province maps.")
        return None, None, None
    if not paths.provinces_shp.exists():
        print(f"[INFO] Province shapefile not found: {paths.provinces_shp}. Skipping province maps.")
        return None, None, None

    gdf = gpd.read_file(paths.provinces_shp)
    if geo.prov_sigla_field not in gdf.columns:
        print(f"[WARN] Province sigla field '{geo.prov_sigla_field}' not found in {paths.provinces_shp}.")
        return None, None, None

    gdf[geo.prov_sigla_field] = gdf[geo.prov_sigla_field].astype(str).str.strip().str.upper()

    dfp = df_prov.copy()
    dfp["province_sigla"] = dfp["province_sigla"].map(norm_upper)
    merged = gdf.merge(dfp, left_on=geo.prov_sigla_field, right_on="province_sigla", how="left")

    if "province" not in merged.columns:
        merged["province"] = ""
    if geo.prov_name_field in merged.columns:
        merged["province"] = merged["province"].where(
            merged["province"].astype(str).str.strip() != "",
            merged[geo.prov_name_field].astype(str).str.strip(),
        )

    merged["province_display"] = merged["province"].astype(str).str.strip()
    merged["province_sigla"] = merged["province_sigla"].astype(str).str.strip().str.upper()
    merged["province_display"] = np.where(
        merged["province_sigla"] != "",
        merged["province_display"] + " (" + merged["province_sigla"] + ")",
        merged["province_display"],
    )

    merged = fill_count_columns(merged, COUNT_COLS)
    merged[METRIC_FEMALE_SHARE] = female_share_from_counts(merged["female_count"], merged["male_count"])
    merged.loc[merged["total_person_count"] == 0, METRIC_FEMALE_SHARE] = np.nan
    merged = add_tooltip_sections(merged)

    merged = merged.to_crs("EPSG:4326") if merged.crs is not None else gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged = simplify_to_wgs84(merged, settings.simplify_tol_prov)
    merged = set_geom_precision(merged)

    center = compute_center(merged)
    return merged, center, METRIC_FEMALE_SHARE


def prepare_region_gdf(
    df_region: pd.DataFrame,
    paths: Paths,
    geo: GeoSchema,
    settings: Settings,
) -> tuple["gpd.GeoDataFrame", list[float], str] | tuple[None, None, None]:
    if gpd is None or folium is None:
        print("[INFO] GeoPandas or Folium not installed. Skipping region maps.")
        return None, None, None
    if df_region is None or df_region.empty:
        print("[INFO] No region counts to map. Skipping region maps.")
        return None, None, None
    if not paths.regions_shp.exists():
        print(f"[INFO] Regions shapefile not found: {paths.regions_shp}. Skipping region maps.")
        return None, None, None

    regions_gdf = gpd.read_file(paths.regions_shp)
    if geo.region_code_field not in regions_gdf.columns:
        print(f"[WARN] Region code field '{geo.region_code_field}' not found in {paths.regions_shp}.")
        return None, None, None

    merged = regions_gdf.merge(df_region, left_on=geo.region_code_field, right_on="region_code", how="left")
    merged = fill_count_columns(merged, COUNT_COLS)

    if METRIC_FEMALE_SHARE not in merged.columns:
        merged[METRIC_FEMALE_SHARE] = np.nan
    merged[METRIC_FEMALE_SHARE] = pd.to_numeric(merged[METRIC_FEMALE_SHARE], errors="coerce").astype(float)

    merged.loc[merged["total_person_count"] == 0, METRIC_FEMALE_SHARE] = np.nan
    merged = add_tooltip_sections(merged)

    merged = merged.to_crs("EPSG:4326") if merged.crs is not None else gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged = simplify_to_wgs84(merged, settings.simplify_tol_region)
    merged = set_geom_precision(merged)

    center = [41.8719, 12.5674]
    return merged, center, METRIC_FEMALE_SHARE


# =========================
# Static PDFs
# =========================

def save_heatmap_pdf_from_gdf(gdf: "gpd.GeoDataFrame", metric: str, label: str, out_path: Path) -> None:
    if gpd is None or gdf is None or gdf.empty:
        print(f"[INFO] No geometries/metrics to plot for {label}. Skipping PDF.")
        return

    values = pd.to_numeric(gdf[metric], errors="coerce").astype(float)
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        print(f"[INFO] Only NaN metrics for {label}. Skipping PDF.")
        return

    # Round min/max to nearest integer percent, then convert back to 0..1
    vmin = round(float(valid.min()) * 100.0) / 100.0
    vmax = round(float(valid.max()) * 100.0) / 100.0

    # Safety: avoid zero-width color scale
    if vmin == vmax:
        vmin = max(0.0, vmin - 0.01)
        vmax = min(1.0, vmax + 0.01)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(DEFAULT_COLORMAP)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    gdf.plot(
        column=metric,
        ax=ax,
        cmap=cmap,
        linewidth=0.2,
        edgecolor="#666666",
        legend=False,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "#f0f0f0"},
    )

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
    cbar.ax.set_ylabel(label, rotation=90)

    # Only 2 ticks: min and max (already rounded)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    minx, miny, maxx, maxy = gdf.total_bounds
    dx = maxx - minx
    dy = maxy - miny
    pad_x = dx * 0.05 if dx > 0 else 0.05
    pad_y = dy * 0.05 if dy > 0 else 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[map] Saved static heatmap PDF: {out_path}")


# =========================
# Choropleth helper (single-level maps)
# =========================

def make_choropleth(
    m: "folium.Map",
    gdf: "gpd.GeoDataFrame",
    key_field: str,
    metric: str,
    legend_name: str,
    bins: int = 7,
) -> None:
    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=[key_field, metric],
        key_on=f"feature.properties.{key_field}",
        fill_color=DEFAULT_COLORMAP,
        fill_opacity=0.8,
        line_opacity=0.25,
        nan_fill_color="#f0f0f0",
        legend_name=legend_name,
        bins=bins,
        name="Choropleth",
    ).add_to(m)


# =========================
# Tooltips
# =========================

# =========================
# Tooltips (refactored)
# =========================

TOOLTIP_STYLE = (
    "background-color: white; color: #333; font-family: Arial; font-size: 12px; "
    "border: 1px solid #999; border-radius: 3px; padding: 8px;"
)

# Shared blocks
_TOOLTIP_STREET_FIELDS = [
    "section_street_counts",
    "person_single_count",
    "person_collective_count",
    "other_count",
    "total_streets_count",
]
_TOOLTIP_STREET_ALIASES = [
    "── Street counts ──",
    "Person single",
    "Person collective",
    "Other (non-person)",
    "Total streets",
]

_TOOLTIP_GENDER_FIELDS = [
    "section_gender_counts",
    "female_count",
    "male_count",
    "female_ratio_pct",
    "mixed_gender_count",
    "unknown_gender_count",
    "total_person_count",
]
_TOOLTIP_GENDER_ALIASES = [
    "── Gender of person streets ──",
    "Female",
    "Male",
    "Female ratio (%)",
    "Mixed (F&M)",
    "Unknown",
    "Total person-dedicated",
]


def _make_tooltip(
    title_field: str,
    title_alias: str,
    extra_fields: list[str] | None = None,
    extra_aliases: list[str] | None = None,
    street_aliases: list[str] | None = None,
    gender_aliases: list[str] | None = None,
) -> "GeoJsonTooltip":
    """
    Build a consistent tooltip:
      [title] + optional extra header fields + shared street block + shared gender block
    """
    extra_fields = extra_fields or []
    extra_aliases = extra_aliases or []
    street_aliases = street_aliases or _TOOLTIP_STREET_ALIASES
    gender_aliases = gender_aliases or _TOOLTIP_GENDER_ALIASES

    fields = [title_field, *extra_fields, *_TOOLTIP_STREET_FIELDS, *_TOOLTIP_GENDER_FIELDS]
    aliases = [title_alias, *extra_aliases, *street_aliases, *gender_aliases]

    return GeoJsonTooltip(
        fields=fields,
        aliases=aliases,
        localize=True,
        sticky=False,
        labels=True,
        style=TOOLTIP_STYLE,
    )


def municipal_tooltip() -> "GeoJsonTooltip":
    # Municipality has an extra "province" header line
    street_aliases = [
        "── Street counts ──",
        "Person single streets",
        "Person collective streets",
        "Other non person streets",
        "Total streets",
    ]
    gender_aliases = [
        "── Gender of person streets ──",
        "Female dedicated streets",
        "Male dedicated streets",
        "Female ratio (%)",
        "Mixed gender (F&M) streets",
        "Unknown gender streets",
        "Total person streets",
    ]
    return _make_tooltip(
        title_field="municipality",
        title_alias="Municipality",
        extra_fields=["province"],
        extra_aliases=["Province"],
        street_aliases=street_aliases,
        gender_aliases=gender_aliases,
    )


def province_tooltip() -> "GeoJsonTooltip":
    return _make_tooltip(
        title_field="province_display",
        title_alias="Province",
    )


def region_tooltip() -> "GeoJsonTooltip":
    return _make_tooltip(
        title_field="region",
        title_alias="Region",
    )


# =========================
# Single-level interactive maps (municipal, province, region)
# =========================

def make_municipal_map_html(
    df_city: pd.DataFrame,
    df_streets: pd.DataFrame,
    paths: Paths,
    geo: GeoSchema,
    settings: Settings,
) -> None:
    if folium is None or gpd is None:
        return

    merged, center, metric = prepare_municipality_gdf(df_city, paths, geo, settings)
    if merged is None:
        return

    ensure_dir(paths.muni_dir)

    df_st = df_streets.copy()
    df_st["istat_key"] = df_st["istat"].astype(str)

    csv_map: dict[str, str] = {}
    keep_cols = [c for c in ["province", "province_sigla", "municipality", "istat", "street", "entity", "category_id", "category", "gender"] if c in df_st.columns]

    for istat_key, sub in df_st.groupby("istat_key"):
        if not istat_key or istat_key.strip() == "":
            continue
        out = sub[keep_cols].sort_values(["municipality", "street"])
        fname = f"streets_{istat_key}.csv"
        out.to_csv(paths.muni_dir / fname, index=False)
        csv_map[istat_key] = fname

    def make_link(istat_code: str) -> str:
        fname = csv_map.get(istat_code, "")
        if not fname:
            return "No street data"
        rel = f"municipal_csv/{fname}"
        return f'<a href="{rel}" download>Download streets CSV</a>'

    merged["csv_link"] = merged["_istat"].map(make_link)

    m = folium.Map(location=center, zoom_start=settings.map_start_zoom, tiles=DEFAULT_TILES)

    make_choropleth(
        m=m,
        gdf=merged,
        key_field="_istat",
        metric=metric,
        legend_name="Female share (F / (F + M))",
    )

    popup = GeoJsonPopup(
        fields=["_istat", "municipality", "province", "csv_link"],
        aliases=["ISTAT", "Municipality", "Province", ""],
        labels=True,
        localize=True,
        style=(
            "background-color: white; color: #333; font-family: Arial; font-size: 12px; "
            "border: 1px solid #999; border-radius: 3px; padding: 8px;"
        ),
    )

    folium.GeoJson(
        merged.to_json(),
        name="Municipalities",
        tooltip=municipal_tooltip(),
        popup=popup,
        style_function=lambda _: {"weight": 0.2, "color": "#666", "fillOpacity": 0.0},
        highlight_function=lambda _: {"weight": 1.0, "color": "#000"},
    ).add_to(m)

    ensure_dir(paths.map_html_muni.parent)
    m.save(str(paths.map_html_muni))
    print(f"[map] Saved interactive municipal map: {paths.map_html_muni}")
    print(f"[map] Per-municipality CSVs in: {paths.muni_dir}")


def make_province_map_html(
    df_prov: pd.DataFrame,
    paths: Paths,
    geo: GeoSchema,
    settings: Settings,
) -> None:
    if folium is None or gpd is None:
        return

    prov_gdf, center, metric = prepare_province_gdf(df_prov, paths, geo, settings)
    if prov_gdf is None or prov_gdf.empty:
        return

    m = folium.Map(location=center, zoom_start=settings.map_start_zoom, tiles=DEFAULT_TILES)

    make_choropleth(
        m=m,
        gdf=prov_gdf,
        key_field="province_sigla",
        metric=metric,
        legend_name="Female share (F / (F + M))",
    )

    folium.GeoJson(
        prov_gdf.to_json(),
        name="Provinces",
        tooltip=province_tooltip(),
        style_function=lambda _: {"weight": 0.5, "color": "#666", "fillOpacity": 0.0},
        highlight_function=lambda _: {"weight": 1.5, "color": "#000"},
    ).add_to(m)

    ensure_dir(paths.map_html_prov.parent)
    m.save(str(paths.map_html_prov))
    print(f"[map] Saved interactive province map: {paths.map_html_prov}")


def make_region_map_html(
    df_region: pd.DataFrame,
    paths: Paths,
    geo: GeoSchema,
    settings: Settings,
) -> None:
    if folium is None or gpd is None:
        return

    region_gdf, center, metric = prepare_region_gdf(df_region, paths, geo, settings)
    if region_gdf is None or region_gdf.empty:
        return

    m = folium.Map(location=center, zoom_start=6, tiles=DEFAULT_TILES)

    make_choropleth(
        m=m,
        gdf=region_gdf,
        key_field="region_code",
        metric=metric,
        legend_name="Female share (F / (F + M))",
    )

    folium.GeoJson(
        region_gdf.to_json(),
        name="Regions",
        tooltip=region_tooltip(),
        style_function=lambda _: {"weight": 0.8, "color": "#444", "fillOpacity": 0.0},
        highlight_function=lambda _: {"weight": 2.0, "color": "#000"},
    ).add_to(m)

    ensure_dir(paths.map_html_region.parent)
    m.save(str(paths.map_html_region))
    print(f"[map] Saved interactive region map: {paths.map_html_region}")

# =========================
# 3-level map (UPDATED: per-level min/max + dynamic legend)
# =========================


def _finite_series(gdf: "gpd.GeoDataFrame", metric: str) -> pd.Series:
    if gdf is None or gdf.empty or metric not in gdf.columns:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(gdf[metric], errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _make_level_colormap(values: pd.Series, caption: str):
    if bcm is None:
        return None, None, None
    if values is None or values.empty:
        return None, None, None
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        vmin = max(0.0, vmin - 1e-6)
        vmax = vmin + 1e-6
    cm = bcm.linear.YlOrRd_09.scale(vmin, vmax)
    cm.caption = caption
    return cm, vmin, vmax


def add_choropleth_base_layer(
    m: "folium.Map",
    gdf: "gpd.GeoDataFrame",
    layer_name: str,
    metric: str,
    tooltip: "GeoJsonTooltip" | None,
    show: bool,
    line_weight: float,
    line_color: str,
    fill_opacity: float,
    colormap,
) -> None:
    def style_fn(feat):
        props = feat.get("properties", {})
        v = props.get(metric, None)
        try:
            v = float(v) if v is not None else None
        except Exception:
            v = None

        if v is None or (isinstance(v, float) and np.isnan(v)):
            return {"weight": line_weight, "color": line_color, "fillOpacity": 0.0}

        return {
            "weight": line_weight,
            "color": line_color,
            "fillColor": colormap(v) if colormap is not None else "#cccccc",
            "fillOpacity": fill_opacity,
        }

    layer = folium.GeoJson(
        gdf.to_json(),
        name=layer_name,
        tooltip=tooltip,
        style_function=style_fn,
        highlight_function=lambda _: {"weight": max(1.5, line_weight * 2), "color": "#000"},
        overlay=False,   # makes it a base layer
        control=True,
        show=show,
    )
    layer.add_to(m)


def add_exclusive_overlays_js(m: "folium.Map", overlay_names: list[str]) -> None:
    """
    When one overlay is activated, all other overlays are turned off.
    """
    if Template is None or MacroElement is None:
        return

    names_js = ", ".join([f'"{n}"' for n in overlay_names])

    tpl = Template(f"""
    {{% macro script(this, kwargs) %}}
    var map = {{{{this._parent.get_name()}}}};

    function findLayerByName(name) {{
      for (var key in map._layers) {{
        var layer = map._layers[key];
        if (layer && layer.options && layer.options.name === name) return layer;
      }}
      return null;
    }}

    var overlayNames = [{names_js}];

    function turnOffOthers(activeName) {{
      overlayNames.forEach(function(n) {{
        if (n === activeName) return;
        var lyr = findLayerByName(n);
        if (lyr && map.hasLayer(lyr)) map.removeLayer(lyr);
      }});
    }}

    map.on("overlayadd", function(e) {{
      var name = (e.layer && e.layer.options) ? e.layer.options.name : null;
      if (!name) return;
      turnOffOthers(name);
    }});
    {{% endmacro %}}
    """)

    macro = MacroElement()
    macro._template = tpl
    m.get_root().add_child(macro)


def add_dynamic_legend_js(
    m: "folium.Map",
    legend_by_layer: dict[str, str],
) -> None:
    if Template is None or MacroElement is None:
        return

    pairs = ",\n".join([f'"{k}": "{v}"' for k, v in legend_by_layer.items()])

    tpl = Template(f"""
    {{% macro script(this, kwargs) %}}
    var map = {{{{this._parent.get_name()}}}};

    var legendByLayer = {{
      {pairs}
    }};

    function isTileLayer(layer) {{
      return layer && (layer._url !== undefined);
    }}

    function getActiveBaseLayerName() {{
      for (var key in map._layers) {{
        var layer = map._layers[key];
        if (!layer || !layer.options || !layer.options.name) continue;

        if (layer.options.overlay === false && map.hasLayer(layer) && !isTileLayer(layer)) {{
          return layer.options.name;
        }}
      }}
      return null;
    }}

    function setLegendVisible(layerName) {{
      for (var lname in legendByLayer) {{
        var elId = legendByLayer[lname];
        var el = document.getElementById(elId);
        if (!el) continue;
        el.style.display = (lname === layerName) ? "block" : "none";
      }}
    }}

    function updateLegend() {{
      var active = getActiveBaseLayerName();
      if (!active) {{
        for (var k in legendByLayer) {{ active = k; break; }}
      }}
      if (active) setLegendVisible(active);
    }}

    map.on("baselayerchange", function(e) {{
      var name = (e.layer && e.layer.options) ? e.layer.options.name : null;
      if (name) setLegendVisible(name);
    }});

    map.on("zoomend", updateLegend);
    setTimeout(updateLegend, 80);
    {{% endmacro %}}
    """)

    macro = MacroElement()
    macro._template = tpl
    m.get_root().add_child(macro)


def add_zoom_switch_js(m: "folium.Map") -> None:
    if Template is None or MacroElement is None:
        return

    tpl = Template("""
    {% macro script(this, kwargs) %}
    var map = {{this._parent.get_name()}};

    function findLayerByName(name) {
      for (var key in map._layers) {
        var layer = map._layers[key];
        if (layer && layer.options && layer.options.name === name) {
          return layer;
        }
      }
      return null;
    }

    var L_regions = findLayerByName("Regions");
    var L_prov = findLayerByName("Provinces");
    var L_muni = findLayerByName("Municipalities");

    function setOnly(target) {
      var layers = [
        {name:"Regions", layer:L_regions},
        {name:"Provinces", layer:L_prov},
        {name:"Municipalities", layer:L_muni}
      ];
      layers.forEach(function(x){
        if (!x.layer) return;
        if (x.name === target) {
          if (!map.hasLayer(x.layer)) map.addLayer(x.layer);
        } else {
          if (map.hasLayer(x.layer)) map.removeLayer(x.layer);
        }
      });
    }

    function updateByZoom() {
      var z = map.getZoom();
      if (z <= 6) setOnly("Regions");
      else if (z <= 9) setOnly("Provinces");
      else setOnly("Municipalities");
    }

    map.on("zoomend", updateByZoom);
    updateByZoom();
    {% endmacro %}
    """)

    macro = MacroElement()
    macro._template = tpl
    m.get_root().add_child(macro)


def add_fixed_legends(m: "folium.Map", legends_html: dict[str, str]) -> None:
    if Template is None or MacroElement is None:
        return

    # legends_html: {"Regions": "<html...>", ...}
    blocks = []
    for name, html in legends_html.items():
        div_id = f"legend_{name.lower()}"
        display = "block" if name == "Regions" else "none"
        blocks.append(f"""
        <div id="{div_id}" style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 9999;
            display: {display};
        ">
        {html}
        </div>
        """)


    tpl = Template(f"""
    {{% macro html(this, kwargs) %}}
    {''.join(blocks)}
    {{% endmacro %}}
    """)
    macro = MacroElement()
    macro._template = tpl
    m.get_root().add_child(macro)


def make_three_level_map_html(
    df_city: pd.DataFrame,
    df_prov: pd.DataFrame,
    df_region: pd.DataFrame,
    paths: Paths,
    geo: GeoSchema,
    settings: Settings,
    out_path: Path,
) -> None:
    if folium is None or gpd is None or bcm is None:
        print("[INFO] Folium/GeoPandas/Branca not installed. Skipping 3-level map.")
        return

    muni_gdf, muni_center, metric_m = prepare_municipality_gdf(df_city, paths, geo, settings)
    prov_gdf, prov_center, metric_p = prepare_province_gdf(df_prov, paths, geo, settings)
    reg_gdf, reg_center, metric_r = prepare_region_gdf(df_region, paths, geo, settings)

    center = reg_center or prov_center or muni_center or [41.8719, 12.5674]
    m = folium.Map(location=center, zoom_start=settings.map_start_zoom, tiles=DEFAULT_TILES)

    # Per-level colormaps
    cm_reg, _, _ = _make_level_colormap(
        _finite_series(reg_gdf, metric_r),
        "Female share (Regions): F / (F + M)"
    )
    cm_prov, _, _ = _make_level_colormap(
        _finite_series(prov_gdf, metric_p),
        "Female share (Provinces): F / (F + M)"
    )
    cm_muni, _, _ = _make_level_colormap(
        _finite_series(muni_gdf, metric_m),
        "Female share (Municipalities): F / (F + M)"
    )

    # Add layers
    if reg_gdf is not None and not reg_gdf.empty and cm_reg is not None:
        add_choropleth_base_layer(
            m=m,
            gdf=reg_gdf,
            layer_name="Regions",
            metric=metric_r,
            tooltip=region_tooltip(),
            show=True,
            line_weight=0.9,
            line_color="#444",
            fill_opacity=0.65,
            colormap=cm_reg,
        )

    if prov_gdf is not None and not prov_gdf.empty and cm_prov is not None:
        add_choropleth_base_layer(
            m=m,
            gdf=prov_gdf,
            layer_name="Provinces",
            metric=metric_p,
            tooltip=province_tooltip(),
            show=False,
            line_weight=0.7,
            line_color="#555",
            fill_opacity=0.70,
            colormap=cm_prov,
        )

    if muni_gdf is not None and not muni_gdf.empty and cm_muni is not None:
        add_choropleth_base_layer(
            m=m,
            gdf=muni_gdf,
            layer_name="Municipalities",
            metric=metric_m,
            tooltip=municipal_tooltip(),
            show=False,
            line_weight=0.25,
            line_color="#666",
            fill_opacity=0.75,
            colormap=cm_muni,
        )

    legends_html = {}
    if cm_reg is not None:
        legends_html["Regions"] = cm_reg._repr_html_()
    if cm_prov is not None:
        legends_html["Provinces"] = cm_prov._repr_html_()
    if cm_muni is not None:
        legends_html["Municipalities"] = cm_muni._repr_html_()

    add_fixed_legends(m, legends_html)

    legend_ids = {
        "Regions": "legend_regions",
        "Provinces": "legend_provinces",
        "Municipalities": "legend_municipalities",
    }

    folium.LayerControl(collapsed=False).add_to(m)

    # Auto-switch layers by zoom (optional)
    if settings.enable_zoom_autoswitch:
        add_zoom_switch_js(m)

    # Dynamic legend switching
    if legend_ids:
        add_dynamic_legend_js(m, legend_ids)

    ensure_dir(out_path.parent)
    m.save(str(out_path))
    print(f"[map] Saved 3-level interactive map: {out_path}")


# =========================
# Plots (bar + confusion)
# =========================

def plot_global_gender_bar(df_streets: pd.DataFrame, out_path: Path) -> None:
    person = df_streets[df_streets["category_id"].isin(PERSON_CATEGORY_IDS)].copy()
    if person.empty:
        print("[plot] No person-dedicated streets found, skipping gender bar chart.")
        return

    cat_single = person["category_id"] == 1
    cat_collective = person["category_id"] == 2
    g = person["gender"]

    female_mask = g == "F"
    male_mask = g == "M"
    mixed_mask = g == "FM"
    undef_mask = (~g.isin(["F", "M", "FM"])) | g.isna()

    female_single = int((cat_single & female_mask).sum())
    female_collective = int((cat_collective & female_mask).sum())

    male_single = int((cat_single & male_mask).sum())
    male_collective = int((cat_collective & male_mask).sum())

    mixed_single = 0
    mixed_collective = int((cat_collective & mixed_mask).sum())

    undef_single = int((cat_single & undef_mask).sum())
    undef_collective = int((cat_collective & undef_mask).sum())

    labels = ["Female", "Male", "Mixed", "Undefined"]
    x = np.arange(len(labels))

    single_vals = [female_single, male_single, mixed_single, undef_single]
    collective_vals = [female_collective, male_collective, mixed_collective, undef_collective]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, single_vals, label="Single person")
    ax.bar(x, collective_vals, bottom=single_vals, label="Collective person")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of dedications")
    ax.legend()

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[plot] Saved gender distribution bar chart: {out_path}")


def plot_province_female_share_ecdf(df_prov: pd.DataFrame, out_path: Path) -> None:
    if df_prov is None or df_prov.empty or METRIC_FEMALE_SHARE not in df_prov.columns:
        print("[plot] No province data/metric found, skipping province ECDF.")
        return

    s = pd.to_numeric(df_prov[METRIC_FEMALE_SHARE], errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        print("[plot] Province female_share series is empty (all NaN), skipping ECDF.")
        return

    x = np.sort(s.values)
    y = np.arange(1, len(x) + 1) / float(len(x))

    # "ECDF median": smallest x such that ECDF(x) >= 0.5
    idx = int(np.searchsorted(y, 0.5, side="left"))
    idx = min(max(idx, 0), len(x) - 1)
    med = float(x[idx])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(x, y, where="post")

    ax.set_xlabel("Female share (F / (F + M))")
    ax.set_ylabel("Cumulative fraction of provinces")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(med, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.plot([med], [0.5], marker="o", markersize=4, color="red")

    ax.annotate(
        f"median = {med*100:.1f}%",
        xy=(med, 0.5),
        xytext=(8, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        # fontsize=9,
        color="red",
    )

    ax.set_ylim(0.0, 1.0)

    ax.set_xlim(x.min(), x.max())

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[plot] Saved province ECDF: {out_path}")


def plot_confusion_matrix_pdf(
    cm_raw: list[list[int]],
    cm_norm: list[list[float]],
    labels: list[str],
    out_path: Path,
) -> None:
    if not cm_raw or not cm_norm or not labels:
        print("[INFO] Empty confusion matrix, skipping confusion plot.")
        return

    cm_raw_arr = np.array(cm_raw, dtype=float)
    cm_norm_arr = np.array(cm_norm, dtype=float)

    if cm_raw_arr.shape != cm_norm_arr.shape:
        print("[WARN] Raw and normalized confusion shapes differ, skipping confusion plot.")
        return

    n = cm_raw_arr.shape[0]
    if n != cm_raw_arr.shape[1] or n != len(labels):
        print("[WARN] Confusion matrix shape/labels mismatch, skipping confusion plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm_arr, interpolation="nearest", cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")

    thresh = cm_norm_arr.max() / 2.0 if cm_norm_arr.max() > 0 else 0.0
    for i in range(n):
        for j in range(n):
            count = int(cm_raw_arr[i, j])
            frac = float(cm_norm_arr[i, j])
            ax.text(
                j, i, f"{count}\n({frac:.2f})",
                ha="center", va="center",
                color="white" if frac > thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[eval] Saved confusion matrix plot: {out_path}")


# =========================
# Evaluation
# =========================

def read_gold(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Reference CSV not found: {path}")

    # try comma first, then semicolon
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        df = pd.read_csv(path, dtype=str, sep=";")

    # normalize headers
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "GENERE" not in df.columns:
        raise KeyError(f"Reference CSV {path} is missing column GENERE")

    if "PROGRESSIVO_NAZIONALE" not in df.columns:
        raise KeyError(
            f"Reference CSV {path} is missing PROGRESSIVO_NAZIONALE. "
            "With direct matching, this column is required."
        )

    gold_label = pd.to_numeric(df["PROGRESSIVO_NAZIONALE"], errors="coerce").astype("Int64")

    def map_gold(x: str) -> str:
        s = str(x).strip().upper()
        if s == "F":
            return "F"
        if s == "M":
            return "M"
        return "U"

    gold_gender = df["GENERE"].map(map_gold)

    out = pd.DataFrame({"gold_label": gold_label, "gold_gender": gold_gender})
    out = out[out["gold_label"].notna()].drop_duplicates(subset=["gold_label"], keep="first")
    return out


def read_gold_dir(gold_dir: Path) -> pd.DataFrame:
    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold directory not found: {gold_dir}")

    files = sorted(gold_dir.glob("STRAD_*_GENDER.csv"))
    files += sorted(gold_dir.glob("STRAD_*_GENDER.CSV"))
    if not files:
        raise FileNotFoundError(f"No STRAD_*_GENDER.csv files found in: {gold_dir}")

    parts = [read_gold(p) for p in files]
    gold_df = pd.concat(parts, ignore_index=True)
    gold_df = gold_df.drop_duplicates(subset=["gold_label"], keep="first")
    return gold_df


def read_gold_dir_per_file(gold_dir: Path) -> list[tuple[Path, pd.DataFrame]]:
    """
    Returns a list of (filepath, gold_df_for_that_file).
    Each gold_df has columns: gold_label, gold_gender (same schema as read_gold()).
    """
    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold directory not found: {gold_dir}")

    files = sorted(gold_dir.glob("STRAD_*_GENDER.csv"))
    files += sorted(gold_dir.glob("STRAD_*_GENDER.CSV"))
    if not files:
        raise FileNotFoundError(f"No STRAD_*_GENDER.csv files found in: {gold_dir}")

    out: list[tuple[Path, pd.DataFrame]] = []
    for p in files:
        out.append((p, read_gold(p)))
    return out


def print_report_metrics(report: dict, labels: list[str], title: str) -> None:
    """
    Prints precision/recall/f1 for each class and accuracy, plus macro/weighted averages if present.
    Expects sklearn-like classification_report(output_dict=True) structure.
    """
    print(f"\n{title}")

    # Per-class
    for lbl in labels:
        rr = report.get(lbl, None)
        if not isinstance(rr, dict):
            continue
        p = float(rr.get("precision", 0.0))
        r = float(rr.get("recall", 0.0))
        f1 = float(rr.get("f1-score", 0.0))
        sup = int(rr.get("support", 0))
        print(f"  {lbl}: precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}, support={sup}")

    # Averages (if available)
    for avg_key in ["macro avg", "weighted avg"]:
        rr = report.get(avg_key, None)
        if isinstance(rr, dict):
            p = float(rr.get("precision", 0.0))
            r = float(rr.get("recall", 0.0))
            f1 = float(rr.get("f1-score", 0.0))
            sup = int(rr.get("support", 0))
            print(f"  {avg_key}: precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}, support={sup}")

    # Accuracy
    acc = report.get("accuracy", None)
    if acc is not None and not isinstance(acc, dict):
        print(f"  accuracy={float(acc):.4f}")


def evaluate_gender_3class(
    df_model: pd.DataFrame,
    join_on: str,
    gold_df: pd.DataFrame,
    unique_by: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    labels3 = ["F", "M", "U"]

    cols = [join_on, "gender"]
    if unique_by is not None and unique_by in df_model.columns:
        cols.append(unique_by)

    model = df_model[cols].copy()
    model = model.dropna(subset=[join_on]).drop_duplicates(subset=[join_on])

    gold = gold_df.rename(columns={"gold_label": join_on}).copy()

    coverage = {
        "gold_total_rows": int(len(gold_df)),
        "gold_unique_labels": int(gold_df["gold_label"].nunique()),
        "model_total_rows": int(len(model)),
        "model_unique_ids": int(model[join_on].nunique()),
        "matched_rows": 0,
    }

    m = model.merge(gold, on=join_on, how="inner", validate="one_to_one")
    coverage["matched_rows"] = int(len(m))

    if m.empty:
        summary_empty = {
            "report": {},
            "labels": labels3,
            "confusion_raw": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "confusion": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "coverage": coverage,
        }
        diag_empty = pd.DataFrame(columns=[join_on, "gender", "gold_gender", "correct", "error"])
        return summary_empty, diag_empty

    if unique_by is not None and unique_by in m.columns:
        m["__name_key"] = m[unique_by].astype(str).str.strip().str.upper()
        m = m[m["__name_key"] != ""]

        def agg_gender(s: pd.Series) -> str:
            s2 = s.dropna().astype(str).str.upper()
            if (s2 == "F").any():
                return "F"
            if (s2 == "M").any():
                return "M"
            return "U"

        m = m.groupby("__name_key", as_index=False).agg(
            gold_gender=("gold_gender", agg_gender),
            gender=("gender", agg_gender),
        )

    def norm3(x: str) -> str:
        s = str(x).strip().upper()
        if s == "F":
            return "F"
        if s == "M":
            return "M"
        return "U"

    y_true = m["gold_gender"].map(norm3)
    y_pred = m["gender"].map(norm3)

    if confusion_matrix is None:
        cm_df = pd.crosstab(y_true, y_pred, dropna=False).reindex(index=labels3, columns=labels3, fill_value=0)
        cm_raw = cm_df.values.astype(int)
    else:
        cm_raw = confusion_matrix(y_true.tolist(), y_pred.tolist(), labels=labels3)

    cm_raw_df = pd.DataFrame(cm_raw, index=labels3, columns=labels3)
    row_sums = cm_raw_df.sum(axis=1).replace(0, np.nan)
    cm_norm_df = cm_raw_df.div(row_sums, axis=0).fillna(0.0)

    if classification_report is not None:
        report = classification_report(
            y_true.tolist(),
            y_pred.tolist(),
            labels=labels3,
            target_names=labels3,
            output_dict=True,
            zero_division=0,
        )
    else:
        report = {}
        total = float(cm_raw.sum())
        acc = float(np.trace(cm_raw)) / total if total > 0 else 0.0
        report["accuracy"] = acc

    summary = {
        "report": report,
        "labels": labels3,
        "confusion_raw": cm_raw.astype(int).tolist(),
        "confusion": cm_norm_df.values.tolist(),
        "coverage": coverage,
    }

    diag = pd.DataFrame()
    if "__name_key" in m.columns:
        diag["name"] = m["__name_key"]
    else:
        diag[join_on] = m[join_on]

    diag["gender"] = y_pred.values
    diag["gold_gender"] = y_true.values
    diag["correct"] = diag["gender"] == diag["gold_gender"]
    diag["error"] = np.where(diag["correct"], "", diag["gold_gender"] + "->" + diag["gender"])

    return summary, diag


# =========================
# Main program
# =========================

def print_classification_source_stats(paths: Paths) -> None:
    if not paths.classifications_csv.is_file():
        print(f"[stats] Classifications CSV not found: {paths.classifications_csv}")
        return

    try:
        df_cls = pd.read_csv(paths.classifications_csv)
    except Exception as exc:  # noqa: BLE001
        print(f"[stats] Failed to read {paths.classifications_csv}: {exc}")
        return

    if "source" not in df_cls.columns:
        print(f"[stats] {paths.classifications_csv} has no 'source' column.")
        return

    total = len(df_cls)
    src = df_cls["source"].astype(str).str.strip().str.lower().value_counts()
    llm = int(src.get("llm", 0))
    heur = int(src.get("heuristic", 0))
    other = int(total - llm - heur)

    print("[stats] Gender labels by source (from classifications CSV)")
    print(f"[stats]   total labeled rows: {total}")
    print(f"[stats]   classified by LLM: {llm}")
    print(f"[stats]   classified by heuristics: {heur}")
    if other > 0:
        print(f"[stats]   other/unknown source: {other}")


def main(db_path: Path) -> int:
    paths = Paths(db_path=db_path)
    geo = GeoSchema()
    settings = Settings()

    ensure_dir(paths.out_dir)
    print_classification_source_stats(paths)

    if not paths.db_path.exists():
        print(f"[ERR] SQLite DB not found: {paths.db_path}")
        return 1

    print(f"[1/3] Loading streets from DB: {paths.db_path}")
    df_streets = load_streets_with_classification(paths.db_path, provinces_filter=settings.provinces_filter)

    print(df_streets["category_id"].value_counts(dropna=False))
    print(df_streets["gender"].value_counts(dropna=False).head())
    print(f"      Loaded {len(df_streets)} rows")

    ensure_dir(paths.out_streets_csv.parent)
    df_streets.to_csv(paths.out_streets_csv, index=False)
    print(f"[2/3] Wrote global streets CSV: {paths.out_streets_csv}")

    plot_global_gender_bar(df_streets, paths.gender_bar_pdf)

    df_city = build_city_counts(df_streets)

    compute_weighted_unweighted_averages(df_city, "municipality")
    
    df_prov = build_province_counts(df_city)

    compute_weighted_unweighted_averages(df_prov, "province")

    plot_province_female_share_ecdf(df_prov, paths.prov_ecdf_pdf)

    print(df_city[["total_person_count", METRIC_FEMALE_SHARE]].head())
    print(int((df_city["total_person_count"] > 0).sum()), "municipalities with person streets")

    top_n = 10
    df_prov_nonempty = df_prov[(df_prov["total_person_count"] > 0) & (df_prov[METRIC_FEMALE_SHARE].notna())].copy()
    df_top = (
        df_prov_nonempty.sort_values(METRIC_FEMALE_SHARE, ascending=False)
        .head(top_n)
        .loc[:, ["province", "province_sigla", "total_person_count", "female_count", METRIC_FEMALE_SHARE]]
    )
    print(f"\nTop {top_n} provinces by female_share (person-dedicated streets only):")
    for _, row in df_top.iterrows():
        perc = float(row[METRIC_FEMALE_SHARE]) * 100
        print(f"- {row['province']} ({row['province_sigla']}): {perc:.2f}% female ({int(row['female_count'])}/{int(row['total_person_count'])})")

    print("[2/3] Generating maps...")

    if gpd is not None and folium is not None:
        # municipal PDF + HTML
        muni_gdf, _, metric_m = prepare_municipality_gdf(df_city, paths, geo, settings)
        if muni_gdf is not None:
            save_heatmap_pdf_from_gdf(muni_gdf, metric=metric_m, label="Female share (F / (F + M))", out_path=paths.map_pdf_muni)
        make_municipal_map_html(df_city, df_streets, paths, geo, settings)

        # province PDF + HTML
        prov_gdf, _, metric_p = prepare_province_gdf(df_prov, paths, geo, settings)
        if prov_gdf is not None:
            save_heatmap_pdf_from_gdf(prov_gdf, metric=metric_p, label="Female share (F / (F + M))", out_path=paths.map_pdf_prov)
        make_province_map_html(df_prov, paths, geo, settings)

        # regions, plus region PDF + HTML, plus 3-level map
        try:
            provinces_gdf = gpd.read_file(paths.provinces_shp)
            regions_gdf = gpd.read_file(paths.regions_shp)
            df_region = build_region_counts_from_provinces(df_prov, provinces_gdf, regions_gdf, geo)

            compute_weighted_unweighted_averages(df_region, "region")

            region_gdf, _, metric_r = prepare_region_gdf(df_region, paths, geo, settings)
            if region_gdf is not None:
                save_heatmap_pdf_from_gdf(region_gdf, metric=metric_r, label="Female share (F / (F + M))", out_path=paths.map_pdf_region)

            print_top_regions(df_region)
            make_region_map_html(df_region, paths, geo, settings)

            make_three_level_map_html(
                df_city=df_city,
                df_prov=df_prov,
                df_region=df_region,
                paths=paths,
                geo=geo,
                settings=settings,
                out_path=paths.map_html_3levels,
            )

        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Region/3-level maps skipped due to error: {exc}")
    else:
        print("[INFO] Skipping maps: geopandas/folium not available.")

    print("[3/3] Evaluation vs reference (merged GOLD files)...")

    if paths.gold_csv_path.exists():
        try:
            gold_files = read_gold_dir_per_file(paths.gold_csv_path)
    
            print("\n[eval] Per-file metrics (each STRAD_*_GENDER.csv):")
    
            for p, gold_df_one in gold_files:
                summary, diag = evaluate_gender_3class(
                    df_model=df_streets,
                    join_on=settings.join_on_model,
                    gold_df=gold_df_one,
                    unique_by="street",
                )
    
                cov = summary.get("coverage", {})
                matched = int(cov.get("matched_rows", 0)) if isinstance(cov, dict) else 0
    
                report = summary.get("report", {})
                labels = summary.get("labels", ["F", "M", "U"])
    
                # If sklearn isn't available, your code may create a report without per-class metrics.
                if not isinstance(report, dict) or not report:
                    print(f"\nFile: {p.name}")
                    print(f"  matched_rows={matched}")
                    print("  [WARN] No per-class report available (sklearn not installed).")
                    continue
    
                print(f"\nFile: {p.name}")
                print(f"  matched_rows={matched}")
                print_report_metrics(report, labels, title="  Metrics:")
    
            # Optional: still compute the merged evaluation exactly like before
            # (keeps your existing outputs report.csv, diagnostics.csv, confusion_matrix.pdf, etc.)
            gold_df_merged = read_gold_dir(paths.gold_csv_path)
    
            summary, diag = evaluate_gender_3class(
                df_model=df_streets,
                join_on=settings.join_on_model,
                gold_df=gold_df_merged,
                unique_by="street",
            )
    
            cov = summary.get("coverage", {})
            if cov:
                print(
                    "\n[eval] Merged GOLD metrics (all files combined): "
                    f"gold_rows={cov.get('gold_total_rows')}, "
                    f"gold_unique={cov.get('gold_unique_labels')}, "
                    f"model_rows={cov.get('model_total_rows')}, "
                    f"model_unique={cov.get('model_unique_ids')}, "
                    f"matched_rows={cov.get('matched_rows')}"
                )
    
            ensure_dir(paths.eval_diag_csv.parent)
            diag.to_csv(paths.eval_diag_csv, index=False)
            print(f"[eval] Wrote diagnostics CSV: {paths.eval_diag_csv}")
    
            diag[(diag["gold_gender"] == "F") & (~diag["correct"])].to_csv(paths.eval_diag_female_mis, index=False)
            print(f"[eval] Wrote misclassified female odonyms CSV: {paths.eval_diag_female_mis}")
    
            diag[(diag["gold_gender"] == "M") & (~diag["correct"])].to_csv(paths.eval_diag_male_mis, index=False)
            print(f"[eval] Wrote misclassified male odonyms CSV: {paths.eval_diag_male_mis}")
    
            diag[(diag["gold_gender"] == "U") & (diag["gender"] == "F") & (~diag["correct"])].to_csv(paths.eval_diag_undef_as_f, index=False)
            print(f"[eval] Wrote undefined misclassified as female CSV: {paths.eval_diag_undef_as_f}")
    
            diag[(diag["gold_gender"] == "U") & (diag["gender"] == "M") & (~diag["correct"])].to_csv(paths.eval_diag_undef_as_m, index=False)
            print(f"[eval] Wrote undefined misclassified as male CSV: {paths.eval_diag_undef_as_m}")
    
            plot_confusion_matrix_pdf(
                summary.get("confusion_raw", []),
                summary.get("confusion", []),
                summary.get("labels", []),
                paths.eval_confusion_pdf,
            )
    
            report = summary.get("report", {})
            if report:
                rep_df = pd.DataFrame(report).T
                rep_df.to_csv(paths.eval_report_csv, index=True)
                print(f"[eval] Wrote metrics CSV: {paths.eval_report_csv}")
    
                # keep your existing per-label prints if you want
                labels = summary.get("labels", [])
                for lbl in labels:
                    if lbl in report and isinstance(report[lbl], dict):
                        rr = report[lbl]
                        try:
                            p = float(rr.get("precision", 0.0))
                            r = float(rr.get("recall", 0.0))
                            f1 = float(rr.get("f1-score", 0.0))
                            sup = int(rr.get("support", 0))
                        except Exception:
                            continue
                        print(f"[eval] {lbl}: precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}, support={sup}")
    
                acc = report.get("accuracy", None)
                if acc is not None and not isinstance(acc, dict):
                    print(f"[eval] Overall accuracy: {float(acc):.4f}")
    
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Evaluation skipped due to error: {exc}")
    else:
        print(f"[INFO] GOLD folder not found: {paths.gold_csv_path}. Skipping evaluation.")

    print(f"Done. Outputs in: {paths.out_dir}")
    print(f"- Streets CSV: {paths.out_streets_csv}")
    print(f"- HTML map (municipal): {paths.map_html_muni}")
    print(f"- HTML map (province): {paths.map_html_prov}")
    print(f"- HTML map (region): {paths.map_html_region}")
    print(f"- HTML map (3 levels): {paths.map_html_3levels}")
    print(f"- PDF map (municipal): {paths.map_pdf_muni}")
    print(f"- PDF map (province): {paths.map_pdf_prov}")
    print(f"- PDF map (region): {paths.map_pdf_region}")
    print(f"- Confusion matrix PDF: {paths.eval_confusion_pdf}")
    print(f"- Province ECDF PDF: {paths.prov_ecdf_pdf}")
    print(f"- Eval diagnostics CSV: {paths.eval_diag_csv}")
    print(f"- Misclassified female odonyms CSV: {paths.eval_diag_female_mis}")
    print(f"- Misclassified male odonyms CSV: {paths.eval_diag_male_mis}")
    print(f"- Undefined->female CSV: {paths.eval_diag_undef_as_f}")
    print(f"- Undefined->male CSV: {paths.eval_diag_undef_as_m}")
    print(f"- Eval metrics CSV: {paths.eval_report_csv}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export gender stats from the SQLite streets DB pipeline.")
    parser.add_argument(
        "db",
        nargs="?",
        type=Path,
        default=Paths().db_path,
        help=f"Path to the SQLite database. Default: {Paths().db_path}",
    )
    args = parser.parse_args()
    sys.exit(main(Path(args.db)))