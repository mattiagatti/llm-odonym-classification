# -*- coding: utf-8 -*-
"""
Classifier for Italian odonyms — precision-first 3-class version (self-contained).

Design:
- Heuristics-first:
    * Step 1: fast "Other" detection (numbers, dates, events, nature, constructions, roads, places)
    * Step 2: detect collectives vs individuals vs other
    * Step 3: infer gender deterministically when possible
- LLM only (local LLaMA endpoint):
    * used ONLY when heuristics are inconclusive at the category level
      (i.e., they cannot decide between 0/1/2 and return category_id=None)
- Deterministic gender for collectives (F/M/FM/U)
- Supports “Sant’…/Santa’…” and initials + roman numerals.
- Assumes odonym prefixes (Via, Piazza, …) are stripped upstream.

Output format:
    {
        "category_id": 0|1|2,
        "category": "Other"|"Individual persons"|"Collective persons",
        # gender semantics:
        # - "F":  female
        # - "M":  male
        # - "FM": collective that can reasonably include both female and male
        # - "U":  undefined / indeterminate (gender would make sense, but we can't infer it)
        # - "NA": not applicable (e.g. category_id == 0, "Other")
        "gender": "F"|"M"|"FM"|"U"|"NA",
        # who produced the final classification:
        # - "heuristic": heuristic-only
        # - "llm":       LLM fallback was used and accepted
        "source": "heuristic"|"llm"
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Dict, Optional

from openai import OpenAI

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Local LLaMA backend configuration
# ---------------------------------------------------------------------
LOCAL_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
LOCAL_BASE_URL = "http://localhost:8000/v1"  # vLLM / OpenAI-compatible local endpoint

# ---------------------------------------------------------------------
# 3-class categories
# ---------------------------------------------------------------------
CATEGORIES = {
    1: "Individual persons",
    2: "Collective persons",
    0: "Other",
}


def _normalize_gender(cat_id: int, gender: Optional[str]) -> str:
    """
    Normalize gender according to semantics:

    - For category 0 ("Other"): always "NA".
    - For persons (1 or 2):
        * "F", "M", "FM", "U" are accepted.
        * None or anything unexpected -> "U".
    """
    if cat_id == 0:
        return "NA"

    if gender in ("F", "M", "FM", "U"):
        return gender

    return "U"


def _make_output(cat_id: int, gender: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Normalize category_id and gender and produce the standard dict output."""
    if cat_id not in CATEGORIES:
        cat_id = 0
    norm_gender = _normalize_gender(cat_id, gender)
    return {
        "category_id": cat_id,
        "category": CATEGORIES[cat_id],
        "gender": norm_gender,
    }


# ---------------------------------------------------------------------
# Embedded ISTAT-style first-name lists (curated core; extend as needed)
# ---------------------------------------------------------------------
FEMALE_NAMES_ISTAT = {
    # very common
    "ada", "maria", "anna", "giulia", "francesca", "chiara", "sofia", "martina", "valentina",
    "alessia", "federica", "elena", "roberta", "laura", "silvia", "serena", "valeria",
    "arianna", "giorgia", "beatrice", "veronica", "alessandra", "nicole", "camilla",
    "gaia", "ilaria", "noemi", "bianca", "aurora", "sara", "teresa", "irene", "carla",
    "luisa", "eleonora", "caterina", "rossella", "angela", "maddalena", "nadia", "elisa",
    "manuela", "raffaella", "flavia", "erika", "paola", "patrizia", "lucia", "alessandra",
    "daniela", "cristina", "simona", "francesca", "alba",
    # saints & historical recurring
    "rita", "agnese", "chiara", "caterina", "lucia", "teresa", "beatrice", "giovanna",
    "maddalena", "rosa", "antonietta", "elisabetta", "claudia", "isabella", "mariachiara",
    "marianna", "mariateresa",
    # modern
    "sofia", "greta", "alice", "emma", "anna", "aurora", "giada", "arianna", "beatrice",
    "chiara", "carlotta", "eva", "francesca", "lucrezia", "marta", "sofia", "noemi",
    "matilde", "ginevra",
    # NEW: extra female names from diagnostics
    "berlinda", "annetta", "carolina", "emilia", "viola", "catulla",
    "teresina", "mara", "liala", "mary", "julitta",
}

MALE_NAMES_ISTAT = {
    # very common
    "giuseppe", "giovanni", "francesco", "antonio", "luigi", "marco", "andrea", "paolo",
    "pietro", "carlo", "stefano", "gabriele", "matteo", "luca", "davide", "riccardo",
    "federico", "alessandro", "roberto", "enzo", "salvatore", "michele", "simone",
    "daniele", "lorenzo", "tommaso", "nicola", "vincenzo", "edoardo", "giorgio",
    "giacomo", "emanuele", "alberto", "massimo", "claudio", "pierpaolo", "raffaele",
    "valerio", "fabio", "franco", "leonardo", "giulio", "piero", "ugo", "pino",
    # popes/saints & recurring odonym forms
    "pio", "benedetto", "giovanni paolo", "paolo vi", "pio xii", "giovanni xxiii",
    "papa", "san", "santo",
    # modern
    "samuele", "christian", "diego", "jacopo", "alessio", "nicolò", "davide",
    "gabriel", "manuel", "mattia",
    # NEW: male saints often appearing as "Sant'...":
    "abbondanzio", "abbondio", "adriano", "agostino", "agrippino",
    "albino", "ambrogio", "anastasio", "angelo", "antonino", "apollinare",
    "arialdo", "elia", "eusebio", "eustorgio", "eutichio", "evasio",
    "ilario", "imerio", "iorio", "isidoro", "onofrio",
}

FEM_EXAMPLES = set(FEMALE_NAMES_ISTAT)
MALE_EXAMPLES = set(MALE_NAMES_ISTAT)

# Weak female tokens which easily appear as surnames/common nouns;
# they should not, by themselves, force a female decision.
WEAK_FEMALE_TOKENS = {
    "rosa",
    # You can extend this by inspecting diagnostics_undefined_misclassified_as_female.csv
    # e.g. "viola", "stella", ...
}


# ---------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------
def _token_regex_from(names: set[str]) -> re.Pattern:
    """Build a regex that matches any of the single-token names."""
    singles = sorted([n for n in names if " " not in n])
    if not singles:
        # "match nothing" regex
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(" + "|".join(map(re.escape, singles)) + r")\b", re.IGNORECASE)


RE_FNAME_F = _token_regex_from(FEM_EXAMPLES)

# Marian / female devotions (treated as single female person – the Virgin Mary)
RE_MARIAN_FEMALE = re.compile(
    r"\b("
    r"madonna|madonnina|madonnetta|"
    r"immacolata|addolorata|assunta|annunziata"
    r")\b",
    re.IGNORECASE,
)

# Collectives
RE_FEMALE_COLLECTIVE = re.compile(
    r"\b("
    r"sante|suore|monache|madonne|sorelle|professoresse|operaie|poliziotte|soldatesse|"
    r"donne|tessitrici|madri|orsoline|canossiane"
    r")\b",
    re.IGNORECASE,
)

# ONLY groups that are necessarily male (traditional male-only roles/orders)
RE_MALE_COLLECTIVE = re.compile(
    r"\b("
    r"frati|monaci|padri|cappuccini|vescovi|"
    r"fratelli|alpini|bersaglieri|cacciatori"
    r")\b",
    re.IGNORECASE,
)

# Collective types that can be mixed or generic
RE_ANY_COLLECTIVE = re.compile(
    r"\b("
    r"santi|martiri|partigiani|artisti|caduti|combattenti|volontari|studenti|maestri|allievi|"
    r"benefattori|carabinieri|professori|operai|poliziotti|soldati|"
    r"donatori|pensionati|magistri"
    r")\b",
    re.IGNORECASE,
)

# Titles: individuals
# NOTE: generic "Sant'..." is handled in infer_gender_single.
FEM_TITLES = re.compile(
    r"\b(madonna|santa|beata|suor|regina|principessa|duchessa|contessa|marchesa|baronessa)\b",
    re.IGNORECASE,
)

# Strict masculine titles: don't match inside words
MAS_TITLES = re.compile(
    r"(?<![a-zà-öø-ÿ])("
    r"san|santo|beato|don|padre|mons\.?|monsignore|principe|re|duca|conte|marchese|barone|papa"
    r")(?![a-zà-öø-ÿ])",
    re.IGNORECASE,
)

# Initials + surname + optional Roman numerals (e.g., "G. B. Vico", "Pio XII")
RE_INITIAL_SURNAME = re.compile(
    r"\b(?:[A-Z]\.?\s*){1,3}[A-Z][a-zà-öø-ÿ]+(?:-[A-Z][a-zà-öø-ÿ]+)?(?:\s+[IVXLCDM]{1,5})?\b"
)

# Capitalized tokens or Roman numerals (for names like "Giovanni Paolo II")
RE_CAP_TOKENS = re.compile(
    r"\b(?:[A-Z][a-zà-öø-ÿ]+(?:-[A-Z][a-zà-öø-ÿ]+)?|[IVXLCDM]{2,})\b"
)

# ---------------------------------------------------------------------
# Fast "Other" detectors (numbers, dates, events, nature, constructions, roads, places)
# ---------------------------------------------------------------------
RE_NUMBERISH = re.compile(
    r"^\s*(\d+([°º])?|[ivxlcdm]{1,6}\.?)\s*$",
    re.IGNORECASE,
)

RE_DATE_IT = re.compile(
    r"\b(\d{1,2}|[ivxlcdm]{1,4})\s+("
    r"gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
    r"settembre|ottobre|novembre|dicembre)\b",
    re.IGNORECASE,
)

EVENT_TOKENS = {
    "battaglia",
    "resistenza",
    "liberazione",
    "unità",
    "unita",
    "costituzione",
    "armistizio",
    "insurrezione",
    "strage",
    "rivoluzione",
    "eccidio",
    "presa di",
    "trattato",
    "plebiscito",
}
RE_EVENTI = re.compile("|".join(re.escape(t) for t in EVENT_TOKENS), re.IGNORECASE)

FAUNA_FLORA = {
    # animals
    "leone",
    "lupo",
    "aquila",
    "orso",
    "cervo",
    "tigre",
    "gatto",
    "rondine",
    "falco",
    # plants/trees
    "quercia",
    "rovere",
    "olmo",
    "pino",
    "tiglio",
    "olivo",
    "acero",
    "cipresso",
    "castagno",
    # landscape
    "bosco",
    "collina",
    "prato",
    "valle",
    "montagna",
    "lago",
    "fiume",
}
RE_FAUNA_FLORA = re.compile(r"\b(" + "|".join(sorted(FAUNA_FLORA)) + r")\b", re.IGNORECASE)

COSTRUZIONI = {
    "ponte",
    "viadotto",
    "galleria",
    "stazione",
    "bastioni",
    "porta",
    "castello",
    "forte",
    "torre",
    "rotonda",
    "monumento",
    "teatro",
    "ospedale",
    "università",
    "cimitero",
    "muro",
    "arsenale",
    "arco",
    "museo",
}
RE_COSTR = re.compile(r"\b(" + "|".join(sorted(COSTRUZIONI)) + r")\b", re.IGNORECASE)

RE_ENTI = re.compile(
    r"\b(anas|strada\s+statale|strada\s+regionale|ss\d+|sr\d+|sp\d+|"
    r"strada\s+provinciale)\b",
    re.IGNORECASE,
)

RE_INFERIORI = re.compile(
    r"\b(vicinale|interpoderale|campestre|carr[a]?bile|poderale|traversa)\b",
    re.IGNORECASE,
)

GEO_TOKENS = {
    "roma",
    "napoli",
    "milano",
    "torino",
    "firenze",
    "venezia",
    "genova",
    "bologna",
    "lombardia",
    "piemonte",
    "sicilia",
    "sardegna",
    "toscana",
    "lazio",
    "alpi",
    "appennino",
    "po",
    "adige",
    "ticino",
    "como",
    "brescia",
    "bergamo",
    "parigi",
    "londra",
    "berlino",
}
RE_GEO = re.compile(r"\b(" + "|".join(sorted(GEO_TOKENS)) + r")\b", re.IGNORECASE)


# ---------------------------------------------------------------------
# Name / token helpers
# ---------------------------------------------------------------------
def contains_multiword_name(label: str, names: set[str]) -> bool:
    """Return True if label contains a multi-token name from `names`."""
    low = label.lower()
    return any((" " in n) and (n in low) for n in names)


def tokens_from_label(label: str) -> list[str]:
    """Tokenize a label into lowercase alphabetic tokens."""
    return re.findall(r"[a-zà-öø-ÿ]+", label.lower())


# Surname-shape classifier (used to *downgrade* weak male-only evidence)
def looks_like_surname(token: str) -> bool:
    """
    Very small heuristic: return True if the token *looks* like a typical
    Italian surname form (often plural or derived), e.g. Rossi, Marini, Antonini.
    Used only to *downgrade* weak male evidence to "U".
    """
    t = token.lower()
    if len(t) <= 3:
        return False
    return t.endswith(("i", "ini", "oni", "etti", "ari", "ani", "ano"))


def known_firstname_gender(label: str) -> Optional[str]:
    """
    If the label clearly contains a known first name with definite gender,
    return "F" or "M". Otherwise return None.

    Logic:
    - If only female hits -> "F" (but if *all* are weak, treat as ambiguous -> None)
    - If only male hits   -> "M" (but if all male tokens look like bare surnames, -> None)
    - If both male and female hits:
        * If "maria" appears with other male cues (names or titles), treat as "M"
        * Otherwise ambiguous -> None
    """
    low = label.lower()
    tokens = tokens_from_label(low)

    fem_multi = contains_multiword_name(low, FEM_EXAMPLES)
    male_multi = contains_multiword_name(low, MALE_EXAMPLES)
    fem_tokens = [t for t in tokens if t in FEM_EXAMPLES]
    male_tokens = [t for t in tokens if t in MALE_EXAMPLES]

    female_hit = fem_multi or bool(fem_tokens)
    male_hit = male_multi or bool(male_tokens)

    # Mixed case -> resolve "Maria" as male middle name when clearly male-coded
    if female_hit and male_hit:
        if "maria" in tokens:
            has_other_male = any(t in MALE_EXAMPLES and t != "maria" for t in tokens)
            if has_other_male or MAS_TITLES.search(label):
                return "M"
        # Otherwise ambiguous: don't guess
        return None

    if female_hit:
        # Are all female tokens weak? If yes, treat as ambiguous -> None
        strong_fem = any(t not in WEAK_FEMALE_TOKENS for t in fem_tokens)
        if fem_multi or strong_fem:
            return "F"
        return None

    if male_hit:
        # If all male tokens "look like surnames", treat as ambiguous
        strong_male = False
        if male_multi:
            strong_male = True
        else:
            for t in male_tokens:
                if not looks_like_surname(t):
                    strong_male = True
                    break
        if strong_male:
            return "M"
        return None

    return None


# ---------------------------------------------------------------------
# Special non-person "San/Santo/Santa + devotional noun" patterns
# ---------------------------------------------------------------------
# Things like "Santa Croce", "Santo Spirito", "Santo Sepolcro", etc.
NON_PERSON_SAN_SANTA_TOKENS = {
    "croce",
    "spirito",
    "sepolcro",
    "nome",
    "volto",
    "rosario",
    "cuore",
    "sacramento",
}


def is_non_person_san_santa(label: str) -> bool:
    """
    Return True if the label begins with San/Santo/Santa followed by
    a devotional/abstract noun like "Croce", "Spirito", etc.
    These are usually churches / concepts, not persons.
    """
    tokens = tokens_from_label(label)
    if len(tokens) < 2:
        return False
    if tokens[0] in ("san", "santo", "santa"):
        # if any of the following tokens is clearly a devotional noun -> Other
        for t in tokens[1:]:
            if t in NON_PERSON_SAN_SANTA_TOKENS:
                return True
    return False


# ---------------------------------------------------------------------
# Heuristics: person vs collective vs other
# ---------------------------------------------------------------------
def is_collective(label: str) -> bool:
    """Return True if label clearly describes a group of people."""
    low = label.lower()
    return bool(
        RE_FEMALE_COLLECTIVE.search(low)
        or RE_MALE_COLLECTIVE.search(low)
        or RE_ANY_COLLECTIVE.search(low)
    )


def is_single_person_shape(label: str) -> bool:
    """
    Return True if the label structurally looks like a single person
    (titles, initial+surname, multi-capital names, etc.).
    Assumes odonym prefixes stripped upstream.
    """
    if FEM_TITLES.search(label) or MAS_TITLES.search(label):
        return True
    if RE_INITIAL_SURNAME.search(label):
        return True

    # Known first-name tokens also give a strong signal
    if known_firstname_gender(label) is not None:
        return True

    caps = RE_CAP_TOKENS.findall(label)
    return len(caps) >= 2


# ---------------------------------------------------------------------
# Heuristics: gender inference
# ---------------------------------------------------------------------
def infer_gender_single(label: str) -> str:
    """
    Infer gender for a single person when possible.
    Returns "F", "M", or "U" (Undefined) when ambiguous.
    """
    low = label.lower()

    # Strong lexicon-based decision
    name_gender = known_firstname_gender(label)
    if name_gender in ("F", "M"):
        return name_gender

    # Titles as hints
    if FEM_TITLES.search(label):
        return "F"
    if MAS_TITLES.search(label):
        # We are now back to the simpler behaviour:
        # any San/Santo/Don/Padre etc. implies "M"
        return "M"

    # generic "sant'" often denotes a saint; default to F only if followed by a common feminine
    if re.search(r"\bsant(?:a)?(?:'|’)", low):
        if RE_FNAME_F.search(low) or contains_multiword_name(low, FEM_EXAMPLES):
            return "F"
        # It could be masculine ("Sant'Angelo") or other; leave undefined
        return "U"

    # initial + surname or unclear -> Undefined
    if RE_INITIAL_SURNAME.search(label):
        return "U"

    return "U"


def infer_gender_collective(label: str) -> str:
    """
    Deterministic gender for collectives:
    - "F"  -> explicitly feminine groups (Suore, Sante, Operaie, ...).
    - "M"  -> explicitly male-only groups (Frati, Monaci, Padri, Cappuccini, Vescovi, Fratelli, Alpini, Bersaglieri, Cacciatori, ...).
    - "FM" -> grammatically masculine or generic plurals that can reasonably
              include both female and male members (Partigiani, Caduti, Carabinieri,
              Professori, Operai, Soldati, Donatori, Pensionati, ...).
    - "U"  -> fallback when we are not even sure it's a collective of people.
    """
    low = label.lower()
    if RE_FEMALE_COLLECTIVE.search(low):
        return "F"
    if RE_MALE_COLLECTIVE.search(low):
        return "M"
    if RE_ANY_COLLECTIVE.search(low):
        # generic/mixed or grammatically masculine but potentially mixed group
        return "FM"
    return "U"


# ---------------------------------------------------------------------
# Heuristic-first classifier (3 classes)
# ---------------------------------------------------------------------
def heuristic_classify(label: str) -> Dict[str, Optional[str]]:
    """
    Heuristic-only classification.
    Returns:
        - a dict with category_id in {0,1,2} when heuristics are confident, or
        - {"category_id": None, "category": None, "gender": None}
          when heuristics are inconclusive and the LLM should decide.
    """
    s = (label or "").strip()
    if not s:
        # empty label -> no decision, let the LLM handle it
        return {"category_id": None, "category": None, "gender": None}

    low = s.lower()

    # -----------------------------------------------------------------
    # 0) Fast "Other" cases: pure numbers, dates, nature, constructions,
    #    roads, and geographic labels that clearly do NOT denote persons.
    #    These are hard-coded as Other and NEVER go to the LLM.
    # -----------------------------------------------------------------
    if RE_NUMBERISH.match(s):
        return _make_output(0, None)

    if RE_DATE_IT.search(low):
        return _make_output(0, None)

    if RE_FAUNA_FLORA.search(low):
        return _make_output(0, None)

    if RE_COSTR.search(low):
        return _make_output(0, None)

    if RE_ENTI.search(low) or RE_INFERIORI.search(low):
        return _make_output(0, None)

    if RE_GEO.search(low):
        return _make_output(0, None)

    # Special non-person "San/Santo/Santa + devotional noun" cases
    # e.g. "Santa Croce", "Santo Spirito", "Santo Sepolcro"
    if is_non_person_san_santa(s):
        return _make_output(0, None)

    # Marian / female devotions -> always treat as a single female person
    # (e.g. Immacolata, Addolorata, Assunta, Madonnina, ...)
    if RE_MARIAN_FEMALE.search(low):
        return _make_output(1, "F")

    # Events: treat as Other only when they are NOT clearly person collectives.
    # This ensures "Partigiani Caduti" etc. remain Collective persons.
    if RE_EVENTI.search(low) and not is_collective(s):
        return _make_output(0, None)

    # -----------------------------------------------------------------
    # 1) Collective persons (deterministic gender)
    # -----------------------------------------------------------------
    if is_collective(s):
        return _make_output(2, infer_gender_collective(s))

    # -----------------------------------------------------------------
    # 2) Individual persons
    #    - If label is clearly a known first name with definite gender,
    #      treat as individual even if single token ("Giulia", "Luigi", ...).
    # -----------------------------------------------------------------
    name_gender = known_firstname_gender(s)
    if name_gender in ("F", "M"):
        return _make_output(1, name_gender)

    # Otherwise, use shape-based detection (titles, initials, capitals...)
    if is_single_person_shape(s):
        return _make_output(1, infer_gender_single(s))  # "F"/"M"/"U"

    # -----------------------------------------------------------------
    # 3) Fallback: heuristics are inconclusive -> let the LLM decide
    # -----------------------------------------------------------------
    return {
        "category_id": None,
        "category": None,
        "gender": None,
    }


# ---------------------------------------------------------------------
# LLM client + prompt (local LLaMA only)
# ---------------------------------------------------------------------


LLM_PROMPT_TEMPLATE = """
You are classifying a SINGLE Italian odonym label (street/square name WITHOUT the
prefix like "Via", "Piazza", etc.) into EXACTLY ONE of these categories:

0 -> Other
1 -> Individual persons
2 -> Collective persons

Return STRICT JSON with EXACT KEYS ONLY:
{"category_id": int, "gender": "F"|"M"|"FM"|"U"|"NA"}

GENERAL POLICY (VERY IMPORTANT)
- Most odonyms are Italian. Prefer Italian interpretations of names and words.
- Be CONSERVATIVE and PRECISION-FIRST.
- If you are NOT CLEARLY SURE that the label refers to a person (individual or collective),
  you MUST use category_id = 0.
- NEVER guess the gender: if there is ANY doubt, use "U" (Undefined), except for
  category 0 where gender is always "NA".
- You MAY use general world knowledge about well-known historical figures
  (e.g. "Buonarroti" = Michelangelo, "Montessori" = Maria Montessori, "Manzoni" =
  Alessandro Manzoni, "Diaz" = Armando Diaz), including when only an initial + surname
  is given (e.g. "A. Manzoni", "A. Diaz").
- You MUST NOT use world knowledge about specific streets, cities, monuments,
  battles, dates, or places. Only world knowledge about PERSONS is allowed.

CATEGORY 0: OTHER
- Use this when the label does NOT clearly describe a person or a group of persons.
  Examples: dates, events, places, rivers, mountains, abstract concepts, generic words.
- gender for category 0: ALWAYS "NA".

CATEGORY 1: INDIVIDUAL PERSONS
Treat the label as an individual person when:
- There are clear personal cues such as titles like San, Santo, Santa, Sant', Beato,
  Beata, Don, Padre, Suor, Regina, Papa, etc.; OR
- It is a full name ("First Last"); OR
- It is an initial + surname ("G. Verdi", "A. Manzoni", "A. Diaz"); OR
- It is a surname alone that clearly refers to a well-known person ("Buonarroti",
  "Montessori", "Manzoni", "Dante", "Foscolo", "Verdi", ...).

Gender for category 1:
- Use "F" or "M" ONLY when evident from the label or from universally known
  information about that person (e.g. Maria Montessori -> "F", Alessandro Manzoni -> "M").
- If the gender is not clear, use "U".
- Do NOT use "FM" or "NA" for category 1.

CATEGORY 2: COLLECTIVE PERSONS
Treat the label as a collective of persons when it clearly refers to a group of people
in the plural (e.g., Santi, Martiri, Partigiani, Frati, Caduti, Volontari, Artisti,
Carabinieri, Professori, Operai, Soldati, Suore, Sante, etc.).

Gender for category 2:
- "F"  for clearly feminine plurals (Sante, Suore, Monache, Operaie, Soldatesse, ...).
- "M"  for groups that are necessarily masculine (Frati, Monaci, Padri, Cappuccini,
  Vescovi, ...).
- "FM" for grammatically masculine but potentially mixed groups (Santi, Martiri,
  Partigiani, Benefattori, Carabinieri, Professori, Operai, Poliziotti, Soldati, ...).
- "U"  ONLY if you are sure it is a collective of persons but the label does NOT
  allow you to decide between "F", "M", and "FM".
- Do NOT use "NA" for category 2.

EXAMPLES (NOT OUTPUT FORMAT, JUST GUIDANCE)
- "Buonarroti"    -> {"category_id": 1, "gender": "M"}
- "Montessori"    -> {"category_id": 1, "gender": "F"}
- "A. Manzoni"    -> {"category_id": 1, "gender": "M"}
- "A. Diaz"       -> {"category_id": 1, "gender": "M"}
- "L. da Vinci"   -> {"category_id": 1, "gender": "M"}
- "Roma"          -> {"category_id": 0, "gender": "NA"}
- "Vittorio Veneto" -> {"category_id": 0, "gender": "NA"}
- "Solferino"     -> {"category_id": 0, "gender": "NA"}
- "Santi Martiri" -> {"category_id": 2, "gender": "FM"}

If you are unsure whether the label refers to a person (individual or collective)
or to something else, ALWAYS choose category_id = 0 and gender = "NA".
If you are unsure between "F", "M", "FM", and "U", ALWAYS choose "U"
(except for category 0 which must be "NA").

Label: "<<LABEL>>"
""".strip()


def build_llm_prompt(label: str) -> str:
    """Fill the LLM prompt template with the raw label (just stripped)."""
    return LLM_PROMPT_TEMPLATE.replace("<<LABEL>>", (label or "").strip())


def llm_classify(label: str, model: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    LLM-based classification using a local LLaMA endpoint.

    - For category_id == 1 (individuals), use the LLM's gender but clamp to {F, M, U}.
    - For category_id == 2 (collectives), use the LLM's gender but clamp to {F, M, FM, U}.
    - For category_id == 0 (Other), always force gender = "NA".
    """
    client = OpenAI(
        base_url=LOCAL_BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY", "local"),  # any non-empty string
    )

    req = {
        "model": model or LOCAL_MODEL,
        "messages": [
            {"role": "system", "content": "Return exactly one JSON object. No commentary."},
            {"role": "user", "content": build_llm_prompt(label)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    resp = client.chat.completions.create(**req)
    text = resp.choices[0].message.content or "{}"

    try:
        data = json.loads(text)
    except Exception:
        return _make_output(0, None)

    cat_id = data.get("category_id")
    if not isinstance(cat_id, int) or cat_id not in (0, 1, 2):
        cat_id = 0

    if cat_id == 1:
        gender = data.get("gender")
        # For individuals, allow only F/M/U; everything else -> "U"
        if gender not in ("F", "M", "U"):
            gender = "U"
        return _make_output(1, gender)

    if cat_id == 2:
        gender = data.get("gender")
        if gender not in ("F", "M", "FM", "U"):
            gender = "U"
        return _make_output(2, gender)

    # cat_id == 0
    return _make_output(0, None)


def needs_llm_fallback(result: Dict[str, Optional[str]]) -> bool:
    """
    Call the LLM whenever heuristics are inconclusive at the CATEGORY level.

    - If category_id is None -> heuristics could not decide between 0/1/2,
      so the LLM must classify.
    - If category_id is 0, 1, or 2 -> we trust the heuristic decision and
      DO NOT ask the LLM (even if gender == "U").
    """
    cat_id = result.get("category_id")
    return cat_id is None


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def classify(label: str, model: Optional[str] = None, **_: object) -> Dict[str, Optional[str]]:
    """
    Heuristic-first classification with local LLaMA (3-class).

    Args:
        label: odonym label, with prefixes (Via, Piazza, ...) already stripped.

    Returns:
        {
            "category_id": 0|1|2,
            "category": str,
            "gender": "F"|"M"|"FM"|"U"|"NA",
            "source": "heuristic"|"llm"
        }
    """
    heuristic = heuristic_classify(label)

    # If we don't need the LLM, clearly heuristic-only
    if not needs_llm_fallback(heuristic):
        return {**heuristic, "source": "heuristic"}

    # Heuristics inconclusive -> ask LLM to classify
    try:
        llm_out = llm_classify(label, model=model)
        if llm_out.get("category_id") in CATEGORIES:
            # Successful LLM decision
            return {**llm_out, "source": "llm"}
    except Exception as exc:
        log.warning("LLM fallback failed (%s): %s", LOCAL_MODEL, exc)

    # If LLM fails for any reason, last-resort fallback: Other
    return {**_make_output(0, None), "source": "heuristic"}


# ---------------------------------------------------------------------
# Debug main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("[DEBUG] Embedded female names:", len(FEM_EXAMPLES))
    print("[DEBUG] Embedded male names:", len(MALE_EXAMPLES))

    tests = [
        "Rita Levi-Montalcini",        # -> (1, "F")   heuristic
        "G. Verdi",                    # -> (1, "U")   heuristic
        "G. B. Vico",                  # -> (1, "U")   heuristic
        "Sant'Agnese",                 # -> (1, "F")   heuristic (feminine name)
        "Sant'Ambrogio",               # -> (1, "M")   heuristic (male name)
        "Padre Luigi Maria Monti",     # -> (1, "M")   heuristic ("Maria" male middle)
        "Giuseppe Maria Stampa",       # -> (1, "M")   heuristic
        "Ampelio Rosa",                # likely -> (1, "U") now (weak female token)
        "MARINI",                      # likely -> (1, "U") now (surname-shape)
        "Santa Croce",                 # -> (0, "NA")  now (non-person devotional)
        "Santo Spirito",               # -> (0, "NA")  now
        "Santo Sepolcro",              # -> (0, "NA")  now
        "San Martino",                 # -> (1, "M")   now (back to male)
        "San Giovanni",                # -> (1, "M")   (clear male name)
        "Papa Giovanni XXIII",         # -> (1, "M")   heuristic
        "Pio XII",                     # -> (1, "M")   heuristic
        "Giovanni Paolo II",           # -> (1, "M")   heuristic
        "Sante Martiri",               # -> (2, "F")   heuristic
        "Suore Benedettine",           # -> (2, "F")   heuristic
        "Frati Cappuccini",            # -> (2, "M")   heuristic
        "Fratelli Cervi",              # -> (2, "M")   heuristic
        "Alpini Caduti",               # -> (2, "M")   heuristic
        "Donatori del Sangue",         # -> (2, "FM")  heuristic
        "Partigiani Caduti",           # -> (2, "FM")  heuristic
        "Sorelle Minime",              # -> (2, "F")   heuristic
        "Carabinieri Caduti",          # -> (2, "FM")  heuristic
        "Professori Caduti",           # -> (2, "FM")  heuristic
        "Vittoria",                    # -> LLM (likely 0, "NA")
        "Libertà",                     # -> LLM (likely 0, "NA")
        "Roma",                        # -> (0, "NA")  heuristic (geo token)
        "25 Aprile",                   # -> (0, "NA")  heuristic (date)
        "Ponte Garibaldi",             # -> (0, "NA")  heuristic (construction)
        "SP11",                        # -> (0, "NA")  heuristic (road of other authority)
        "Giulia",                      # -> (1, "F")   heuristic
        "Luigi",                       # -> (1, "M"),  heuristic
    ]

    for x in tests:
        print(f"{x:30s} -> {classify(x)}")