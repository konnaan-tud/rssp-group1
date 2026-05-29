"""
Select salad recipes from a recipe CSV and remove duplicate recipes by name.

Usage in Terminal:
    python3 select_unique_salads.py --csv "recipes(1).csv"

Outputs one file by default:
    unique_salad_recipes_no_description.csv

The script does NOT use recipe names for similarity scoring. Names are only used
to remove repeated copies of the same recipe that appear in multiple categories.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


TITLE_CANDIDATES = ["recipe_title", "recipe_name", "title", "name"]
CATEGORY_CANDIDATES = ["category", "subcategory", "cuisine_path", "type"]
TEXT_CANDIDATES = ["ingredients", "directions"]


def find_column(columns, candidates, required=True):
    lower_to_original = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def normalize_title(title):
    """Normalize names so repeated category copies collapse to one recipe."""
    text = "" if pd.isna(title) else str(title).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(recipe|recipes|easy|best|homemade|classic|simple)\b", " ", text)
    return " ".join(text.split())


def contains_salad(series):
    return series.astype(str).str.contains(r"\bsalads?\b", case=False, na=False, regex=True)


def contains_dressing(series):
    return series.astype(str).str.contains(r"\bdressings?\b", case=False, na=False, regex=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="recipes.csv", help="Input recipe CSV")
    parser.add_argument("--out", default="salad_recipes.csv", help="Output CSV")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only use category/subcategory fields for the salad filter, ignoring recipe title.",
    )
    parser.add_argument(
        "--keep-dressings",
        action="store_true",
        help="Keep salad dressing recipes. By default they are removed.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    title_col = find_column(df.columns, TITLE_CANDIDATES, required=True)
    category_cols = [c for c in CATEGORY_CANDIDATES if c in df.columns]
    if not category_cols and args.metadata_only:
        raise ValueError(
            "No category/subcategory column found. Remove --metadata-only "
            "to also filter based on recipe title."
        )

    # Main salad filter: use metadata categories plus recipe title.
    # We do NOT use the description because phrases like "serve this with a salad" are noisy.
    salad_mask = pd.Series(False, index=df.index)
    dressing_mask = pd.Series(False, index=df.index)

    for col in category_cols:
        salad_mask |= contains_salad(df[col])
        dressing_mask |= contains_dressing(df[col])

    if not args.metadata_only:
        salad_mask |= contains_salad(df[title_col])
        dressing_mask |= contains_dressing(df[title_col])

    if not args.keep_dressings:
        salad_mask &= ~dressing_mask

    salads = df.loc[salad_mask].copy()
    salads["normalized_recipe_name"] = salads[title_col].apply(normalize_title)
    salads = salads[salads["normalized_recipe_name"] != ""].copy()

    # Prefer rows with more complete recipe information when the same title appears
    # multiple times under different categories/subcategories.
    info_cols = [c for c in TEXT_CANDIDATES if c in salads.columns]
    salads["_completeness_score"] = 0
    for col in info_cols:
        salads["_completeness_score"] += salads[col].fillna("").astype(str).str.len()

    # Prefer rows where the actual category/subcategory is salad, not just title match.
    salads["_salad_metadata_score"] = 0
    for col in category_cols:
        salads["_salad_metadata_score"] += contains_salad(salads[col]).astype(int)

    salads = salads.sort_values(
        by=["_salad_metadata_score", "_completeness_score"],
        ascending=[False, False],
    )

    unique_salads = salads.drop_duplicates("normalized_recipe_name", keep="first")
    unique_salads = unique_salads.drop(columns=["_completeness_score", "_salad_metadata_score"])

    # Drop the description column from the final dataset to save space.
    # It is not needed for the video/VLM task, where ingredients and directions matter.
    description_cols = [c for c in unique_salads.columns if c.lower() == "description"]
    if description_cols:
        unique_salads = unique_salads.drop(columns=description_cols)

    unique_salads = unique_salads.sort_values(title_col).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    unique_salads.to_csv(args.out, index=False)

    print("Unique salad subset created")
    print(f"Input rows:                  {len(df):,}")
    print(f"Rows matching salad filter:  {len(salads):,}")
    print(f"Unique salad recipe names:   {len(unique_salads):,}")
    print(f"Duplicate salad rows removed:{len(salads) - len(unique_salads):,}")
    print(f"Output:                      {args.out}")


if __name__ == "__main__":
    main()
