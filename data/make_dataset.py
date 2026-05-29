"""
Create a similarity-based salad recipe dataset and simple validation figures.

Usage in Terminal / PowerShell:
    python make_dataset.py --csv salad_recipes.csv

Main output:
    dataset.csv

Figure output folder:
    graphs/

The script is deliberately lightweight:
- no matplotlib
- no networkx
- no scikit-learn

It uses recipe titles only for deduplication/display, never for similarity scoring.
Similarity is based on cleaned ingredient names and simple action sequences from directions.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TITLE_CANDIDATES = ["recipe_title", "recipe_name", "title", "name"]
INGREDIENT_CANDIDATES = ["ingredients", "ingredient_list", "recipe_ingredients"]
DIRECTION_CANDIDATES = ["directions", "instructions", "steps", "method"]
NORMALIZED_TITLE_CANDIDATES = ["normalized_recipe_name", "normalized_title"]

UNIT_WORDS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon", "teaspoons", "tsp",
    "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs", "gram", "grams", "g",
    "kg", "ml", "milliliter", "milliliters", "liter", "liters", "pinch", "dash",
    "package", "packages", "packet", "packets", "can", "cans", "jar", "jars", "bottle",
    "bottles", "box", "boxes", "bag", "bags", "clove", "cloves", "slice", "slices",
    "piece", "pieces", "bunch", "bunches", "head", "heads", "stalk", "stalks", "sprig",
    "sprigs", "leaf", "leaves", "large", "medium", "small"
}

DESCRIPTOR_WORDS = {
    "fresh", "frozen", "dried", "dry", "canned", "ripe", "raw", "cooked", "uncooked",
    "boiled", "roasted", "toasted", "grilled", "baked", "fried", "softened", "melted",
    "warm", "cold", "hot", "chopped", "sliced", "diced", "minced", "crushed", "grated",
    "shredded", "peeled", "seeded", "hulled", "quartered", "halved", "cubed", "julienned",
    "drained", "rinsed", "packed", "divided", "optional", "finely", "coarsely", "thinly",
    "roughly", "ground", "extra", "virgin", "low", "fat", "reduced", "free", "unsalted",
    "salted", "sweetened", "unsweetened", "plain", "light", "style", "flavored", "flavour",
    "taste", "needed", "plus", "more"
}

# These are real ingredients, so they stay in clean_ingredients.
# They are excluded from core_ingredients_for_similarity because they are too common
# and can make unrelated recipes look similar.
PANTRY_FOR_SIMILARITY = {
    "salt", "pepper", "black pepper", "water", "sugar", "oil", "vegetable oil", "canola oil",
    "olive oil", "vinegar", "mayonnaise", "mustard", "honey", "lemon juice", "lime juice",
    "garlic", "onion powder", "garlic powder"
}

COMPOUND_FIXES = {
    "olive oil": "olive oil",
    "vegetable oil": "vegetable oil",
    "canola oil": "canola oil",
    "sesame oil": "sesame oil",
    "red wine vinegar": "red wine vinegar",
    "white wine vinegar": "white wine vinegar",
    "apple cider vinegar": "apple cider vinegar",
    "balsamic vinegar": "balsamic vinegar",
    "soy sauce": "soy sauce",
    "fish sauce": "fish sauce",
    "cream cheese": "cream cheese",
    "blue cheese": "blue cheese",
    "feta cheese": "feta cheese",
    "cheddar cheese": "cheddar cheese",
    "parmesan cheese": "parmesan cheese",
    "goat cheese": "goat cheese",
    "bell pepper": "bell pepper",
    "green onion": "green onion",
    "red onion": "onion",
    "white onion": "onion",
    "yellow onion": "onion",
    "romaine lettuce": "lettuce",
    "iceberg lettuce": "lettuce",
    "cherry tomato": "tomato",
    "grape tomato": "tomato",
}

SYNONYMS = {
    "scallion": "green onion",
    "scallions": "green onion",
    "spring onion": "green onion",
    "mayo": "mayonnaise",
    "cilantro": "coriander",
    "garbanzo bean": "chickpea",
    "garbanzo beans": "chickpea",
    "chickpeas": "chickpea",
    "cranberries": "cranberry",
    "strawberries": "strawberry",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "apples": "apple",
    "oranges": "orange",
    "walnuts": "walnut",
    "pecans": "pecan",
}

ACTION_VERBS = [
    "add", "arrange", "bake", "beat", "blend", "boil", "chill", "chop", "coat", "combine",
    "cook", "cover", "cut", "dice", "drain", "dress", "fold", "fry", "garnish", "grate",
    "grill", "heat", "layer", "marinate", "mix", "peel", "place", "pour", "process",
    "refrigerate", "rinse", "roast", "season", "serve", "slice", "sprinkle", "stir", "toss",
    "transfer", "whisk"
]

STOPWORDS = {
    "and", "or", "with", "without", "into", "over", "under", "until", "then", "than",
    "the", "a", "an", "in", "on", "for", "of", "to", "from", "by", "at", "about"
}


def find_column(columns: Iterable[str], candidates: list[str], required: bool = True) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


def normalize_title(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(recipe|recipes|easy|best|homemade|classic|simple)\b", " ", text)
    return " ".join(text.split())


def parse_list_field(value: object) -> list[str]:
    """Parse a field that may be a Python/JSON-like list, newlines, semicolons, or plain text."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    if "\n" in text:
        return [x.strip() for x in text.splitlines() if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    # Last fallback for comma-separated ingredient strings.
    if "," in text and len(text) > 80:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("oes"):
        return token[:-2]
    if len(token) > 3 and token.endswith("es") and not token.endswith(("ses", "ss")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def clean_ingredient_line(line: object) -> str:
    text = normalize_text(line)
    if not text:
        return ""

    for phrase, canonical in sorted(COMPOUND_FIXES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return canonical

    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.split(r",| - | or to taste| to taste| as needed| such as", text)[0]
    text = text.translate(str.maketrans({"¼": " ", "½": " ", "¾": " ", "⅓": " ", "⅔": " ", "⅛": " ", "⅜": " ", "⅝": " ", "⅞": " "}))
    text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = []
    for token in text.split():
        token = singularize(token)
        if token in STOPWORDS or token in UNIT_WORDS or token in DESCRIPTOR_WORDS:
            continue
        tokens.append(token)

    if not tokens:
        return ""

    name = " ".join(tokens)
    name = SYNONYMS.get(name, name)
    words = name.split()
    if len(words) > 4:
        name = " ".join(words[-3:])
    return name.strip()


def clean_ingredients(value: object) -> list[str]:
    cleaned = []
    for line in parse_list_field(value):
        ingredient = clean_ingredient_line(line)
        if ingredient:
            cleaned.append(ingredient)
    return list(dict.fromkeys(cleaned))


def join_directions(value: object) -> str:
    return " ".join(parse_list_field(value))


def split_steps(value: object) -> list[str]:
    steps = []
    for item in parse_list_field(value):
        for part in re.split(r"(?<=[.!?])\s+", item.strip()):
            part = part.strip()
            if part:
                steps.append(part)
    return steps


def primary_action(step: str) -> str:
    text = normalize_text(step)
    for verb in ACTION_VERBS:
        if re.search(rf"\b{verb}(?:s|ed|ing)?\b", text):
            return verb
    return "step"


def action_sequence(value: object) -> list[str]:
    return [primary_action(step) for step in split_steps(value)]




def tokenize_for_tfidf(text: object) -> list[str]:
    text = normalize_text(text)
    tokens = []
    for token in re.sub(r"[^a-z\s]", " ", text).split():
        token = singularize(token)
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.append(token)
    # Include bigrams so phrases such as "fold in" or "boiling water" contribute too.
    bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
    return tokens + bigrams


def tfidf_cosine_matrix(texts: list[str], max_features: int = 3000) -> np.ndarray:
    """Small TF-IDF cosine implementation to avoid scikit-learn as a dependency."""
    token_lists = [tokenize_for_tfidf(t) for t in texts]
    n = len(token_lists)
    df_counter = Counter()
    tf_counters = []
    for tokens in token_lists:
        counts = Counter(tokens)
        tf_counters.append(counts)
        df_counter.update(counts.keys())

    # Keep terms that appear often enough to be useful, but not too many.
    vocab = [term for term, _ in df_counter.most_common(max_features)]
    vocab_set = set(vocab)
    idf = {term: math.log((n + 1) / (df_counter[term] + 1)) + 1.0 for term in vocab}

    vectors = []
    postings = defaultdict(list)
    for i, counts in enumerate(tf_counters):
        vec = {}
        for term, count in counts.items():
            if term not in vocab_set:
                continue
            weight = (1.0 + math.log(count)) * idf[term]
            vec[term] = weight
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm > 0:
            vec = {term: weight / norm for term, weight in vec.items()}
        vectors.append(vec)
        for term, weight in vec.items():
            postings[term].append((i, weight))

    similarity = np.zeros((n, n), dtype=np.float32)
    for posting in postings.values():
        m = len(posting)
        for a in range(m):
            i, wi = posting[a]
            for b in range(a + 1, m):
                j, wj = posting[b]
                value = wi * wj
                similarity[i, j] += value
                similarity[j, i] += value
    np.fill_diagonal(similarity, 0.0)
    return similarity


def normalized_overlap_matrix(sets: list[set[str]]) -> np.ndarray:
    vocab = sorted({item for s in sets for item in s})
    index = {item: i for i, item in enumerate(vocab)}
    X = np.zeros((len(sets), len(vocab)), dtype=np.float32)
    for row, s in enumerate(sets):
        for item in s:
            X[row, index[item]] = 1.0

    intersections = X @ X.T
    sizes = X.sum(axis=1)
    sizes[sizes == 0] = np.nan
    similarity = 0.5 * (intersections / sizes[:, None] + intersections / sizes[None, :])
    similarity = np.nan_to_num(similarity, nan=0.0).astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def upper_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=np.float32)
    return matrix[np.triu_indices_from(matrix, k=1)]


def mean_upper(matrix: np.ndarray) -> float:
    values = upper_values(matrix)
    return float(values.mean()) if values.size else 0.0


def quantile_upper(matrix: np.ndarray, q: float) -> float:
    values = upper_values(matrix)
    return float(np.quantile(values, q)) if values.size else 0.0


def connected_components_above(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    n = matrix.shape[0]
    seen = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = np.flatnonzero(matrix[node] >= threshold)
            for nb in neighbors:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
        components.append(component)
    return components


def select_dense_subset(combined: np.ndarray, target_size: int) -> tuple[list[int], float]:
    """Select a dense component, then take the most central recipes inside it."""
    best_indices = None
    best_threshold = 0.0
    best_mean = -1.0

    # Start strict and relax until a component can supply target_size recipes.
    for threshold in np.round(np.arange(0.80, 0.09, -0.01), 2):
        components = connected_components_above(combined, float(threshold))
        candidates = [c for c in components if len(c) >= target_size]
        if not candidates:
            continue
        for component in candidates:
            sub = combined[np.ix_(component, component)]
            centrality = sub.sum(axis=1)
            local = np.argsort(-centrality)[:target_size]
            chosen = [component[i] for i in local]
            chosen_mean = mean_upper(combined[np.ix_(chosen, chosen)])
            if chosen_mean > best_mean:
                best_indices = chosen
                best_threshold = float(threshold)
                best_mean = chosen_mean
        break

    if best_indices is None:
        centrality = combined.sum(axis=1)
        best_indices = np.argsort(-centrality)[:target_size].tolist()

    selected_matrix = combined[np.ix_(best_indices, best_indices)]
    order = np.argsort(-selected_matrix.sum(axis=1))
    return [best_indices[i] for i in order], best_threshold


def histogram(values: np.ndarray, bins: int = 20) -> list[int]:
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return counts.tolist()


def svg_bar_chart(path: Path, title: str, labels: list[str], series: dict[str, list[float]]) -> None:
    if path.exists():
        return
    width, height = 900, 520
    margin_left, margin_bottom, margin_top = 90, 90, 70
    plot_w, plot_h = width - margin_left - 40, height - margin_top - margin_bottom
    max_value = max([0.001] + [v for values in series.values() for v in values])
    max_value = min(1.0, max(0.1, math.ceil(max_value * 10) / 10))
    n_groups = len(labels)
    n_series = len(series)
    group_w = plot_w / n_groups
    bar_w = group_w / (n_series + 1.4)
    greys = ["#444444", "#888888", "#BBBBBB", "#DDDDDD"]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="35" text-anchor="middle" font-size="22" font-family="Arial">{html.escape(title)}</text>')

    # Axes.
    x0, y0 = margin_left, margin_top + plot_h
    parts.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="black"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+plot_w}" y2="{y0}" stroke="black"/>')
    for tick in range(0, 6):
        val = max_value * tick / 5
        y = y0 - (val / max_value) * plot_h
        parts.append(f'<line x1="{x0-5}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#DDDDDD"/>')
        parts.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{val:.2f}</text>')

    for si, (name, values) in enumerate(series.items()):
        for gi, value in enumerate(values):
            x = x0 + gi * group_w + 0.25 * group_w + si * bar_w
            h = (value / max_value) * plot_h
            y = y0 - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{h:.1f}" fill="{greys[si % len(greys)]}"/>')
            parts.append(f'<text x="{x+bar_w*0.42:.1f}" y="{y-5:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{value:.2f}</text>')
    for gi, label in enumerate(labels):
        x = x0 + gi * group_w + group_w / 2
        parts.append(f'<text x="{x:.1f}" y="{y0+25}" text-anchor="middle" font-size="13" font-family="Arial">{html.escape(label)}</text>')

    legend_x, legend_y = width - 220, 65
    for si, name in enumerate(series.keys()):
        y = legend_y + si * 24
        parts.append(f'<rect x="{legend_x}" y="{y}" width="16" height="16" fill="{greys[si % len(greys)]}"/>')
        parts.append(f'<text x="{legend_x+24}" y="{y+13}" font-size="13" font-family="Arial">{html.escape(name)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_histogram(path: Path, title: str, all_values: np.ndarray, selected_values: np.ndarray, bins: int = 20) -> None:
    if path.exists():
        return
    width, height = 900, 520
    left, top, bottom = 75, 70, 85
    plot_w, plot_h = width - left - 40, height - top - bottom
    all_counts = histogram(all_values, bins)
    sel_counts = histogram(selected_values, bins)
    max_count = max(all_counts + sel_counts + [1])
    bar_w = plot_w / bins
    y0 = top + plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="35" text-anchor="middle" font-size="22" font-family="Arial">{html.escape(title)}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{y0}" stroke="black"/>')
    parts.append(f'<line x1="{left}" y1="{y0}" x2="{left+plot_w}" y2="{y0}" stroke="black"/>')

    for i in range(bins):
        x = left + i * bar_w
        h1 = all_counts[i] / max_count * plot_h
        h2 = sel_counts[i] / max_count * plot_h
        parts.append(f'<rect x="{x+2:.1f}" y="{y0-h1:.1f}" width="{bar_w*0.42:.1f}" height="{h1:.1f}" fill="#BBBBBB"/>')
        parts.append(f'<rect x="{x+bar_w*0.50:.1f}" y="{y0-h2:.1f}" width="{bar_w*0.42:.1f}" height="{h2:.1f}" fill="#444444"/>')
        if i % 2 == 0:
            label = f"{i/bins:.1f}"
            parts.append(f'<text x="{x:.1f}" y="{y0+18}" font-size="10" font-family="Arial">{label}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-size="14" font-family="Arial">Similarity score</text>')
    parts.append(f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20,{top+plot_h/2})" text-anchor="middle" font-size="14" font-family="Arial">Pair count</text>')
    parts.append(f'<rect x="650" y="65" width="16" height="16" fill="#BBBBBB"/><text x="674" y="78" font-size="13" font-family="Arial">all salads</text>')
    parts.append(f'<rect x="650" y="90" width="16" height="16" fill="#444444"/><text x="674" y="103" font-size="13" font-family="Arial">selected dataset</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    if path.exists():
        return
    n = matrix.shape[0]
    cell = 18
    left, top = 260, 40
    width = left + n * cell + 40
    height = top + n * cell + 260
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="50%" y="25" text-anchor="middle" font-size="20" font-family="Arial">Selected dataset similarity heatmap</text>')
    for i in range(n):
        safe_label = html.escape(labels[i][:30])
        y = top + i * cell + cell * 0.75
        parts.append(f'<text x="{left-8}" y="{y:.1f}" text-anchor="end" font-size="9" font-family="Arial">{safe_label}</text>')
        parts.append(f'<text x="{left+i*cell+cell/2:.1f}" y="{top+n*cell+8}" transform="rotate(70 {left+i*cell+cell/2:.1f},{top+n*cell+8})" font-size="9" font-family="Arial">{safe_label}</text>')
    for r in range(n):
        for c in range(n):
            val = 1.0 if r == c else float(matrix[r, c])
            # grayscale: lower similarity = light, higher similarity = dark
            shade = int(245 - 190 * max(0.0, min(1.0, val)))
            color = f"rgb({shade},{shade},{shade})"
            x = left + c * cell
            y = top + r * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white" stroke-width="0.5"/>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_top_ingredients(path: Path, ingredients: list[list[str]], top_n: int = 20) -> None:
    if path.exists():
        return
    counts = Counter(x for row in ingredients for x in row)
    top = counts.most_common(top_n)
    if not top:
        return
    labels, values = zip(*top)
    max_value = max(values)
    width, height = 900, 600
    left, top_margin, bar_h = 190, 50, 22
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="50%" y="30" text-anchor="middle" font-size="22" font-family="Arial">Most common core ingredients in selected dataset</text>')
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top_margin + i * (bar_h + 5)
        bar_w = (value / max_value) * 620
        parts.append(f'<text x="{left-10}" y="{y+16}" text-anchor="end" font-size="13" font-family="Arial">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="#666666"/>')
        parts.append(f'<text x="{left+bar_w+8:.1f}" y="{y+16}" font-size="13" font-family="Arial">{value}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_action_transition_graph(path: Path, action_lists: list[list[str]]) -> None:
    if path.exists():
        return
    edge_counts = Counter()
    action_counts = Counter()
    for actions in action_lists:
        action_counts.update(actions)
        for a, b in zip(actions, actions[1:]):
            if a != b:
                edge_counts[(a, b)] += 1
    top_edges = edge_counts.most_common(18)
    if not top_edges:
        return

    # Simple static layout: source actions on left, target actions on right.
    left_actions = []
    right_actions = []
    for (a, b), _ in top_edges:
        if a not in left_actions:
            left_actions.append(a)
        if b not in right_actions:
            right_actions.append(b)
    width, height = 900, max(520, 70 + 34 * max(len(left_actions), len(right_actions)))
    x_left, x_right = 250, 650
    y_left = {a: 80 + i * 34 for i, a in enumerate(left_actions)}
    y_right = {a: 80 + i * 34 for i, a in enumerate(right_actions)}
    max_edge = max(w for _, w in top_edges)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#555"/></marker></defs>')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="50%" y="30" text-anchor="middle" font-size="22" font-family="Arial">Common sequential action transitions</text>')
    parts.append('<text x="250" y="55" text-anchor="middle" font-size="14" font-family="Arial">earlier action</text>')
    parts.append('<text x="650" y="55" text-anchor="middle" font-size="14" font-family="Arial">next action</text>')

    for (a, b), w in top_edges:
        stroke_w = 1.0 + 5.0 * (w / max_edge)
        parts.append(f'<line x1="{x_left+60}" y1="{y_left[a]}" x2="{x_right-60}" y2="{y_right[b]}" stroke="#666" stroke-width="{stroke_w:.1f}" marker-end="url(#arrow)" opacity="0.75"/>')
        mx = (x_left + x_right) / 2
        my = (y_left[a] + y_right[b]) / 2
        parts.append(f'<text x="{mx}" y="{my-4:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{w}</text>')

    for a in left_actions:
        parts.append(f'<rect x="{x_left-70}" y="{y_left[a]-14}" width="140" height="26" rx="5" fill="#EEEEEE" stroke="#555"/>')
        parts.append(f'<text x="{x_left}" y="{y_left[a]+5}" text-anchor="middle" font-size="13" font-family="Arial">{html.escape(a)} ({action_counts[a]})</text>')
    for a in right_actions:
        parts.append(f'<rect x="{x_right-70}" y="{y_right[a]-14}" width="140" height="26" rx="5" fill="#EEEEEE" stroke="#555"/>')
        parts.append(f'<text x="{x_right}" y="{y_right[a]+5}" text-anchor="middle" font-size="13" font-family="Arial">{html.escape(a)} ({action_counts[a]})</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="salad_recipes.csv", help="Input salad recipe CSV")
    parser.add_argument("--out", default="dataset.csv", help="Final selected dataset CSV")
    parser.add_argument("--graphs-dir", default="graphs", help="Folder for SVG validation figures")
    parser.add_argument("--target-size", type=int, default=30, help="Number of recipes to select")
    parser.add_argument("--ingredient-weight", type=float, default=0.85, help="Weight for ingredient similarity in selection")
    parser.add_argument("--force-graphs", action="store_true", help="Overwrite existing SVG graphs")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    title_col = find_column(df.columns, TITLE_CANDIDATES)
    ingredient_col = find_column(df.columns, INGREDIENT_CANDIDATES)
    direction_col = find_column(df.columns, DIRECTION_CANDIDATES)
    normalized_title_col = find_column(df.columns, NORMALIZED_TITLE_CANDIDATES, required=False)

    # Drop long free-text description column if present.
    df = df.drop(columns=[c for c in df.columns if c.lower() == "description"], errors="ignore")

    if normalized_title_col is None:
        df["normalized_recipe_name"] = df[title_col].apply(normalize_title)
        normalized_title_col = "normalized_recipe_name"

    raw_rows = len(df)
    df = df[df[normalized_title_col].fillna("").astype(str).str.strip() != ""].copy()
    df = df.drop_duplicates(normalized_title_col, keep="first").copy()

    df["clean_ingredients"] = df[ingredient_col].apply(clean_ingredients)
    df["core_ingredients_for_similarity"] = df["clean_ingredients"].apply(
        lambda xs: [x for x in xs if x not in PANTRY_FOR_SIMILARITY]
    )
    df["clean_ingredient_count"] = df["clean_ingredients"].apply(len)
    df["core_ingredient_count"] = df["core_ingredients_for_similarity"].apply(len)
    df["direction_text"] = df[direction_col].apply(join_directions)
    df["action_sequence"] = df[direction_col].apply(action_sequence)

    # Keep recipes with enough non-pantry ingredients and remove exact duplicate ingredient sets.
    df = df[df["core_ingredient_count"] >= 2].copy()
    df["_ingredient_signature"] = df["core_ingredients_for_similarity"].apply(lambda xs: "||".join(sorted(xs)))
    before_exact = len(df)
    df = df.drop_duplicates("_ingredient_signature", keep="first").drop(columns="_ingredient_signature").reset_index(drop=True)
    exact_duplicates_removed = before_exact - len(df)

    if len(df) < args.target_size:
        raise ValueError(f"Only {len(df)} usable recipes remain; target size is {args.target_size}.")

    ingredient_sets = [set(xs) for xs in df["core_ingredients_for_similarity"]]
    ingredient_similarity = normalized_overlap_matrix(ingredient_sets)
    process_similarity = tfidf_cosine_matrix(df["direction_text"].fillna("").astype(str).tolist())

    ingredient_weight = min(1.0, max(0.0, args.ingredient_weight))
    combined_similarity = ingredient_weight * ingredient_similarity + (1.0 - ingredient_weight) * process_similarity
    np.fill_diagonal(combined_similarity, 0.0)

    selected_indices, selected_threshold = select_dense_subset(combined_similarity, args.target_size)
    dataset = df.iloc[selected_indices].copy().reset_index(drop=True)

    selected_ingredient = ingredient_similarity[np.ix_(selected_indices, selected_indices)]
    selected_process = process_similarity[np.ix_(selected_indices, selected_indices)]
    selected_combined = combined_similarity[np.ix_(selected_indices, selected_indices)]

    dataset["mean_similarity_to_dataset"] = selected_combined.sum(axis=1) / max(1, len(dataset) - 1)
    dataset["mean_ingredient_similarity_to_dataset"] = selected_ingredient.sum(axis=1) / max(1, len(dataset) - 1)
    dataset["mean_process_similarity_to_dataset"] = selected_process.sum(axis=1) / max(1, len(dataset) - 1)

    # Store list columns as JSON strings, so the CSV remains readable and reproducible.
    dataset_to_save = dataset.copy()
    dataset_to_save["clean_ingredients"] = dataset_to_save["clean_ingredients"].apply(json.dumps)
    dataset_to_save["core_ingredients_for_similarity"] = dataset_to_save["core_ingredients_for_similarity"].apply(json.dumps)
    dataset_to_save["action_sequence"] = dataset_to_save["action_sequence"].apply(json.dumps)
    dataset_to_save.to_csv(args.out, index=False)

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    if args.force_graphs:
        for p in graphs_dir.glob("*.svg"):
            p.unlink()

    all_ing_values = upper_values(ingredient_similarity)
    selected_ing_values = upper_values(selected_ingredient)

    svg_bar_chart(
        graphs_dir / "similarity_comparison.svg",
        "Similarity comparison: all salads vs selected dataset",
        labels=["ingredient", "process", "combined"],
        series={
            "all salads mean": [
                mean_upper(ingredient_similarity),
                mean_upper(process_similarity),
                mean_upper(combined_similarity),
            ],
            "selected mean": [
                mean_upper(selected_ingredient),
                mean_upper(selected_process),
                mean_upper(selected_combined),
            ],
        },
    )
    svg_histogram(
        graphs_dir / "ingredient_similarity_distribution.svg",
        "Ingredient similarity distribution",
        all_ing_values,
        selected_ing_values,
    )
    svg_heatmap(
        graphs_dir / "selected_similarity_heatmap.svg",
        selected_combined,
        dataset[title_col].astype(str).tolist(),
    )
    svg_top_ingredients(
        graphs_dir / "top_core_ingredients.svg",
        dataset["core_ingredients_for_similarity"].tolist(),
    )
    svg_action_transition_graph(
        graphs_dir / "action_transition_graph.svg",
        dataset["action_sequence"].tolist(),
    )

    print("Done")
    print(f"Input rows:                         {raw_rows:,}")
    print(f"Usable unique salad recipes:         {len(df):,}")
    print(f"Exact duplicate ingredient sets removed: {exact_duplicates_removed:,}")
    print(f"Selected recipes in dataset:         {len(dataset):,}")
    print(f"Selection threshold used:            {selected_threshold:.2f}")
    print(f"All salads mean ingredient similarity:      {mean_upper(ingredient_similarity):.3f}")
    print(f"Dataset mean ingredient similarity:         {mean_upper(selected_ingredient):.3f}")
    print(f"All salads mean process similarity:          {mean_upper(process_similarity):.3f}")
    print(f"Dataset mean process similarity:             {mean_upper(selected_process):.3f}")
    print(f"Output CSV:                         {args.out}")
    print(f"Graphs folder:                      {args.graphs_dir}")


if __name__ == "__main__":
    main()
