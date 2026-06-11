# Belief Updater V2 — Technical Documentation

## Overview

This document describes the redesigned probabilistic belief updater for the DSAIT4230 cooking intent recognition system. It covers what was built, why it was redesigned, how it works, and how it improved in response to supervisor feedback.

---

## Background — What Changed and Why

### Original Implementation (V1)

The first belief updater used **IDF-weighted exact string matching** to compute likelihoods. For each observation, it counted how many of a recipe's ingredients and actions had been seen, weighting rare terms more heavily using Inverse Document Frequency (IDF). Scores were combined with a stage consistency bonus and a contradiction penalty.

While functional on controlled test data, this approach had significant limitations identified through feedback:

**1. Lack of generalisability.** The system only worked if the recipe being prepared existed in the fixed recipe set. A recipe outside the set would produce meaningless likelihoods since matching relied on exact vocabulary.

**2. Synonym blindness.** Exact string matching meant "capsicum" and "bell pepper" were treated as completely different ingredients despite being the same thing. Similarly "whisk" and "mix" produced no overlap despite being semantically equivalent.

**3. Arbitrary weight values.** The scoring weights (0.45 for ingredients, 0.35 for actions, stage bonus of 0.2, contradiction penalty of 0.3) were set by intuition with no principled justification.


### Redesign Goals

In response to feedback, the redesigned system needed to:

- Handle recipes outside the training set (generalisability)
- Understand semantic similarity between related cooking terms (synonyms)
- Produce a belief distribution that moves meaningfully in response to observations and clarification answers
- Ensure clarification answers cause measurably larger entropy reduction than passive observation alone

---

## Architecture — Three Files

```
embedder.py           → embed any text → 768-dimensional vector, with disk caching
recipe_vectoriser.py  → recipe JSON → per-term vectors, saved to recipe_terms.json
belief_updater_v2.py  → maintain P(recipe|obs), update, entropy, IG_Q
```

---

## Embedder

**Model:** `all-mpnet-base-v2` from sentence-transformers (Reimers & Gurevych, 2019).

Chosen for the following reasons:

- Specifically trained to maximise semantic similarity between related phrases
- 768-dimensional vectors with strong semantic understanding of short phrases
- Handles cooking vocabulary well — trained on broad corpora including recipe content
- Runs entirely locally on CPU, no API dependency
- Well cited in NLP research, easy to justify in the paper

**Caching:** All embeddings are cached to `embedding_cache.json`. The same string is never embedded twice. This means the model only runs on startup and when genuinely new terms are encountered.

---

## Recipe Representation

Rather than embedding whole recipe descriptions into a single vector (which caused the similarity collapse problem described below), each recipe is represented as a **set of individual term vectors** — one vector per ingredient, one per action.

For example, the creamy cucumber salad recipe produces:

```
{
  "cucumber":       [0.12, -0.34, 0.91, ...],   # 768 dims
  "sour cream":     [0.45,  0.23, -0.11, ...],
  "white vinegar":  [...],
  "white sugar":    [...],
  "slice":          [...],
  "whisk":          [...],
  "mix":            [...],
  "refrigerate":    [...],
  ...
}
```

This is saved to `recipe_terms.json` and loaded at startup. New recipes are added by computing their term vectors and appending to the file — no retraining needed. This is what enables the growing recipe database approach.

### Why Individual Terms Rather Than Whole-Recipe Embeddings

An early version embedded the entire recipe as a single text string and computed cosine similarity between that vector and an observation vector. This failed because all cucumber salad recipes are semantically very similar as whole texts — they all produce nearly identical vectors. Cosine similarity between all recipe vectors and any observation vector came out around 0.94-0.96 regardless of what was observed. Softmax over near-identical values produces a near-uniform distribution — entropy barely dropped across the entire session.

Keeping terms separate preserves the discriminating signal. "Sour cream" as an individual vector sits in a very specific region of embedding space. When "sour cream" is observed, it strongly pulls similarity toward recipes that contain it, and away from recipes that do not.

---

## Similarity Function — Bidirectional F1 Matching

After several iterations, the system uses **bidirectional F1-style matching** between observed terms and recipe terms.

### Why Not Unidirectional Soft Matching

An intermediate version used one-directional soft matching: for each observed term, find its best-matching term in the recipe and average the scores. This was too generous — every recipe could match "cucumber" and "slice" perfectly since they appear in all recipes. Similarity scores ended up close across all recipes even after discriminating ingredients were observed.

### F1 Matching

The F1 approach matches in both directions:

**Coverage (recall):** For each recipe term, find its best cosine match in the observed terms. High coverage means most of the recipe's ingredients and actions have been seen. A recipe whose key ingredient (sour cream) has not been observed gets penalised here.

**Precision:** For each observed term, find its best cosine match in the recipe. High precision means what was observed belongs in this recipe. A recipe that doesn't contain sour cream gets penalised here when sour cream is observed.

**F1:** Harmonic mean of coverage and precision. A recipe only scores high when both directions agree — its terms have been observed AND what was observed fits the recipe.

```python
coverage  = mean(best_match(rec_term, obs_terms) for rec_term in recipe)
precision = mean(best_match(obs_term, rec_terms) for obs_term in observed)
F1        = 2 × coverage × precision / (coverage + precision)
```

A similarity threshold of 0.5 is applied — cosine similarity below 0.5 does not count as a match. This prevents weak semantic associations from inflating scores.

### Cosine Similarity Only

Euclidean distance was considered as an additional signal but was dropped. At the individual term level, cosine similarity captures semantic relatedness more reliably than euclidean distance in high-dimensional embedding space. Euclidean distance is sensitive to vector magnitude which varies across terms and adds noise rather than signal at this granularity.

---

## Belief Update

The Bayesian update structure from the original system is preserved:

```
P_t(r) ∝ P_{t-1}(r) × F1_similarity(obs, recipe_r)
```

Normalised so all probabilities sum to 1.

A softmax with temperature T converts raw F1 scores to a proper probability distribution before the Bayesian update:

```
softmax_sim(r) = exp(F1(r) / T) / sum(exp(F1(r') / T))
```

Temperature T = 0.1 is used. Lower temperature amplifies differences between F1 scores, producing a sharper distribution. This was necessary because F1 scores between similar salad recipes tend to be close together — without sharpening, the distribution remains flat.

---

## Clarification Answer Incorporation

When the human provides a clarification answer, the text is processed to extract compound terms matched against the known recipe vocabulary.

### Compound Term Extraction

This was a critical improvement. An earlier version split answers into individual words ("sour cream" → ["sour", "cream", "make", "creamy", "dressing"]). Individual words like "sour" and "cream" are ambiguous and add noise. "Sour cream" as a compound term maps precisely to the recipe vocabulary.

The extraction algorithm:

1. Generate all unigrams, bigrams, and trigrams from the answer text
2. Match each ngram against the set of all known recipe terms
3. Prefer longer matches — if "sour cream" matches, do not also add "sour" and "cream" separately
4. Filter common stopwords from unigrams
5. Also embed the full answer text as one term for broader semantic coverage

For the answer "I am adding sour cream to make a creamy dressing":
```
Extracted: ['sour cream', 'I am adding sour cream to make a creamy dressing']
```

This produces a precise, noise-free signal that maps directly to recipe vocabulary.

---

## Information Gain Measurement

The primary metric from the research question is:

```
IG_Q = H(belief before answer) − H(belief immediately after answer)
```

This is captured by snapshotting entropy immediately before and immediately after each answer is incorporated. The snapshot is stored permanently and not affected by subsequent observations. This gives a clean measurement of exactly what the clarification answer contributed, independent of what the VLM subsequently observes.

For comparison, IG_obs is computed per window:

```
IG_obs = H(belief before observation) − H(belief after observation)
```
---

## Alignment with Research Question

The research question is: *"How do different forms of clarification questions affect a VLM-based observer's ability to understand human intent over a sequence of observed actions, as measured by information gain?"*

Every component of the belief updater serves this measurement:

- **F1 matching** produces a well-calibrated belief distribution so entropy is meaningful
- **Bayesian update** correctly accumulates evidence across clips
- **IG_Q snapshot** measures exactly what the clarification answer contributed, independent of subsequent observations
- **Temperature** controls how much room clarification questions have to contribute — if passive observation converged too fast, IG_Q would be small by ceiling effects

The learning mechanism (updating weights from wrong predictions across sessions) is kept as a secondary system property separate from the main IG_Q comparison. Weights are frozen during evaluation sessions so they do not confound the measurement of question form effects.

---

## Known Limitations

**Recipes are still finite.** The system supports a growing recipe database but cannot handle completely unknown recipes with no representation. Adding a new recipe requires computing its term vectors — lightweight but still requires the recipe to be known at some point.

**Synonym coverage depends on embedding model.** The all-mpnet-base-v2 model handles common English cooking synonyms well but may struggle with regional terms or non-English ingredient names.

**Temperature is manually tuned.** T = 0.1 was chosen by inspection on the test session. It should be validated on real recorded sessions before final experiments.

**F1 threshold is fixed at 0.5.** Terms with cosine similarity below 0.5 do not count as matches. This value was set by inspection and should be validated empirically.

---

## Files

| File | Purpose |
|------|---------|
| `embedder.py` | Wraps all-mpnet-base-v2, disk caching |
| `recipe_vectoriser.py` | Builds recipe_terms.json from recipe JSONs |
| `belief_updater_v2.py` | Full belief update logic |
| `test_belief_v2.py` | Test session with 5 mock recipes |
| `recipe_terms.json` | Per-term vectors for all 30 recipes (generated) |
| `embedding_cache.json` | Cached embeddings (generated) |
