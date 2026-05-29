# Methodology Report: Similarity-Based Salad Recipe Dataset

## 1. Purpose
This dataset was constructed for an experiment in which cooking videos will be shown to a vision-language model (VLM). The VLM must maintain a probability belief state over possible candidate recipes and may ask clarifying questions when the observed ingredients or actions are insufficient to identify the recipe. The dataset therefore should not contain random salads, nor should it contain exact duplicates. It should contain a small set of recipes that are similar enough to create genuine ambiguity, but still distinct enough to be meaningful candidate recipes.

## 2. Input and output
The input was a salad recipe CSV containing 1,282 rows. After cleaning, removing unusable rows, and removing recipes with too few usable core ingredients, 1,269 usable unique salad recipes remained. The final output is `dataset.csv`, containing 30 selected recipes. The selected recipes were chosen automatically using ingredient/process similarity, not manually by name.

The script also creates a `graphs/` folder with validation figures. These graphs are only created if they do not already exist, unless the script is run with `--force-graphs`. This prevents accidental overwriting of figures already prepared for a paper.

## 3. Columns used and not used
Similarity is computed from observable recipe evidence: cleaned ingredients and recipe directions. Recipe titles are used only for deduplication and display. Titles are not used in the similarity score, because recipe names would not be visible to the VLM in a cooking video and would leak information. Metadata columns such as category, subcategory, ratings, URLs, or descriptions are not used for similarity. Long description fields are dropped to keep the final dataset small.

## 4. Ingredient cleaning
Raw ingredient strings contain quantities, units, preparation states, and notes, for example “1.5 cups apple slices, chopped”. The script converts these into simplified ingredient names. It removes quantities, units, common descriptors, punctuation, and preparation words, then canonicalizes some synonyms and compounds. The cleaned representation is intended to capture the ingredient identity rather than the exact textual phrasing.

Two ingredient columns are produced:
- `clean_ingredients`: a transparent simplified list of ingredients, retaining real ingredients such as salt, pepper, oil, vinegar, and mayonnaise.
- `core_ingredients_for_similarity`: a more discriminative list used for similarity scoring. Very common pantry items and seasonings are removed so that unrelated salads do not look similar just because they both contain salt, pepper, oil, or vinegar.

This separation is important scientifically: the dataset remains faithful to the recipes, while the similarity score focuses on ingredients that are more visually and procedurally informative in a cooking video. Bell pepper, jalapeno pepper, and similar visible pepper ingredients are retained as core ingredients; black pepper and generic pepper seasonings are removed from the similarity representation.

## 5. Ingredient similarity
The main ingredient similarity measure is normalized ingredient overlap. For two recipes A and B, let size(A) and size(B) be the numbers of core ingredients in each recipe, and let shared(A,B) be the number of shared core ingredients. The score is:

    similarity(A, B) = 0.5 * ( shared(A,B) / size(A) + shared(A,B) / size(B) )

This was chosen instead of plain Jaccard similarity. Jaccard divides the overlap by the union of both sets, which can strongly penalize recipes with different ingredient-list lengths. The normalized overlap score instead asks: “what fraction of each recipe is shared with the other recipe?” This is better suited to recipe comparison when one recipe may contain the same core ingredients plus additional ingredients. Bellingeri et al. (2025) use this normalized recipe-similarity-network weighting to account for recipe size and avoid size bias.

## 6. Process/action similarity
Process similarity is computed from the recipe directions. The script extracts a normalized direction text and also a simple `action_sequence` column by identifying common cooking verbs such as combine, mix, toss, chill, drain, season, cover, whisk, and serve. The direction text is compared using a lightweight TF-IDF cosine similarity implementation, inspired by recipe-matching and recipe-deduplication work that compares recipes using TF-IDF representations of ingredients and instructions.

The action sequence is also used for validation graphs, especially the action-transition graph. This graph is a simplified static analogue of recipe flow graph work, where recipe procedures are represented as sequences and interactions among ingredients, actions, tools, and intermediate products.

## 7. Combined similarity score
The final selection score combines ingredient and process similarity:

    combined_similarity = 0.85 * ingredient_similarity + 0.15 * process_similarity

Ingredients receive the highest weight because the experiment is about visual ambiguity in recipe recognition: ingredients are often directly visible in video and strongly constrain the belief state. Process similarity is still included because recipes with the same ingredients but different preparation procedures may be easier to distinguish from video actions.

## 8. Subset-selection algorithm
The script selects the final dataset as follows:
1. Load the salad recipe CSV.
2. Drop description columns to reduce size.
3. Normalize recipe names and remove duplicate names.
4. Clean ingredient strings into simplified ingredient names.
5. Remove pantry/seasoning items from the core similarity representation.
6. Keep only recipes with at least two core ingredients.
7. Remove exact duplicate core-ingredient sets.
8. Compute pairwise normalized ingredient overlap across all usable salads.
9. Compute pairwise process similarity from directions.
10. Combine ingredient and process similarities using the weighted score above.
11. Search from a strict threshold downward until it finds a connected similarity component large enough to contain the target number of recipes.
12. Within that component, choose the most central recipes: the recipes with the highest average similarity to the rest of the selected group.

For the current run, the selected similarity threshold was 0.63, and the final dataset contains 30 recipes.

## 9. Validation statistics
The selected dataset is substantially more internally similar than the full salad pool:

| Comparison | All usable salads | Selected dataset |
|---|---:|---:|
| Mean ingredient similarity | 0.056 | 0.396 |
| Mean process similarity | 0.062 | 0.100 |
| Mean combined similarity | not shown in terminal output | 0.351 |

This shows that the selected recipes form a more coherent and confusable subgroup than the full salad dataset. The final set is not just the top 30 most similar recipes by title; names are not used in the similarity score. The selected recipes are mainly cucumber/tomato/slaw/corn/avocado-style salads with overlapping ingredients and partially overlapping preparation steps.

## 10. Final dataset structure
The final `dataset.csv` contains the original useful recipe fields plus added columns for reproducibility and analysis. The important added columns are:
- `clean_ingredients`: JSON list of simplified ingredient names.
- `core_ingredients_for_similarity`: JSON list of ingredients used for similarity scoring.
- `clean_ingredient_count`: number of cleaned ingredients.
- `core_ingredient_count`: number of core ingredients used for similarity.
- `direction_text`: flattened recipe directions.
- `action_sequence`: JSON list of extracted cooking actions.
- `mean_similarity_to_dataset`: mean combined similarity to the other selected recipes.
- `mean_ingredient_similarity_to_dataset`: mean ingredient similarity to the other selected recipes.
- `mean_process_similarity_to_dataset`: mean process similarity to the other selected recipes.

In the current final dataset, recipes have on average 8.3 cleaned ingredients, with a range of 4 to 14. They have on average 5.9 core ingredients used for similarity, with a range of 3 to 12. The average number of recipe steps is 2.3, with a range of 1 to 5.

The most common core ingredients in the selected dataset are:
- cucumber: 24
- onion: 22
- tomato: 19
- white sugar: 13
- distilled white vinegar: 8
- bell pepper: 8
- coriander: 7
- avocado: 7
- lime: 6
- red wine vinegar: 4
- whole kernel corn: 4
- jalapeno pepper: 4
- white vinegar: 3
- dill: 3
- white wine vinegar: 3

The most common extracted actions are:
- combine: 12
- cover: 12
- step: 11
- season: 10
- chill: 8
- add: 8
- mix: 8
- toss: 7
- whisk: 5
- coat: 5
- stir: 4
- place: 4
- boil: 4
- drain: 4
- dress: 3

## 11. Validation figures
The script creates the following static SVG figures in the `graphs/` folder:
- `similarity_comparison.svg`: compares mean ingredient, process, and combined similarity between all salads and the selected dataset.
- `ingredient_similarity_distribution.svg`: compares the pairwise ingredient-similarity distribution of all salads with the selected dataset.
- `selected_similarity_heatmap.svg`: shows pairwise similarity within the selected 30-recipe dataset.
- `top_core_ingredients.svg`: shows the most frequent core ingredients in the selected dataset.
- `action_transition_graph.svg`: shows common sequential action transitions in the selected recipes.

Together, these figures support the claim that the final dataset is a similarity-based subgroup rather than a random salad subset. They also show that the selection is not only ingredient-driven: the process/action comparison gives a secondary validation of procedural coherence.

## 12. Limitations
The ingredient cleaning is heuristic rather than a trained ingredient parser. It is transparent and reproducible, but it may occasionally over-merge or under-merge ingredients. For example, “salt and pepper” variants should be excluded from core similarity, while visible pepper ingredients such as bell pepper should remain. The current script explicitly separates these cases. The action sequence is also heuristic: it extracts likely action verbs from directions, but it is not a full semantic recipe-flow parser. This is acceptable for dataset validation, but a later version could use a trained recipe named-entity / recipe-flow model for more precise action and object extraction.

## 13. Reproducibility command
Run the dataset construction with:

    python make_dataset.py --csv "salad_recipes.csv"

The default outputs are:
- `dataset.csv`
- `graphs/`
- optional methodology/report text if generated separately

## 14. References
- Bellingeri et al. (2025). The recipe similarity network: a new algorithm to extract relevant information from cookbooks. Scientific Reports. https://www.nature.com/articles/s41598-025-17189-6
- Bien et al. (2020). RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation. INLG 2020. https://aclanthology.org/2020.inlg-1.4/
- Yamakata, Mori and Carroll (2020). English Recipe Flow Graph Corpus. LREC 2020. https://aclanthology.org/2020.lrec-1.638/
- Trattner and Jannach (2019). An Evaluation of Recommendation Algorithms for Online Recipe Portals. https://www.christophtrattner.com/pubs/healthrecsys2019.pdf
