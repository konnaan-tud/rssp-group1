# Recipe Clip Folders — How to Add a New Recipe

Each subdirectory here holds the ordered video clips for one cooking
session. The three orchestrators auto-discover folders via
`--recipe <folder_name>` — **no code changes needed to add a new
recipe**, as long as you follow the conventions below.

---

## TL;DR — adding a new recipe

```bash
# 1. Make sure the recipe is in data/italian_recipe_scenes.json
#    (already covers 29 Italian pasta recipes)

# 2. Create a folder under data/clips/ named after the recipe
mkdir data/clips/arrabbiata

# 3. Drop your ordered clip files inside (leading zeros for ordering)
mv ~/Downloads/*.mp4 data/clips/arrabbiata/
ls data/clips/arrabbiata/
#   01_pour_water.mp4
#   02_slice_garlic.mp4
#   03_heat_oil.mp4
#   ...

# 4. Run the orchestrators — the loader auto-resolves "arrabbiata"
#    to "arrabbiata pasta" in italian_recipe_scenes.json
python orchestrator_obs_only.py    --recipe arrabbiata
python orchestrator_v2_vlm.py      --recipe arrabbiata --answer-mode text
python orchestrator_polar.py       --recipe arrabbiata --answer-mode text
```

That's it. No code changes. Each new folder works automatically.

---

## Folder naming — three options

| Option | When to use | Example |
|---|---|---|
| **Exact** (underscores → spaces) | Recipe name has multiple words | `carbonara_bianca/` → `"carbonara bianca"` |
| **Prefix shorthand** | Recipe name starts with a unique word | `arrabbiata/` → `"arrabbiata pasta"` |
| **Manifest file** | Folder name is ambiguous or memorable | `my_run/recipe.txt` with `carbonara` inside |

### Worked examples — what folder name to use

| You want recipe | Use folder name | Why |
|---|---|---|
| `carbonara` | `carbonara/` | exact name, no space needed |
| `carbonara bianca` | `carbonara_bianca/` | spaces → underscores |
| `pesto pasta` | `pesto/` | prefix shorthand (unique) |
| `arrabbiata pasta` | `arrabbiata/` | prefix shorthand |
| `pea pesto pasta` | `pea_pesto/` | `pea/` alone is ambiguous! |
| `pea butter pasta` | `pea_butter/` | `pea/` alone is ambiguous! |
| `cacio e pepe` | `cacio_e_pepe/` | spaces → underscores |
| `vodka sauce pasta` | `vodka/` | prefix shorthand |
| `clam pasta` | `clam/` | prefix shorthand |

### When auto-resolution fails

Two failure modes, both with clear error messages telling you exactly what to do:

**Ambiguous folder name** (multiple recipes match the prefix):
```
ValueError: Folder name 'pea' is ambiguous — matches multiple recipes:
pea butter pasta, pea pesto pasta. Either rename the folder to be more
specific (e.g. 'pea_butter_pasta'), or add a recipe.txt file in the
folder containing the exact recipe name.
```
→ rename `pea/` to `pea_pesto/` or `pea_butter/`.

**No matching recipe**:
```
ValueError: Folder name 'whatever' does not match any recipe in
data/italian_recipe_scenes.json. [...]
Available recipes: alfredo pasta, amatriciana pasta, arrabbiata pasta, ...
```
→ pick a name from the list, or add the recipe to the JSON first.

### Manifest override (`recipe.txt`)

If you can't or don't want to rename the folder, drop a one-line file:

```bash
# data/clips/my_evening_dinner/recipe.txt
echo "carbonara" > data/clips/my_evening_dinner/recipe.txt

python orchestrator_v2_vlm.py --recipe my_evening_dinner --answer-mode text
# [session] recipe='my_evening_dinner'  ground_truth='carbonara'  N clips
```

The manifest's first line is taken as the canonical recipe name. It overrides auto-resolution and is validated against the JSON.

---

## Clip file naming inside the folder

Files are loaded in **lexicographic (alphabetical) order**. Use leading zeros so the sequence is unambiguous:

```
✓ Good:                ✗ Bad:
01_pour_water.mp4      pour_water.mp4
02_crack_egg.mp4       crack_egg.mp4
03_grate_pecorino.mp4  grate_pecorino.mp4
...                    11_drain.mp4  ← would sort BEFORE 2_*
10_serve.mp4
```

**Supported extensions:** `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`

Anything else (image files, README, manifest, etc.) is ignored.

---

## Ignoring a folder (deprecated/backup/WIP)

Folders whose name starts with `_` or `.` are **invisible** to the
orchestrators — they won't appear in `--help`, can't be selected via
`--recipe`, and won't error out about unrecognised recipe names.

Use this for old or in-progress folders you want to keep around without
breaking the CLI:

```bash
mv data/clips/depricated_pesto data/clips/_depricated_pesto
```

Now `_depricated_pesto/` is silently ignored. Reverse by renaming back.

---

## Adding a recipe that's NOT in `italian_recipe_scenes.json`

The belief updater only considers recipes that are in the JSON (and embedded into `recipe_sequences.json`). If your test recipe isn't yet there:

```bash
# 1. Open data/italian_recipe_scenes.json and add an entry:
#    {
#      "dish": "your new recipe name",
#      "scenes": [
#        "A cook pours water into a large pot.",
#        ...
#      ]
#    }

# 2. Re-embed so the belief updater knows about it
python main.py --mode bootstrap

# 3. Now create the clip folder + run as usual
mkdir data/clips/your_recipe
# ... drop clips in ...
python orchestrator_v2_vlm.py --recipe your_recipe --answer-mode text
```

---

## See also

- `data/italian_recipe_scenes.json` — list of all 29 recipes the belief updater knows.
- `utils/clip_loader.py` — the auto-resolution logic itself, with docstrings.
- `scripts/prepare_clips.py` — optional helper for trimming raw videos to 5-second clips before placing them in a folder.
