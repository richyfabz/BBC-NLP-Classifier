# BBC News NLP Classification Pipeline

A multi-phase NLP pipeline that takes 2,225 raw BBC News articles and automatically
discovers meaningful sub-categories, extracts named entities, and powers a keyword-based
article recommendation engine without any predefined labels.

## Dataset

| Category      | Articles |
|---------------|----------|
| Business      | 510      |
| Sport         | 511      |
| Entertainment | 386      |
| Politics      | 417      |
| Tech          | 401      |
| **Total**     | **2,225**|

Raw .txt files, one article per file, organised into category folders.

## Architecture overview

```
Raw .txt files
      ↓
Phase 0 — Foundation (TF-IDF, NMF, LDA baseline)
      ↓
Phase 1 — Sub-category discovery (BGE-M3 → UMAP → HDBSCAN → BERTopic)
      ↓
Phase 2 — Named Entity Recognition (spaCy NER → role classification)
      ↓
Phase 3 — Recommendation engine (entity keyword search + displacy rendering)
```

---

## Phase 0: Foundation & exploration

**Notebook:** `bbc_nlp.ipynb`

Traditional NLP baseline to understand the dataset before moving to transformers.

- Loaded 2,225 articles, removed 106 duplicates → 2,119 unique articles
- Built TF-IDF matrix: 2,119 × 1,000
- Ran LDA topic modelling → coherence score: **0.5973**
- Ran NMF topic modelling → coherence score: **0.6441** (NMF outperformed LDA)

**Key lesson:** TF-IDF + NMF works well for broad topic separation but struggles
with story-specific news clusters inside a single category.

## Phase 1: Sub-category discovery

**Notebooks:** `business.ipynb`, `sport.ipynb`, `entertainment.ipynb`,
              `politics.ipynb`, `tech.ipynb`

A 6-module BERTopic pipeline runs independently on each category.

### Pipeline modules

| Module | Tool | Purpose |
|--------|------|---------|
| 1 | Data ingestion | Light cleaning, deduplication |
| 2 | BGE-M3 (BAAI/bge-m3) | 1024-dim semantic embeddings, 8192 token context |
| 3 | UMAP | 1024 → 5 dimensions, n_neighbors=15, random_state=42 |
| 4 | HDBSCAN | Automatic cluster detection, noise labelling |
| 5a | CountVectorizer | Stopword removal, ngram_range=(1,2), max_df=0.8 |
| 5b | c-TF-IDF | BM25-weighted keyword scoring across clusters |
| 6 | KeyBERTInspired | Semantic keyword refinement |

### Output per category

Each category produces a `final_df` with columns:

```
filename | category | sub_category_id | sub_category | confidence_score | preview
```

### Sub-categories discovered

| Category      | Sub-categories | Noise remapped |
|---------------|---------------|----------------|
| Business      | 24            | 70 → 7 new labels |
| Sport         | 6             | 24 → existing labels |
| Entertainment | 12            | 51 → 4 new labels |
| Politics      | 10            | 70 → existing labels |
| Tech          | 6             | 14 → existing labels |

All noise articles (HDBSCAN topic -1) were manually reviewed and remapped
to human-readable labels using a `noise_remap` dict and `consolidate_and_remap()`
function. No article is left as "Unassigned".

### Key lessons

- BGE-M3 with 8,192 token context handles full BBC articles without truncation
- UMAP outperforms PCA on dense transformer embeddings (preserves non-linear structure)
- HDBSCAN automatically determines cluster count — superior to KMeans/NMF for news
- HDBSCAN clustering sometimes groups by writing style rather than topic — fixed with
  post-clustering signal scoring and manual remapping per category


## Phase 2: Named Entity Recognition

**Notebook:** `NER_{category}.ipynb` × 5

spaCy `en_core_web_sm` pipeline extracts entities from every article.
A custom role classifier assigns job roles to each detected person.

### Entity types extracted

| Label  | Meaning                        | Example          |
|--------|--------------------------------|------------------|
| PERSON | People's names                 | Tony Blair       |
| ORG    | Organisations, companies       | Federal Reserve  |
| GPE    | Countries, cities              | United States    |
| MONEY  | Monetary values                | £1.13bn          |
| DATE   | Dates and time references      | December 2004    |
| NORP   | Nationalities, political groups| European         |

### Role classification

A keyword context-window classifier (`classify_role_*`) searches a 400-character
window around each detected name for job-title keywords.

Category-specific role sets:

- **Business:** CEO/Executive, Economist, Politician, Spokesperson, Lawyer
- **Sport:** Coach/Manager, Rugby Player, Tennis Player, Footballer, Athlete, Official, Pundit/Analyst
- **Entertainment:** Film Director, Writer/Author, Music Artist, TV/Film Actor, Visual Artist, Dancer/Performer
- **Politics:** Prime Minister, Chancellor, Government Minister, Party Leader, MP, Royal Family, Civil Servant
- **Tech:** Tech Executive, Software Developer, Tech Researcher, Game Developer, Blogger, Regulator/Legal

### Output per category

```
filename | sub_category | people | people_roles | organisations | locations | money_refs | dates
```

All list/dict columns stored as clean comma-separated strings for CSV compatibility.

### Filters applied

- `FALSE_PERSON_FILTER` — removes company names, nationalities, product names,
  fictional characters, and jargon spaCy incorrectly tags as PERSON
- `' ' in name` guard — requires full name (first + last), drops bare surnames
- `clean_entity_name()` — strips possessives (`'s`), trailing dashes, punctuation
- `org_names_lower` cross-filter — drops names that also appear as ORG in same article
- `clean_money` filter — retains only values containing £/$/ €/bn/million etc.

---

## Phase 3: Recommendation engine

**Notebook:** `recommendation.ipynb`

Keyword search across all five NER DataFrames, with displacy entity rendering.

### How it works

1. All five NER DataFrames are loaded and stacked into one `master_df` (2,119 rows)
2. User calls `recommend("keyword")` 
3. Each article is scored across entity columns with different weights:

| Column        | Weight | Rationale |
|---------------|--------|-----------|
| people        | 3      | Explicit spaCy PERSON tag |
| organisations | 3      | Explicit spaCy ORG tag |
| locations     | 2      | Explicit spaCy GPE tag |
| people_roles  | 1      | Role label match |
| preview text  | 1      | Raw text fallback |

4. Results sorted by score → confidence_score
5. Each result rendered with `displacy.render(doc, style="ent")` — full article text
   with all entities highlighted inline with colour-coded labels

### Usage

```python
recommend("Tony Blair")           # person search
recommend("Microsoft")            # organisation search
recommend("Iraq", top_n=10)       # location search
recommend("Coach/Manager")        # role type search
recommend("Paula Radcliffe", top_n=3)
```

## Tech stack

| Layer | Tools |
|-------|-------|
| Environment | Google Colab (GPU/CPU), Python 3.12, Google Drive |
| Data | pandas, numpy |
| Traditional NLP | scikit-learn (TF-IDF, NMF, CountVectorizer), gensim (coherence) |
| Embeddings | sentence-transformers (BAAI/bge-m3) |
| Dimensionality reduction | umap-learn |
| Clustering | hdbscan |
| Topic modelling | bertopic |
| NER | spaCy (en_core_web_sm), displacy |
| Visualisation | matplotlib, displacy |
| Version control | Git + GitHub |


## Repository structure

```
bbc-nlp/
├── loader.py                          # Shared data loader utility
├── bbc_nlp.ipynb                      # Phase 0 — exploration
├── business.ipynb                     # Phase 1 — Business sub-categories
├── sport.ipynb                        # Phase 1 — Sport sub-categories
├── entertainment.ipynb                # Phase 1 — Entertainment sub-categories
├── politics.ipynb                     # Phase 1 — Politics sub-categories
├── tech.ipynb                         # Phase 1 — Tech sub-categories
├── NER_Business.ipynb                 # Phase 2 — Business NER
├── NER_Sport.ipynb                    # Phase 2 — Sport NER
├── NER_Entertainment.ipynb            # Phase 2 — Entertainment NER
├── NER_Politics.ipynb                 # Phase 2 — Politics NER
├── NER_Tech.ipynb                     # Phase 2 — Tech NER
├── recommendation.ipynb               # Phase 3 — Recommendation engine
└── data/
    ├── bbc/                           # Raw .txt files
    ├── embeddings/                    # .npy BGE-M3 embeddings (per category)
    ├── final_*_Subcategories.csv      # Phase 1 output
    └── enriched_*_NER.csv             # Phase 2 output
```

## Author

**Fabunmi Richard**  
Data Engineer | AI Engineer | Web3 Developer  
Lagos, Nigeria  
GitHub: richyfabz | LinkedIn: linkedin.com/in/fabunmi-richard-a686ab23b