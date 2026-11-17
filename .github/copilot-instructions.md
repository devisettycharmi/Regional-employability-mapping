# GitHub Copilot Prompt: NLP-Driven Regional Employability Mapping of Engineering Students in Warangal

Paste this whole file into Copilot Chat (Agent mode) in an empty project folder. For best results, also save it as `.github/copilot-instructions.md`. Build step by step: after each phase, run the code, fix errors and show me the output before moving on.

---

## 1. Project goal

Build an end-to-end, NLP-driven system that:

1. Extracts skills from engineering students' free-text and listed data.
2. Matches students to job roles using semantic similarity.
3. Computes an explainable employability score and predicts placement likelihood.
4. Finds each student's skill gaps for their target role.
5. Maps employability regionally (by college and locality around Warangal, Telangana).
6. Serves everything through an interactive web app (Streamlit).

## 2. Tech stack

- Python 3.10+, pandas, numpy, scikit-learn, xgboost, shap
- spaCy (`en_core_web_sm`), NLTK, sentence-transformers (`all-MiniLM-L6-v2`), keybert, bertopic (optional)
- geopy (Nominatim), folium, plotly, matplotlib, seaborn
- Streamlit for the website, joblib for saving models, pyyaml for config
- Create `requirements.txt` with pinned versions.

## 3. Project structure

```
employability-mapping/
├── data/
│   ├── raw/                 # student_employability_dataset.csv, job_descriptions.csv
│   ├── processed/
│   └── geo/                 # geocode cache
├── src/
│   ├── config.py            # paths, constants, load config.yaml
│   ├── preprocessing.py     # cleaning + text normalisation
│   ├── skill_extraction.py  # taxonomy + PhraseMatcher + KeyBERT
│   ├── embeddings.py        # Sentence-BERT, caching
│   ├── matching.py          # student<->job similarity, top-N roles
│   ├── scoring.py           # employability score + skill gap
│   ├── clustering.py        # KMeans/BERTopic skill profiles
│   ├── model.py             # train/evaluate/save placement classifier + SHAP
│   ├── geo.py               # geocoding + regional aggregation
│   └── evaluation.py        # metrics, extraction accuracy check
├── notebooks/               # EDA.ipynb
├── models/                  # saved artefacts
├── app/
│   ├── app.py               # Streamlit entry
│   └── pages/               # 1_Student_View.py, 2_Regional_Dashboard.py, 3_Insights.py, 4_About.py
├── config.yaml              # includes `data_description:` text shown on the About page
├── requirements.txt
└── README.md
```

## 4. Input data (do not change files in `data/raw/`)

### 4.1 `student_employability_dataset.csv` columns

`Student ID, Full Name, College/Institution, College Locality, Home Locality, Branch/Department, Current Year, Current CGPA, Which technical skills do you currently have, Describe your technical skills in your own words, Describe your most important project, Do you have any internship experience, If yes, mention domain and technologies, Which job role are you interested in, Where would you prefer to work, Which skills do you feel you are currently lagging, Which skills do you want to learn in next 6-12 months, What is your placement status, How job ready do you consider yourself (1-10), num_projects, num_certifications, hackathons_participated, active_github, backlogs, communication_rating`

Notes: `If yes, mention domain and technologies` is empty for students without an internship. Placement status values: `Placed`, `Interviewing`, `Placement preparation`, `Not placed`. Years: `2nd Year`, `3rd Year`, `Final Year`.

### 4.2 `job_descriptions.csv` columns

`job_id, role, domain, job_description, required_skills (semicolon-separated), good_to_have_skills (semicolon-separated), experience_level`

Student `Which job role are you interested in` values match the `role` column exactly.

## 5. Phase-by-phase tasks

### Phase 1: Data understanding and cleaning (`preprocessing.py`, `notebooks/EDA.ipynb`)

- Load both CSVs, rename columns to snake_case short names (e.g., `skills_listed`, `skills_text`, `project`, `internship_detail`, `target_role`, `placement_status`).
- Check missing values, duplicates, dtypes, CGPA range (0-10), and report a data-quality summary.
- Encode: `has_internship` (0/1), `active_github` (0/1), `is_placed` (1 if `Placed`), `is_cs_branch`.
- Text normalisation function: lowercase, strip punctuation noise, unify variants ("ML" → "machine learning"), tokenisation, stop-word removal, lemmatisation (spaCy).
- EDA plots: placement status by year/college/branch, CGPA distribution, top skills, top target roles, preferred work location. Save figures to `reports/figures/`.

### Phase 2: Skill extraction (`skill_extraction.py`)

- Build a skills taxonomy from every skill in `required_skills` and `good_to_have_skills` of the job file, plus an alias dictionary (e.g., `ml`→`machine learning`, `dsa`→`data structures`, `js`→`javascript`).
- Use spaCy `PhraseMatcher` (case-insensitive) over: listed skills, free-text skills, project text and internship text. Handle short skills like `c` and `r` only from the comma-separated skills column to avoid false matches.
- Add KeyBERT keyphrase extraction as a secondary method for skills outside the taxonomy; store them in `extra_keyphrases`.
- Output per student: `skills` (list), `n_skills`, `skills_str`.
- In `evaluation.py`, sample 50 random students, export a CSV for manual verification, and compute precision/recall of the extractor against manually corrected labels once available.

### Phase 3: Embeddings and matching (`embeddings.py`, `matching.py`)

- Create a student profile text: skills + project + internship detail.
- Create a job profile text: role + job_description + required_skills + good_to_have_skills.
- Encode both with `all-MiniLM-L6-v2` (normalised embeddings), cache to `models/*.npy`.
- Baseline comparison: TF-IDF + cosine similarity. Report both and compare how often the student's chosen role appears in the top-3 matches (Top-1, Top-3 accuracy).
- For each student output `top3_roles`, `top3_scores`, and `match_score_target_role`.
- Also compute a hybrid score = 0.5 × embedding similarity + 0.5 × required-skill overlap (Jaccard or coverage) and make the weights configurable.

### Phase 4: Skill gap and employability score (`scoring.py`)

- `skill_gap` = required skills of target role − student skills; `role_coverage` = share of required skills the student has; also list matched good-to-have skills.
- Employability score (0-100), transparent weighted formula with weights in `config.yaml`:

- 35% role-skill match (hybrid score), 20% CGPA (scaled), 15% internship, 10% projects, 8% certifications, 5% hackathons, 4% GitHub, 3% communication, minus a penalty for backlogs.
- Add readiness bands: `Needs Improvement` (<40), `Developing` (40-60), `Job Ready` (60-80), `Highly Employable` (>80).
- Output a recommendation list: top 5 missing skills with a one-line learning suggestion.

### Phase 5: Clustering / topic modelling (`clustering.py`)

- KMeans on student embeddings; choose k from 3-10 using silhouette score; save the best.
- Label clusters automatically from their most frequent skills (e.g., "Web/Software", "Data & AI", "Core Electronics").
- 2-D visualisation with UMAP or PCA in Plotly. Optionally BERTopic on project text.

### Phase 6: Placement prediction (`model.py`)

- Target: `is_placed`. To avoid mixing students who have not yet reached placement season, train on **Final Year** students (also try all years with `year` as a feature) and compare.
- Features: cgpa, backlogs, has_internship, num_projects, num_certifications, hackathons_participated, active_github, communication_rating, n_skills, role_coverage, hybrid match score, branch, college, is_cs_branch.
- **Do NOT use** `placement_status`, `self_readiness`, or anything derived from the label as features (data leakage). Explain this in a code comment.
- Models: Logistic Regression, Random Forest, XGBoost. Use stratified 5-fold cross-validation, class weights, and report Accuracy, Precision, Recall, F1, ROC-AUC and a confusion matrix.
- SHAP summary and per-student explanation (waterfall) for the best model.
- Save the pipeline with joblib (`models/placement_model.joblib`).

### Phase 7: Regional mapping (`geo.py`)

- Geocode unique values of `college` + `college_locality` and `home_locality` (add ", Warangal, Telangana, India") with geopy Nominatim. Respect rate limits (1 request/sec), set a proper `user_agent`, and cache results to `data/geo/geocode_cache.csv`. Include a manual override dict for failed lookups.
- Aggregate by college and by locality: student count, mean employability score, mean CGPA, placement rate, top 5 missing skills.
- Folium map 1: colleges as circle markers sized by student count and coloured by mean employability. Folium map 2: home-locality heatmap or markers. Map 3: preferred work-location counts (bar/choropleth).
- Save maps as HTML in `reports/maps/`.

### Phase 8: Streamlit website (`app/`)

Pages:

1. **Student View**: choose an existing student (by ID/name) or enter skills manually/paste text. Show employability score gauge, readiness band, top-3 matched roles with scores, skill-gap list with learning suggestions, predicted placement probability and a SHAP explanation.
2. **Regional Dashboard**: the Folium maps (use `streamlit-folium`), filters for college/branch/year, KPI cards (students, avg score, placement rate), college comparison bar chart.
3. **Insights**: top missing skills overall and per college, skill-demand vs skill-supply chart (job-required skills vs students' skills), branch-wise employability, cluster visualisation, role popularity.
4. **About**: project description, methodology diagram, metrics, limitations, and a **Data** section that displays the `data_description` text read from `config.yaml` (do not hard-code any claim about how the data was obtained).

- Use `st.cache_data` / `st.cache_resource` so models and embeddings load once. Clean, consistent theme and sidebar navigation.

### Phase 9: Evaluation, docs and packaging

- `evaluation.py` produces a single metrics report (JSON + markdown): extraction check, Top-k matching accuracy (embeddings vs TF-IDF), clustering silhouette, classifier CV metrics, score vs placement correlation.
- README: setup, how to run each phase (`python -m src.pipeline`), how to launch (`streamlit run app/app.py`), architecture diagram (Mermaid), screenshots placeholder, limitations and future work.
- Add a `src/pipeline.py` that runs all phases in order, and a `Makefile` or simple script with `run_pipeline` and `run_app` targets.
- Add basic unit tests (pytest) for the skill extractor, score function and gap computation.

## 6. Rules and quality requirements

- Reproducibility: set `random_state=42` everywhere; read all paths and weights from `config.py` / `config.yaml`; never hard-code absolute paths.
- Write modular, typed, documented functions (docstrings + type hints); handle errors and missing files gracefully with clear messages.
- Keep the code data-agnostic: the pipeline must work if I replace the two CSVs in `data/raw/` with different files that have the same columns.
- Do not invent or hard-code any statements about how the data was collected. Anything about data origin must come from `data_description` in `config.yaml`, which I will fill in myself.
- Print progress logs; show the key metrics after each phase.
- Do not change the raw data files. Write all generated files to `data/processed/`, `models/`, `reports/`.

## 7. Build order

Start with the folder structure, `requirements.txt`, `config.py` and `config.yaml`. Then Phase 1, and stop to show the data-quality report. Continue phase by phase, running and testing each module before moving on. Finish with the Streamlit app and the README.
