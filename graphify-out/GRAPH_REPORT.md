# Graph Report - hinglish_thesis  (2026-11-18)

## Corpus Check
- Large corpus: 51 files · ~988,168 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 339 nodes · 637 edges · 17 communities (15 shown, 2 thin omitted)
- Extraction: 94% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 34 edges (avg confidence: 0.84)
- Token cost: 284,774 input · 0 output

## Community Hubs (Navigation)
- Devanagari Transliteration (D)
- NEG Control and Review Workbook
- Analysis and Statistics
- Training Flips and Tests
- Dataset Build
- Output Files and Hashes
- Excel Export and Xlit Merge
- Flip Metrics Concepts
- Regression Odds Ratios Chart
- Flip Rates Chart
- NEG vs Typos Chart
- Tokenizer Statistics
- Transliteration Environment
- Training Environment and Revisions
- Complete Tweet Subgroup Chart
- Pandas Pin (xlit env)
- Urduhack Pin

## God Nodes (most connected - your core abstractions)
1. `main()` - 14 edges
2. `Transliterator` - 14 edges
3. `Regression Odds Ratios Forest Plot` - 13 edges
4. `main()` - 11 edges
5. `run()` - 10 edges
6. `main()` - 9 edges
7. `variants()` - 9 edges
8. `main()` - 9 edges
9. `main()` - 9 edges
10. `load_data()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `output/HASHES.txt manifest` --semantically_similar_to--> `Root SHA-256 hash manifest of output/`  [INFERRED] [semantically similar]
  output/HASHES.txt → HASHES.txt
- `REVISIONS (pinned HF commits in train_eval.py)` --semantically_similar_to--> `Pinned Hugging Face model revisions`  [INFERRED] [semantically similar]
  scripts_train/README_train.md → results/tokenizer_versions.txt
- `load()` --calls--> `preds_path()`  [INFERRED]
  analysis/analyze.py → scripts_train/train_eval.py
- `a_performance()` --calls--> `macro_f1()`  [INFERRED]
  analysis/analyze.py → scripts_train/train_eval.py
- `flips_long.csv` --conceptually_related_to--> `Flip rate (variant vs A, same model and seed)`  [INFERRED]
  scripts_train/README_train.md → analysis/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Perturbation variant construction (typos, transliteration, negation)** — build_variants, transliterate_devanagari, logs_build_neg_control_neg_control_set, hashes_variants_d_csv, logs_build_variants_nested_typo_levels [INFERRED 0.85]
- **Colab training and flip computation flow** — scripts_train_run_all, scripts_train_train_eval, scripts_train_readme_train_preds_csv, scripts_train_make_flips, scripts_train_readme_train_flips_long_csv [EXTRACTED 1.00]
- **Flip robustness statistical analysis** — analysis_readme_flip_rate, analysis_readme_seed_disagreement, analysis_readme_cluster_bootstrap, analysis_readme_mcnemar_holm, analysis_readme_gee_logistic_regression [EXTRACTED 1.00]
- **Three multilingual encoders compared on prediction flip robustness** — analysis_flip_rates_mbert, analysis_flip_rates_xlm_roberta, analysis_flip_rates_muril, analysis_flip_rates_flip_rate_vs_a, analysis_flip_rates_strict_flip_rate [EXTRACTED 1.00]
- **Flip rates judged against seed noise with bootstrap CIs** — analysis_flip_rates_flip_rate_vs_a, analysis_flip_rates_seed_disagreement_baseline, analysis_flip_rates_cluster_bootstrap_ci [INFERRED 0.85]
- **Multilingual encoders compared on NEG vs typo robustness** — analysis_neg_vs_typos_mbert, analysis_neg_vs_typos_xlm_roberta, analysis_neg_vs_typos_muril, analysis_neg_vs_typos_chart [EXTRACTED 1.00]
- **Perturbation conditions compared in regression** — analysis_regression_odds_ratios_a_typos, analysis_regression_odds_ratios_b_typos_case_emoji, analysis_regression_odds_ratios_c_d_only [EXTRACTED 1.00]
- **Perturbation magnitude/tokenization predictors with OR>1** — analysis_regression_odds_ratios_delta_fertility, analysis_regression_odds_ratios_n_changed_words, analysis_regression_odds_ratios_char_edit_dist_ci [INFERRED 0.85]
- **Three encoders compared on the complete-tweet subgroup** — analysis_subgroups_complete_tweet_mbert, analysis_subgroups_complete_tweet_xlmr, analysis_subgroups_complete_tweet_muril, analysis_subgroups_complete_tweet_complete_tweet_subgroup [EXTRACTED 1.00]

## Communities (17 total, 2 thin omitted)

### Community 0 - "Devanagari Transliteration (D)"
Cohesion: 0.06
Nodes (22): output/variants_D.csv, Variant build checks (1,000 rows x 8 variants), caps_share(), context_rule(), is_caps(), is_letter_spelling(), is_protected(), lang() (+14 more)

### Community 1 - "NEG Control and Review Workbook"
Cohesion: 0.07
Nodes (17): main(), neg_op(), log(), main(), rich_diff(), write_sheet(), check(), editable() (+9 more)

### Community 2 - "Analysis and Statistics"
Cohesion: 0.13
Nodes (23): a_performance(), bar_panel(), cluster_boot(), figures(), fit_gee(), flip_rates(), fmt(), forest() (+15 more)

### Community 3 - "Training Flips and Tests"
Cohesion: 0.11
Nodes (20): flips(), main(), preds/{model}_{seed}.csv, test_flips(), test_leakage_fires(), test_leakage_fires_in_load_data(), test_macro_f1(), test_pred_frame() (+12 more)

### Community 4 - "Dataset Build"
Cohesion: 0.13
Nodes (12): dedup_key(), glue_underscores(), grams(), handle_end(), hin_share(), main(), near_duplicates(), parse() (+4 more)

### Community 5 - "Output Files and Hashes"
Cohesion: 0.09
Nodes (21): Typo variants T1/T2/T3, Variant NEG (negation control, 352 sentences), output/base_1000.csv, Root SHA-256 hash manifest of output/, output/neg_control.csv, output/train.csv, output/val.csv, output/variants_D_before_rules.csv (identical hash to variants_D_old.csv) (+13 more)

### Community 6 - "Excel Export and Xlit Merge"
Cohesion: 0.13
Nodes (8): main(), write(), bad_value(), main(), read_review(), openpyxl>=3.1, main(), val_path()

### Community 7 - "Flip Metrics Concepts"
Cohesion: 0.10
Nodes (17): Analysis of the 9 fine-tuning runs, bert-base-multilingual-cased, Case/emoji variants LC, UC, NE, complete_tweet flag (complete vs truncated tweets), delta_fertility (tokenizer fertility change), Flip rate (variant vs A, same model and seed), margin_A (prediction margin on A), google/muril-base-cased (+9 more)

### Community 8 - "Regression Odds Ratios Chart"
Cohesion: 0.19
Nodes (14): Condition a_typos, Condition b_typos_case_emoji, Condition c_D_only, char_edit_dist_ci, Regression Odds Ratios Forest Plot, complete_tweet, delta_fertility, margin_A (+6 more)

### Community 9 - "Flip Rates Chart"
Cohesion: 0.27
Nodes (11): Flip Rates Figure (flip rate vs A and strict flip rate by condition and model), 95% CI via Cluster Bootstrap (sentence ids, 3 seeds pooled, 2,000 resamples), Condition D, Condition NE, Perturbation Conditions T1, T2, T3, LC, NE, UC, D, NEG, Flip Rate vs A, bert-base-multilingual-cased, muril-base-cased (+3 more)

### Community 10 - "NEG vs Typos Chart"
Cohesion: 0.25
Nodes (11): NEG minus Typo Flip-Rate Difference Chart, NEG minus typo, flip (352 ids), Flip Rate, bert-base-multilingual-cased, muril-base-cased, Negation (NEG) Perturbation, NEG minus typo, strict flip (352 ids), Strict Flip Rate (+3 more)

### Community 11 - "Tokenizer Statistics"
Cohesion: 0.31
Nodes (5): levenshtein(), load(), main(), text_stats(), tok_stats()

### Community 12 - "Transliteration Environment"
Cohesion: 0.29
Nodes (7): IndicXlit (top-1 transliteration), Transliteration dictionary (59.32% of D tokens), xlit_dictionary_proposed.tsv (not merged), ai4bharat-transliteration==1.1.3, fairseq==0.12.2, numpy==1.26.4, torch==2.5.1

### Community 13 - "Training Environment and Revisions"
Cohesion: 0.33
Nodes (7): Colab training environment (Python 3.13, torch 2.11 cu130, Tesla T4), Pinned Hugging Face model revisions, Tokenizer environment versions (transformers 4.57.6, tokenizers 0.22.2), Training on Colab (3 models x 3 seeds), tokenizers==0.22.2, transformers==4.57.6, REVISIONS (pinned HF commits in train_eval.py)

### Community 14 - "Complete Tweet Subgroup Chart"
Cohesion: 0.47
Nodes (6): Categories T1, T2, T3, LC, NE, UC, D, NEG, Subgroup Chart: Complete Tweet (yes/no) by Category and Model, Complete Tweet Subgroup (yes vs no), bert-base-multilingual-cased, muril-base-cased, xlm-roberta-base

## Ambiguous Edges - Review These
- `Perturbation Conditions T1, T2, T3, LC, NE, UC, D, NEG` → `Condition D`  [AMBIGUOUS]
  analysis/flip_rates.png · relation: conceptually_related_to
- `Typo Perturbation` → `Perturbation Tiers T1-T3`  [AMBIGUOUS]
  analysis/neg_vs_typos.png · relation: conceptually_related_to

## Knowledge Gaps
- **38 isolated node(s):** `output/variants_D_before_rules.csv (identical hash to variants_D_old.csv)`, `output/variants_long.csv`, `output/xlit_candidates.json`, `Strict flip rate (only rows with A correct)`, `margin_A (prediction margin on A)` (+33 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 126 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Flip rate (variant vs A, same model and seed)` connect `Flip Metrics Concepts` to `Analysis and Statistics`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **What connects `output/variants_D_before_rules.csv (identical hash to variants_D_old.csv)`, `output/variants_long.csv`, `output/xlit_candidates.json` to the rest of the system?**
  _38 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Devanagari Transliteration (D)` be split into smaller, more focused modules?**
  _Cohesion score 0.059932659932659935 - nodes in this community are weakly interconnected._
- **Why does `Transliterator` connect `Devanagari Transliteration (D)` to `NEG Control and Review Workbook`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Should `NEG Control and Review Workbook` be split into smaller, more focused modules?**
  _Cohesion score 0.06509803921568627 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Perturbation Conditions T1, T2, T3, LC, NE, UC, D, NEG` and `Condition D`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Typo Perturbation` and `Perturbation Tiers T1-T3`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._