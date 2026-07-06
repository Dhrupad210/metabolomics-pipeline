# MTBLS13136 — Paroxysmal vs Persistent AF Metabolomics Analysis

## Study
Targeted UPLC-MS/MS plasma metabolomics, 100 patients (65 persistent AF, 35 paroxysmal AF),
two centers in China. Original paper reported 8 differential metabolites and an XGBoost
model (AUC 0.751 discovery / 0.985 validation).

## Data pipeline
1. **Sample sheet parsing** (`s_MTBLS13136.txt`) — mapped 100 samples to disease labels.
   Confirmed "FSY"/"LNK" filename tags are just sample-type codes, not QC/blank groups
   (`sample_type` column was constant/uninformative across all rows).
2. **Metabolite matrix** — loaded MAF quant file (`m_..._maf.tsv`), 197 metabolites
   already identified/quantified by the original authors (targeted panel, not
   untargeted feature detection — no raw mzML processing needed).
3. **Validation against paper**: reproduced all 8 reported metabolites via Mann-Whitney U
   test (all p<0.005), confirming matrix integrity. One metabolite had a typo in the
   source data (`N-Acetyaspartic acid`, missing an "l").
4. **Run Via**: mtbls13136_pipeline.py

## Differential analysis
Mann-Whitney U + Benjamini-Hochberg FDR across 197 metabolites.

**10 metabolites significant at FDR<0.05**, all elevated in persistent AF:

| Metabolite | FDR | log2FC |
|---|---|---|
| GCDCA-3S (bile acid conjugate) | 0.0028 | 1.27 |
| Isocitric acid | 0.0028 | 0.36 |
| Malic acid | 0.0029 | 0.38 |
| Oxoglutaric acid | 0.0029 | 0.28 |
| Citramalic acid | 0.0268 | 0.73 |
| N-Acetylaspartic acid | 0.0276 | 0.34 |
| Adipic acid | 0.0298 | 0.80 |
| Glyceric acid | 0.0298 | 0.16 |
| Kynurenine | 0.0298 | 0.40 |
| Phenylalanine | 0.0298 | 0.16 |

**GCDCA-3S** (sulfated glycochenodeoxycholic acid) is a novel hit — highest-ranked
signal by both p-value and Random Forest importance, not reported in the original paper.

## Unsupervised structure: PCA
PC1+PC2 explain only ~29% of variance; no clean visual separation between groups.
Consistent with a subtle, non-dominant biological signal rather than a strong
disease axis.

## OPLS-DA — failed permutation test
Initial OPLS-DA showed apparent separation (persistent mean t-score +1.84 vs
paroxysmal −3.41). **Permutation testing (200 shuffles) showed this is not
significant** (p=0.925 — permuted labels scored *higher* on average than real
labels). With 197 features and 100 samples, OPLS-DA overfits and finds a
separating axis for almost any label assignment. This result should not be
reported as evidence of group separation, and casts some doubt on similarly
unvalidated supervised multivariate results in the literature (e.g. the
original paper's very high validation-cohort AUC of 0.985).

## ML models — validated with permutation testing
5-fold stratified CV + 100-permutation significance test per model:

| Model | Real AUC | Permuted mean AUC | p-value |
|---|---|---|---|
| Random Forest | 0.763 | 0.516 | 0.0099 |
| LightGBM | 0.719 | 0.506 | 0.0198 |
| Logistic Regression | 0.670 | 0.497 | 0.0198 |
| XGBoost | 0.686 | 0.502 | 0.0396 |

All four models beat chance significantly. Random Forest performed best and is
roughly consistent with the paper's own discovery-cohort AUC (0.751).

## Feature importance (Random Forest) vs univariate FDR results
All 10 FDR-significant metabolites rank in RF's top 20 features; the top 8 by
RF importance are identical to the top 8 by FDR. Independent convergence
between univariate and multivariate methods on the same core signal.

Several additional top-20 RF features that don't individually pass FDR
(succinic acid, 2-hydroxyglutaric acid, creatine, azelaic acid) are
biologically consistent with the same theme — TCA cycle / citrate cycle /
glyoxylate-dicarboxylate metabolism — matching the pathway enrichment the
original paper reports.

## Effect sizes
Added Cliff's delta and Cohen's d to the 10 FDR-significant metabolites. All show
medium-to-large effect (Cliff's delta 0.39–0.51), confirming these aren't just
statistically significant with trivial magnitude. One inconsistency: GCDCA-3S has
large Cliff's delta (0.51) but small Cohen's d (0.18) — this divergence indicates a
skewed/outlier-influenced distribution (typical for bile acid abundances), so the
rank-based Cliff's delta is the more reliable estimate for this metabolite.

## Robustness checks: imputation and scaling
- **Imputation (median vs KNN, k=5):** no difference in any downstream result.
  Missingness in this dataset (already filtered to <30% per metabolite) is not a
  meaningful factor.
- **Scaling, Random Forest:** invariant by construction — tree splits depend only
  on value ordering, so standard/Pareto/Vast/no scaling all gave identical AUC
  (0.763). This confirms the RF result is not an artifact of scaling choice.
- **Scaling, Logistic Regression:** scaling matters here, as expected for a linear
  model. Vast scaling performed best (AUC 0.727) vs standard scaling (0.673),
  pareto (0.684), and no scaling (0.681). Vast scaling's stronger downweighting of
  high-variance features likely helps in this small-sample/many-correlated-feature
  regime.

## Feature-selection leakage check
Confirmed no leakage: FDR filtering and ML modeling were run independently on the
full 197-metabolite matrix in every case. FDR results were never used to select
which features went into the RF/LogReg/XGBoost/LightGBM models — the two analyses
were only compared after the fact. Reported CV-AUCs are not inflated by this
mechanism.

## Pathway enrichment — not completed in this pass
Metabolite name lists (10 significant + 197 background) were exported for manual
KEGG/MetaboAnalyst lookup. Flagged issue: several significant/background hits are
sulfated or conjugated bile acids (GCDCA-3S, GDCA-3S, LCA-3S, GLCA-3S, TCDCA, TCA,
CA, CDCA, DCA, UDCA, GUDCA, bUDCA, NorCA) that are unlikely to resolve cleanly
against KEGG's default compound library and may need manual HMDB cross-referencing.

## Pathway enrichment — completed via MetaboAnalyst
KEGG-based metabolite set enrichment on the 10 FDR-significant metabolites (197
as background). Two pathways significant after FDR correction:

| Pathway | Hits/Total | FDR |
|---|---|---|
| Citrate cycle (TCA cycle) | 3/20 | 0.0031 |
| Glyoxylate and dicarboxylate metabolism | 3/32 | 0.0066 |

All other candidate pathways had only 1 hit each and FDR>0.4 (not significant —
single-metabolite coincidental membership). Isocitric acid, malic acid, and
oxoglutaric acid are the shared members driving both hits; these three were also
the top FDR-significant and top RF-importance metabolites, giving three
independent lines of convergent evidence. This result matches the pathway
enrichment reported in the original paper.

## Bottom line
- Core finding replicated: TCA-cycle-related organic acids + a few amino
  acid derivatives are robustly elevated in persistent vs paroxysmal AF.
- One credible novel candidate: **GCDCA-3S**, a sulfated bile acid conjugate.
- Genuine (permutation-validated) predictive signal exists (RF AUC 0.76),
  though weaker than the paper's headline validation-cohort number, which
  likely reflects some overfitting in the original small-cohort ML pipeline.
- OPLS-DA, as commonly used in metabolomics papers without permutation
  testing, would have produced a misleading "clear separation" result here —
  worth flagging if reviewing similar literature.
- No feature-selection leakage in the ML pipeline; RF result is scaling-invariant
  and imputation-invariant, so the 0.763 AUC is a stable estimate.
- Four independent lines of evidence now converge on TCA cycle / glyoxylate-
  dicarboxylate metabolism as the core disrupted pathway in persistent AF:
  univariate FDR significance, effect size magnitude, RF feature importance,
  and KEGG pathway enrichment.
- Remaining gaps before this would be publication-ready: external cohort
  validation, and metabolite annotation confidence for GCDCA-3S beyond the
  vendor-provided targeted panel (bile acid conjugates didn't resolve in KEGG
  enrichment and would need dedicated bile-acid pathway analysis, e.g. via
  SMPDB or manual literature search).
