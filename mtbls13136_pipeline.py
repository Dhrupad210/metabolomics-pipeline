"""
MTBLS13136 — Paroxysmal vs Persistent AF Metabolomics Pipeline
Consolidated end-to-end script. Run from the metabolomics-copilot directory
containing: s_MTBLS13136.txt, a_MTBLS13136_LC-MS_..._profiling.txt,
m_MTBLS13136_LC-MS_..._maf.tsv

Requires: pandas numpy scipy statsmodels scikit-learn xgboost lightgbm pyopls
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pyopls import OPLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# STEP 1: Parse sample sheet, build filename -> disease/group mapping
# ---------------------------------------------------------------------------
def build_sample_groups(sample_sheet="s_MTBLS13136.txt", mzml_dir="."):
    df = pd.read_csv(sample_sheet, sep="\t")
    df = df.rename(columns={
        "Source Name": "source_name",
        "Sample Name": "sample_name",
        "Factor Value[Disease]": "disease",
    })

    mzml_files = sorted(Path(mzml_dir).glob("*.mzML"))

    def classify(fname):
        stem = fname.stem
        if "_FSY_" in stem:
            return "FSY"
        if "_LNK_" in stem:
            return "LNK"
        return "sample"

    records = []
    for f in mzml_files:
        stem = f.stem
        group = classify(f)
        match = df[df["sample_name"].apply(lambda s: str(s) in stem)]
        disease = match["disease"].values[0] if len(match) else None
        records.append({"file": f.name, "group": group, "disease": disease})

    out = pd.DataFrame(records)
    out.to_csv("sample_groups.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# STEP 2: Load MAF (metabolite assignment file) quant table, merge with groups
# ---------------------------------------------------------------------------
def build_metabolite_matrix(
    maf_file="m_MTBLS13136_LC-MS_alternating_reverse-phase_metabolite_profiling_v2_maf.tsv",
    sample_sheet="s_MTBLS13136.txt",
):
    maf = pd.read_csv(maf_file, sep="\t")

    meta_cols = [
        "database_identifier", "chemical_formula", "smiles", "inchi",
        "metabolite_identification", "mass_to_charge", "fragmentation",
        "modifications", "charge", "retention_time", "taxid", "species",
        "database", "database_version", "reliability", "uri", "search_engine",
        "search_engine_score", "smallmolecule_abundance_sub",
        "smallmolecule_abundance_stdev_sub", "smallmolecule_abundance_std_error_sub",
    ]
    sample_cols = [c for c in maf.columns if c not in meta_cols]

    mat = maf.set_index("metabolite_identification")[sample_cols].T
    mat.index.name = "sample_name"
    mat = mat.reset_index()

    # disease labels straight from the sample sheet (robust to filename regex quirks)
    sdf = pd.read_csv(sample_sheet, sep="\t")
    sdf = sdf.rename(columns={"Sample Name": "sample_name",
                               "Factor Value[Disease]": "disease"})
    disease_map = sdf.set_index("sample_name")["disease"].to_dict()

    merged = mat.copy()
    merged["disease"] = merged["sample_name"].map(disease_map)

    merged.to_csv("metabolite_matrix.csv", index=False)
    return merged


# ---------------------------------------------------------------------------
# STEP 3: Univariate differential analysis (Mann-Whitney U + BH-FDR)
# ---------------------------------------------------------------------------
def differential_analysis(merged, meta_cols=("sample_name", "disease", "group")):
    metab_cols = [c for c in merged.columns if c not in meta_cols]

    results = []
    for m in metab_cols:
        per = pd.to_numeric(merged[merged["disease"] == "persistent"][m], errors="coerce").dropna()
        par = pd.to_numeric(merged[merged["disease"] == "paroxysmal"][m], errors="coerce").dropna()
        if len(per) < 3 or len(par) < 3:
            continue
        stat, p = stats.mannwhitneyu(per, par, alternative="two-sided")
        log2fc = np.log2((per.median() + 1e-6) / (par.median() + 1e-6))
        results.append({
            "metabolite": m, "p_value": p, "log2FC": log2fc,
            "median_persistent": per.median(), "median_paroxysmal": par.median(),
        })

    res_df = pd.DataFrame(results)
    res_df["fdr"] = multipletests(res_df["p_value"], method="fdr_bh")[1]
    res_df["neglog10p"] = -np.log10(res_df["p_value"])
    res_df["sig"] = res_df["fdr"] < 0.05
    res_df = res_df.sort_values("p_value")
    res_df.to_csv("differential_results.csv", index=False)
    return res_df


# ---------------------------------------------------------------------------
# Helper: build log2-scaled feature matrix (median-imputed, <30% missing kept)
# ---------------------------------------------------------------------------
def build_feature_matrix(merged, meta_cols=("sample_name", "disease", "group"),
                          max_missing_frac=0.3):
    metab_cols = [c for c in merged.columns if c not in meta_cols]
    X = merged[metab_cols].apply(pd.to_numeric, errors="coerce")
    miss_frac = X.isna().mean()
    keep_cols = miss_frac[miss_frac < max_missing_frac].index
    X = X[keep_cols].fillna(X[keep_cols].median())
    X_log = np.log2(X.values + 1)
    y = (merged["disease"] == "persistent").astype(int).values
    return X_log, y, list(keep_cols)


# ---------------------------------------------------------------------------
# STEP 4: PCA (unsupervised)
# ---------------------------------------------------------------------------
def run_pca(merged, n_components=5):
    X_log, y, keep_cols = build_feature_matrix(merged)
    Xs = StandardScaler().fit_transform(X_log)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"])
    pca_df["disease"] = merged["disease"].values
    pca_df["sample_name"] = merged["sample_name"].values
    pca_df.to_csv("pca_scores.csv", index=False)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_[:5])
    return pca_df, pca


# ---------------------------------------------------------------------------
# STEP 5: OPLS-DA with permutation test (supervised — DO NOT TRUST without this)
# ---------------------------------------------------------------------------
def run_opls_da(merged, n_perm=200, random_state=42):
    X_log, y, keep_cols = build_feature_matrix(merged)
    Xs = StandardScaler().fit_transform(X_log)

    opls = OPLS(n_components=2)
    Z = opls.fit_transform(Xs, y)
    pls = PLSRegression(n_components=1)
    scores = pls.fit_transform(Z, y)[0]

    opls_df = pd.DataFrame({
        "t_pred": scores.ravel(),
        "sample_name": merged["sample_name"].values,
        "disease": merged["disease"].values,
    })
    opls_df.to_csv("opls_scores.csv", index=False)

    real_score = np.corrcoef(scores.ravel(), y)[0, 1] ** 2
    rng = np.random.default_rng(random_state)
    perm_scores = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        opls_p = OPLS(n_components=2)
        Z_p = opls_p.fit_transform(Xs, y_perm)
        pls_p = PLSRegression(n_components=1)
        s_p = pls_p.fit_transform(Z_p, y_perm)[0]
        perm_scores.append(np.corrcoef(s_p.ravel(), y_perm)[0, 1] ** 2)
    perm_scores = np.array(perm_scores)
    p_perm = (np.sum(perm_scores >= real_score) + 1) / (n_perm + 1)

    print(f"OPLS-DA real R2={real_score:.4f}, perm mean={perm_scores.mean():.4f}, "
          f"perm p-value={p_perm:.4f}")
    if p_perm >= 0.05:
        print("WARNING: OPLS-DA separation is NOT significant vs permuted labels. "
              "Do not report this as a real group difference.")
    return opls_df, real_score, p_perm


# ---------------------------------------------------------------------------
# STEP 6: ML models with 5-fold CV + permutation testing
# ---------------------------------------------------------------------------
def run_ml_models(merged, n_perm=100, random_state=42):
    X_log, y, keep_cols = build_feature_matrix(merged)

    models = {
        "LogReg": Pipeline([("scale", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000, C=0.1))]),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=4,
                                                random_state=random_state),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  eval_metric="logloss", random_state=random_state),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    random_state=random_state, verbosity=-1),
    }

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("=== Real labels: 5-fold CV AUC ===")
    real_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_log, y, cv=outer_cv, scoring="roc_auc")
        real_results[name] = scores.mean()
        print(f"{name}: {scores.mean():.3f} +/- {scores.std():.3f}")

    print(f"\n=== Permutation test ({n_perm} shuffles) ===")
    rng = np.random.default_rng(random_state)
    perm_results = {}
    for name, model in models.items():
        perm_aucs = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            scores = cross_val_score(model, X_log, y_perm, cv=outer_cv, scoring="roc_auc")
            perm_aucs.append(scores.mean())
        perm_aucs = np.array(perm_aucs)
        real = real_results[name]
        p_val = (np.sum(perm_aucs >= real) + 1) / (n_perm + 1)
        perm_results[name] = {"real_auc": real, "perm_mean": perm_aucs.mean(), "p_value": p_val}
        print(f"{name}: real={real:.3f}, perm_mean={perm_aucs.mean():.3f}, p={p_val:.4f}")

    return real_results, perm_results, models, X_log, y, keep_cols


# ---------------------------------------------------------------------------
# STEP 7: Effect sizes (Cliff's delta + Cohen's d) for FDR-significant hits
# ---------------------------------------------------------------------------
def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    diff_matrix = x[:, None] - y[None, :]
    greater = np.sum(diff_matrix > 0)
    less = np.sum(diff_matrix < 0)
    return (greater - less) / (n_x * n_y)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else np.nan


def interpret_cliffs(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def add_effect_sizes(merged, diff_results):
    effect_sizes = []
    for m in diff_results["metabolite"]:
        per = pd.to_numeric(merged[merged["disease"] == "persistent"][m], errors="coerce").dropna()
        par = pd.to_numeric(merged[merged["disease"] == "paroxysmal"][m], errors="coerce").dropna()
        if len(per) < 3 or len(par) < 3:
            continue
        d = cliffs_delta(per.values, par.values)
        cd = cohens_d(per.values, par.values)
        effect_sizes.append({"metabolite": m, "cliffs_delta": d, "cohens_d": cd})

    es_df = pd.DataFrame(effect_sizes)
    diff_full = diff_results.merge(es_df, on="metabolite").sort_values("p_value")
    diff_full["effect_magnitude"] = diff_full["cliffs_delta"].apply(interpret_cliffs)
    diff_full.to_csv("differential_results.csv", index=False)
    return diff_full


# ---------------------------------------------------------------------------
# STEP 8: Robustness check — imputation (median/KNN) x scaling (standard/
# pareto/vast/none), tested with both RF (scaling-invariant) and LogReg
# (scaling-sensitive) to show the difference.
# ---------------------------------------------------------------------------
def pareto_scale(X):
    return (X - X.mean()) / np.sqrt(X.std())


def vast_scale(X):
    return ((X - X.mean()) / X.std()) * (X.mean() / X.std())


def robustness_check(merged, meta_cols=("sample_name", "disease", "group"),
                      random_state=42):
    metab_cols = [c for c in merged.columns if c not in meta_cols]
    X_raw = merged[metab_cols].apply(pd.to_numeric, errors="coerce")
    miss_frac = X_raw.isna().mean()
    keep_cols = miss_frac[miss_frac < 0.3].index
    X_raw = X_raw[keep_cols]
    y = (merged["disease"] == "persistent").astype(int).values

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=4, random_state=random_state),
        "LogReg": LogisticRegression(max_iter=5000, C=0.1),
    }

    imputations = {
        "median": X_raw.fillna(X_raw.median()),
        "knn": pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X_raw), columns=X_raw.columns),
    }

    results = []
    for clf_name, clf in classifiers.items():
        for imp_name, X_imp in imputations.items():
            X_log = np.log2(X_imp.values + 1)
            X_log_df = pd.DataFrame(X_log, columns=X_imp.columns)

            scalings = {
                "standard": StandardScaler().fit_transform(X_log_df),
                "pareto": pareto_scale(X_log_df).values,
                "vast": vast_scale(X_log_df).values,
                "none": X_log_df.values,
            }

            for scale_name, X_scaled in scalings.items():
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                scores = cross_val_score(clf, X_scaled, y, cv=outer_cv, scoring="roc_auc")
                results.append({
                    "classifier": clf_name, "imputation": imp_name, "scaling": scale_name,
                    "mean_auc": scores.mean(), "std_auc": scores.std(),
                })

    res_df = pd.DataFrame(results).sort_values(["classifier", "mean_auc"], ascending=[True, False])
    res_df.to_csv("robustness_check.csv", index=False)
    print(res_df.to_string(index=False))
    return res_df


# ---------------------------------------------------------------------------
# STEP 9: Random Forest feature importance vs FDR-significant metabolites
# ---------------------------------------------------------------------------
def rf_feature_importance(merged, differential_results, random_state=42):
    X_log, y, keep_cols = build_feature_matrix(merged)

    rf = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=random_state)
    rf.fit(X_log, y)

    importances = pd.DataFrame({
        "metabolite": keep_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    fdr_sig_set = set(differential_results[differential_results["fdr"] < 0.05]["metabolite"])
    importances["fdr_significant"] = importances["metabolite"].isin(fdr_sig_set)
    importances.to_csv("rf_feature_importance.csv", index=False)

    print(importances.head(20).to_string(index=False))
    print("\nOf top 20 RF features, FDR-significant count:",
          importances.head(20)["fdr_significant"].sum())
    return importances, rf


# ---------------------------------------------------------------------------
# STEP 10: Export metabolite name lists for pathway enrichment
# (upload to MetaboAnalyst: significant list as query, all list as background)
# ---------------------------------------------------------------------------
def export_metabolite_lists(diff_results):
    diff_results["metabolite"].to_csv("all_metabolites.txt", index=False, header=False)
    sig = diff_results[diff_results["fdr"] < 0.05]["metabolite"]
    sig.to_csv("significant_metabolites.txt", index=False, header=False)
    print(f"Exported {len(diff_results)} background metabolites -> all_metabolites.txt")
    print(f"Exported {len(sig)} significant metabolites -> significant_metabolites.txt")
    print("NOTE: sulfated/conjugated bile acids (GCDCA-3S, GDCA-3S, LCA-3S, GLCA-3S,")
    print("TCDCA, TCA, CA, CDCA, DCA, UDCA, GUDCA, bUDCA, NorCA) may not resolve in")
    print("KEGG's default compound library — cross-check against HMDB/SMPDB manually.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(">> Step 1: sample groups")
    groups = build_sample_groups()

    print("\n>> Step 2: metabolite matrix")
    merged = build_metabolite_matrix()
    print(merged["disease"].value_counts(dropna=False))

    print("\n>> Step 3: differential analysis (Mann-Whitney + FDR)")
    diff_results = differential_analysis(merged)
    print(f"Significant (FDR<0.05): {diff_results['sig'].sum()}")
    print(diff_results.head(15).to_string(index=False))

    print("\n>> Step 4: PCA")
    pca_df, pca_model = run_pca(merged)

    print("\n>> Step 5: OPLS-DA + permutation test")
    opls_df, opls_r2, opls_p = run_opls_da(merged)

    print("\n>> Step 6: ML models (LogReg/RF/XGBoost/LightGBM) + permutation test")
    real_results, perm_results, models, X_log, y, keep_cols = run_ml_models(merged)

    print("\n>> Step 7: Effect sizes (Cliff's delta + Cohen's d)")
    diff_results = add_effect_sizes(merged, diff_results)
    print(diff_results[diff_results["fdr"] < 0.05][
        ["metabolite", "fdr", "cliffs_delta", "cohens_d", "effect_magnitude"]
    ].to_string(index=False))

    print("\n>> Step 8: Robustness check (imputation x scaling, RF vs LogReg)")
    robustness_df = robustness_check(merged)

    print("\n>> Step 9: Random Forest feature importance")
    importances, rf_model = rf_feature_importance(merged, diff_results)

    print("\n>> Step 10: Export metabolite lists for pathway enrichment")
    export_metabolite_lists(diff_results)

    print("\n=== PIPELINE COMPLETE ===")
    print("Outputs: sample_groups.csv, metabolite_matrix.csv, differential_results.csv,")
    print("         pca_scores.csv, opls_scores.csv, robustness_check.csv,")
    print("         rf_feature_importance.csv, all_metabolites.txt, significant_metabolites.txt")
    print("\nNext manual step: upload all_metabolites.txt (background) and")
    print("significant_metabolites.txt (query) to MetaboAnalyst for KEGG pathway enrichment.")
