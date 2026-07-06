import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

merged = pd.read_csv("metabolite_matrix.csv")
meta_cols = ["sample_name", "disease", "group"]
metab_cols = [c for c in merged.columns if c not in meta_cols]

X_raw = merged[metab_cols].apply(pd.to_numeric, errors="coerce")
miss_frac = X_raw.isna().mean()
keep_cols = miss_frac[miss_frac < 0.3].index
X_raw = X_raw[keep_cols]
y = (merged["disease"] == "persistent").astype(int).values

def pareto_scale(X):
    return (X - X.mean()) / np.sqrt(X.std())

def vast_scale(X):
    return ((X - X.mean()) / X.std()) * (X.mean() / X.std())

results = []
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42)

imputations = {
    "median": X_raw.fillna(X_raw.median()),
    "knn": pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X_raw), columns=X_raw.columns),
}

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
        scores = cross_val_score(rf, X_scaled, y, cv=outer_cv, scoring="roc_auc")
        results.append({
            "imputation": imp_name, "scaling": scale_name,
            "mean_auc": scores.mean(), "std_auc": scores.std()
        })

res_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False)
print(res_df.to_string(index=False))
res_df.to_csv("robustness_check.csv", index=False)
