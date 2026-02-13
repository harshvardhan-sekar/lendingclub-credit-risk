"""Run Notebook 04 pipeline: XGBoost + LightGBM + SHAP."""
import sys
import json
import time
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_PROCESSED_PATH, DATA_RESULTS_PATH, DATA_MODELS_PATH,
    TARGET_COL, RANDOM_STATE, FRED_SERIES,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_metrics(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = (tpr - fpr).max()
    brier = brier_score_loss(y_true, y_pred_proba)
    return {"AUC": round(auc, 4), "Gini": round(gini, 4),
            "KS": round(ks, 4), "Brier": round(brier, 4)}


def main():
    # ── Load data ──
    print("Loading data...", flush=True)
    train = pd.read_parquet(DATA_PROCESSED_PATH / "train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_PATH / "val.parquet")
    test = pd.read_parquet(DATA_PROCESSED_PATH / "test.parquet")
    y_train, y_val, y_test = train[TARGET_COL], val[TARGET_COL], test[TARGET_COL]
    print(f"  train={train.shape}, val={val.shape}, test={test.shape}", flush=True)

    # ── Engineer credit_history_years ──
    for df in [train, val, test]:
        df["credit_history_years"] = (
            (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
        ).clip(lower=0)

    # ── Feature exclusion ──
    LEAKAGE_COLS = [
        "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
        "last_pymnt_d", "last_fico_range_high", "last_fico_range_low",
        "next_pymnt_d", "last_credit_pull_d", "hardship_flag",
        "debt_settlement_flag",
    ]
    META_COLS = [
        TARGET_COL, "issue_d", "issue_month", "earliest_cr_line",
        "emp_title", "title", "zip_code",
    ]
    DROP_COLS = set(LEAKAGE_COLS + META_COLS)
    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
    print(f"  Features: {len(feature_cols)} ({len(cat_cols)} categorical)", flush=True)

    # ── Label encode ──
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([
            train[col].fillna("__MISSING__"),
            val[col].fillna("__MISSING__"),
            test[col].fillna("__MISSING__"),
        ]).unique()
        le.fit(all_vals)
        train[col] = le.transform(train[col].fillna("__MISSING__"))
        val[col] = le.transform(val[col].fillna("__MISSING__"))
        test[col] = le.transform(test[col].fillna("__MISSING__"))
        label_encoders[col] = le

    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]

    # ══════════════════════════════════════════════════════════════
    # XGBoost Optuna
    # ══════════════════════════════════════════════════════════════
    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_STATE, "n_jobs": -1,
            "eval_metric": "auc", "early_stopping_rounds": 50, "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print("\nXGBoost Optuna (30 trials)...", flush=True)
    xgb_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    t0 = time.time()
    xgb_study.optimize(xgb_objective, n_trials=30)
    xgb_tuning_time = time.time() - t0
    print(f"  Done: {xgb_tuning_time:.0f}s, best AUC: {xgb_study.best_value:.4f}", flush=True)

    # Final XGBoost
    bp = xgb_study.best_params.copy()
    bp.update({"random_state": RANDOM_STATE, "n_jobs": -1,
               "eval_metric": "auc", "early_stopping_rounds": 50, "verbosity": 0})
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(**bp)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_train_time = time.time() - t0
    xgb_test = compute_metrics(y_test, xgb_model.predict_proba(X_test)[:, 1])
    print(f"  XGB test: AUC={xgb_test['AUC']}, Gini={xgb_test['Gini']}, KS={xgb_test['KS']}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # LightGBM Optuna
    # ══════════════════════════════════════════════════════════════
    def lgbm_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print("\nLightGBM Optuna (30 trials)...", flush=True)
    lgbm_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    t0 = time.time()
    lgbm_study.optimize(lgbm_objective, n_trials=30)
    lgbm_tuning_time = time.time() - t0
    print(f"  Done: {lgbm_tuning_time:.0f}s, best AUC: {lgbm_study.best_value:.4f}", flush=True)

    # Final LightGBM
    bp2 = lgbm_study.best_params.copy()
    bp2.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    t0 = time.time()
    lgbm_model = lgb.LGBMClassifier(**bp2)
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgbm_train_time = time.time() - t0
    lgbm_test = compute_metrics(y_test, lgbm_model.predict_proba(X_test)[:, 1])
    print(f"  LGBM test: AUC={lgbm_test['AUC']}, Gini={lgbm_test['Gini']}, KS={lgbm_test['KS']}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # SHAP
    # ══════════════════════════════════════════════════════════════
    if xgb_test["AUC"] >= lgbm_test["AUC"]:
        shap_model, sname = xgb_model, "XGBoost"
        best_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    else:
        shap_model, sname = lgbm_model, "LightGBM"
        best_test_proba = lgbm_model.predict_proba(X_test)[:, 1]

    print(f"\nSHAP on {sname} (10K samples)...", flush=True)
    np.random.seed(RANDOM_STATE)
    shap_idx = np.random.choice(len(X_test), size=10_000, replace=False)
    X_shap = X_test.iloc[shap_idx]

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    print(f"  SHAP shape: {shap_vals.shape}", flush=True)

    # Feature importance
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    fi = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs_shap})
    fi = fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    macro_set = set(FRED_SERIES)
    fi["is_macro"] = fi["feature"].isin(macro_set)

    print("\nTop 20 features:", flush=True)
    for i, row in fi.head(20).iterrows():
        marker = " (MACRO)" if row["is_macro"] else ""
        print(f"  {i+1:2d}. {row['feature']:<35s} {row['mean_abs_shap']:.4f}{marker}", flush=True)

    macro_ranks = fi[fi["is_macro"]].index + 1
    macro_range = f"{macro_ranks.min()}-{macro_ranks.max()}"
    print(f"Macro ranks: {macro_range}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # SHAP-based Feature Selection
    # ══════════════════════════════════════════════════════════════
    shap_ranked_features = fi["feature"].tolist()
    K_CANDIDATES = [15, 20, 25, 30, 40, 50]
    sweep_results = []

    print("\nSHAP Feature Selection Sweep...", flush=True)
    for k in K_CANDIDATES:
        selected = shap_ranked_features[:k]
        X_tr_k, X_va_k, X_te_k = X_train[selected], X_val[selected], X_test[selected]

        xkp = xgb_study.best_params.copy()
        xkp.update({"random_state": RANDOM_STATE, "n_jobs": -1,
                     "eval_metric": "auc", "early_stopping_rounds": 50, "verbosity": 0})
        xk = xgb.XGBClassifier(**xkp)
        xk.fit(X_tr_k, y_train, eval_set=[(X_va_k, y_val)], verbose=False)
        xk_val = roc_auc_score(y_val, xk.predict_proba(X_va_k)[:, 1])
        xk_test = roc_auc_score(y_test, xk.predict_proba(X_te_k)[:, 1])

        lkp = lgbm_study.best_params.copy()
        lkp.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
        lk = lgb.LGBMClassifier(**lkp)
        lk.fit(X_tr_k, y_train, eval_set=[(X_va_k, y_val)], eval_metric="auc",
               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        lk_val = roc_auc_score(y_val, lk.predict_proba(X_va_k)[:, 1])
        lk_test = roc_auc_score(y_test, lk.predict_proba(X_te_k)[:, 1])

        sweep_results.append({
            "k": k, "xgb_val_auc": xk_val, "xgb_test_auc": xk_test,
            "lgbm_val_auc": lk_val, "lgbm_test_auc": lk_test,
            "xgb_gap": xgb_test["AUC"] - xk_test,
            "lgbm_gap": lgbm_test["AUC"] - lk_test,
        })
        print(f"  k={k:3d}: XGB={xk_test:.4f} LGBM={lk_test:.4f}", flush=True)

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df["max_gap"] = sweep_df[["xgb_gap", "lgbm_gap"]].max(axis=1)
    eligible = sweep_df[sweep_df["max_gap"] <= 0.005]
    if len(eligible) > 0:
        best_row = eligible.loc[eligible["k"].idxmin()]
    else:
        best_row = sweep_df.loc[sweep_df["max_gap"].idxmin()]
    BEST_K = int(best_row["k"])
    selected_features = shap_ranked_features[:BEST_K]
    print(f"  Selected k={BEST_K}", flush=True)

    # Retrain on selected features
    X_train_sel, X_val_sel, X_test_sel = X_train[selected_features], X_val[selected_features], X_test[selected_features]

    xsp = xgb_study.best_params.copy()
    xsp.update({"random_state": RANDOM_STATE, "n_jobs": -1,
                "eval_metric": "auc", "early_stopping_rounds": 50, "verbosity": 0})
    t0 = time.time()
    xgb_sel_model = xgb.XGBClassifier(**xsp)
    xgb_sel_model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)
    xgb_sel_train_time = time.time() - t0
    xgb_sel_test = compute_metrics(y_test, xgb_sel_model.predict_proba(X_test_sel)[:, 1])
    xgb_sel_train_m = compute_metrics(y_train, xgb_sel_model.predict_proba(X_train_sel)[:, 1])
    xgb_sel_val_m = compute_metrics(y_val, xgb_sel_model.predict_proba(X_val_sel)[:, 1])

    lsp = lgbm_study.best_params.copy()
    lsp.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    t0 = time.time()
    lgbm_sel_model = lgb.LGBMClassifier(**lsp)
    lgbm_sel_model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], eval_metric="auc",
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lgbm_sel_train_time = time.time() - t0
    lgbm_sel_test = compute_metrics(y_test, lgbm_sel_model.predict_proba(X_test_sel)[:, 1])
    lgbm_sel_train_m = compute_metrics(y_train, lgbm_sel_model.predict_proba(X_train_sel)[:, 1])
    lgbm_sel_val_m = compute_metrics(y_val, lgbm_sel_model.predict_proba(X_val_sel)[:, 1])

    print(f"  XGB sel test: AUC={xgb_sel_test['AUC']}, Gini={xgb_sel_test['Gini']}", flush=True)
    print(f"  LGBM sel test: AUC={lgbm_sel_test['AUC']}, Gini={lgbm_sel_test['Gini']}", flush=True)

    macro_survived = [f for f in selected_features if f in macro_set]

    # ══════════════════════════════════════════════════════════════
    # Save artifacts
    # ══════════════════════════════════════════════════════════════
    DATA_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    DATA_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    print("\nSaving...", flush=True)
    # Primary: SHAP-selected models
    with open(DATA_MODELS_PATH / "pd_xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_sel_model, f)
    with open(DATA_MODELS_PATH / "pd_lgbm_model.pkl", "wb") as f:
        pickle.dump(lgbm_sel_model, f)
    # Reference: full models
    with open(DATA_MODELS_PATH / "pd_xgboost_full_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(DATA_MODELS_PATH / "pd_lgbm_full_model.pkl", "wb") as f:
        pickle.dump(lgbm_model, f)
    print("  Models saved (primary=selected, reference=full)", flush=True)

    # Selected features
    sel_feat_out = {
        "selected_k": BEST_K,
        "selected_features": selected_features,
        "macro_features_retained": macro_survived,
        "full_feature_count": len(feature_cols),
        "sweep_results": sweep_df.to_dict(orient="records"),
        "selection_criteria": "smallest k with test AUC within 0.005 of full model",
    }
    with open(DATA_RESULTS_PATH / "selected_features.json", "w") as f:
        json.dump(sel_feat_out, f, indent=2)
    print("  selected_features.json saved", flush=True)

    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
    else:
        exp_val = float(exp_val)
    shap_out = {
        "shap_values": shap_vals,
        "feature_names": list(X_shap.columns),
        "sample_indices": shap_idx,
        "model_name": sname,
        "explainer_expected_value": exp_val,
    }
    with open(DATA_RESULTS_PATH / "shap_values.pkl", "wb") as f:
        pickle.dump(shap_out, f)
    print("  SHAP values saved", flush=True)

    # Load LR metrics for comparison
    with open(DATA_RESULTS_PATH / "pd_scorecard_metrics.json", "r") as f:
        lr_m = json.load(f)

    xgb_all = {
        "train": compute_metrics(y_train, xgb_model.predict_proba(X_train)[:, 1]),
        "validation": compute_metrics(y_val, xgb_model.predict_proba(X_val)[:, 1]),
        "test": xgb_test,
    }
    lgbm_all = {
        "train": compute_metrics(y_train, lgbm_model.predict_proba(X_train)[:, 1]),
        "validation": compute_metrics(y_val, lgbm_model.predict_proba(X_val)[:, 1]),
        "test": lgbm_test,
    }
    best_ml = "xgboost" if xgb_test["AUC"] >= lgbm_test["AUC"] else "lightgbm"
    mc = {
        "logistic_regression": {
            "train": lr_m["train"], "validation": lr_m["validation"],
            "test": lr_m["test"], "n_features": lr_m["n_features"],
            "features": lr_m["features"],
        },
        "xgboost_full": {
            **xgb_all, "n_features": len(feature_cols),
            "best_params": {
                k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
                for k, v in xgb_study.best_params.items()
            },
            "best_iteration": int(xgb_model.best_iteration),
            "train_time_s": round(xgb_train_time, 1),
            "tuning_time_s": round(xgb_tuning_time, 1),
        },
        "lightgbm_full": {
            **lgbm_all, "n_features": len(feature_cols),
            "best_params": {
                k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
                for k, v in lgbm_study.best_params.items()
            },
            "best_iteration": int(lgbm_model.best_iteration_),
            "train_time_s": round(lgbm_train_time, 1),
            "tuning_time_s": round(lgbm_tuning_time, 1),
        },
        "xgboost_selected": {
            "train": xgb_sel_train_m, "validation": xgb_sel_val_m,
            "test": xgb_sel_test, "n_features": BEST_K,
            "selected_features": selected_features,
            "best_iteration": int(xgb_sel_model.best_iteration),
            "train_time_s": round(xgb_sel_train_time, 1),
        },
        "lightgbm_selected": {
            "train": lgbm_sel_train_m, "validation": lgbm_sel_val_m,
            "test": lgbm_sel_test, "n_features": BEST_K,
            "selected_features": selected_features,
            "best_iteration": int(lgbm_sel_model.best_iteration_),
            "train_time_s": round(lgbm_sel_train_time, 1),
        },
        "best_model": best_ml,
        "selected_feature_columns": selected_features,
        "feature_columns": feature_cols,
        "label_encoders": {col: list(le.classes_) for col, le in label_encoders.items()},
    }
    with open(DATA_RESULTS_PATH / "model_comparison.json", "w") as f:
        json.dump(mc, f, indent=2, default=str)
    fi.to_csv(DATA_RESULTS_PATH / "xgboost_feature_importance.csv", index=False)
    print("  JSON + CSV saved", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════
    print("\nGenerating plots...", flush=True)

    # ROC + KS + Distribution
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for name, proba, color in [
        ("XGBoost", xgb_model.predict_proba(X_test)[:, 1], "#1f77b4"),
        ("LightGBM", lgbm_model.predict_proba(X_test)[:, 1], "#ff7f0e"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc_val:.4f})")
    axes[0].plot([], [], color="#2ca02c", linewidth=2, linestyle="--",
                 label=f"LR (AUC={lr_m['test']['AUC']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC — Test", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    si2 = np.argsort(best_test_proba)
    cpct = np.arange(1, len(si2) + 1) / len(si2)
    sl = y_test.values[si2]
    cb = np.cumsum(sl) / sl.sum()
    cg = np.cumsum(1 - sl) / (1 - sl).sum()
    axes[1].plot(cpct, cb, "r-", linewidth=2, label="Bad")
    axes[1].plot(cpct, cg, "b-", linewidth=2, label="Good")
    km = np.argmax(np.abs(cb - cg))
    axes[1].axvline(cpct[km], color="green", linestyle="--")
    axes[1].annotate(f"KS={np.abs(cb - cg)[km]:.4f}",
                     xy=(cpct[km], (cb[km] + cg[km]) / 2),
                     fontsize=11, fontweight="bold", color="green")
    axes[1].set_title(f"KS — {sname}", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].hist(best_test_proba[y_test == 0], bins=50, alpha=0.6,
                 label="Good", color="#2ca02c", density=True)
    axes[2].hist(best_test_proba[y_test == 1], bins=50, alpha=0.6,
                 label="Bad", color="#d62728", density=True)
    axes[2].set_title(f"PD Distribution — {sname}", fontweight="bold")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(DATA_RESULTS_PATH / "pd_ml_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pd_ml_diagnostics.png", flush=True)

    # SHAP beeswarm
    shap.summary_plot(shap_vals, X_shap, max_display=20, show=False)
    plt.title(f"SHAP — {sname} (Top 20)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DATA_RESULTS_PATH / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_beeswarm.png", flush=True)

    # SHAP top10 bar
    top10 = fi.head(10)
    colors = ["#d62728" if m else "#1f77b4" for m in top10["is_macro"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(9, -1, -1), top10["mean_abs_shap"], color=colors)
    ax.set_yticks(range(9, -1, -1))
    ax.set_yticklabels(top10["feature"])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top 10 — {sname}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(handles=[
        Patch(facecolor="#1f77b4", label="Standard"),
        Patch(facecolor="#d62728", label="Macro"),
    ])
    plt.tight_layout()
    plt.savefig(DATA_RESULTS_PATH / "shap_top10_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_top10_bar.png", flush=True)

    # SHAP dependence
    top3 = fi.head(3)["feature"].tolist()
    macro_in_top3 = [f for f in top3 if f in macro_set]
    dep_feats = top3 + ([fi[fi["is_macro"]].iloc[0]["feature"]] if not macro_in_top3 else [])
    n_dep = len(dep_feats)
    fig, axs = plt.subplots(1, n_dep, figsize=(6 * n_dep, 5))
    if n_dep == 1:
        axs = [axs]
    for i, feat in enumerate(dep_feats):
        feat_idx = list(X_shap.columns).index(feat)
        shap.dependence_plot(feat_idx, shap_vals, X_shap, ax=axs[i], show=False)
        label = f"{feat} (MACRO)" if feat in macro_set else feat
        axs[i].set_title(label, fontweight="bold")
    fig.tight_layout()
    plt.savefig(DATA_RESULTS_PATH / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_dependence.png", flush=True)

    # Individual explanations
    y_shap = y_test.values[shap_idx]
    proba_shap = best_test_proba[shap_idx]
    di = np.where(y_shap == 1)[0]
    ndi = np.where(y_shap == 0)[0]
    mdi = di[np.argmin(np.abs(proba_shap[di] - np.median(proba_shap[di])))]
    mndi = ndi[np.argmin(np.abs(proba_shap[ndi] - np.median(proba_shap[ndi])))]

    fig, axs2 = plt.subplots(2, 1, figsize=(14, 8))
    for ai, (idx, lab) in enumerate([(mdi, "Default"), (mndi, "Non-Default")]):
        sv = shap_vals[idx]
        fv = X_shap.iloc[idx]
        pp = proba_shap[idx]
        ti = np.argsort(np.abs(sv))[-10:][::-1]
        tf = [X_shap.columns[j] for j in ti]
        ts = sv[ti]
        c2 = ["#d62728" if v > 0 else "#2ca02c" for v in ts]
        axs2[ai].barh(range(9, -1, -1), ts, color=c2)
        axs2[ai].set_yticks(range(9, -1, -1))
        axs2[ai].set_yticklabels([f"{f}={fv[f]:.1f}" for f in tf], fontsize=9)
        axs2[ai].axvline(0, color="k", linewidth=0.5)
        axs2[ai].set_title(f"{lab} — PD={pp:.4f}", fontweight="bold")
        axs2[ai].grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    plt.savefig(DATA_RESULTS_PATH / "shap_individual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_individual.png", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("NOTEBOOK 04 SUMMARY (5 Models)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Model':<25s} {'Test AUC':>10s} {'Test Gini':>10s} {'Test KS':>10s} {'N Feat':>8s}", flush=True)
    print("-" * 66, flush=True)
    print(f"{'LR Scorecard':<25s} {lr_m['test']['AUC']:>10.4f} "
          f"{lr_m['test']['Gini']:>10.4f} {lr_m['test']['KS']:>10.4f} "
          f"{lr_m['n_features']:>8d}", flush=True)
    print(f"{'XGBoost (Full)':<25s} {xgb_test['AUC']:>10.4f} "
          f"{xgb_test['Gini']:>10.4f} {xgb_test['KS']:>10.4f} "
          f"{len(feature_cols):>8d}", flush=True)
    print(f"{'LightGBM (Full)':<25s} {lgbm_test['AUC']:>10.4f} "
          f"{lgbm_test['Gini']:>10.4f} {lgbm_test['KS']:>10.4f} "
          f"{len(feature_cols):>8d}", flush=True)
    print(f"{'XGBoost (Top ' + str(BEST_K) + ')':<25s} {xgb_sel_test['AUC']:>10.4f} "
          f"{xgb_sel_test['Gini']:>10.4f} {xgb_sel_test['KS']:>10.4f} "
          f"{BEST_K:>8d}", flush=True)
    print(f"{'LightGBM (Top ' + str(BEST_K) + ')':<25s} {lgbm_sel_test['AUC']:>10.4f} "
          f"{lgbm_sel_test['Gini']:>10.4f} {lgbm_sel_test['KS']:>10.4f} "
          f"{BEST_K:>8d}", flush=True)

    best_auc = max(xgb_test["AUC"], lgbm_test["AUC"])
    print(f"\nBest full: {best_ml} (AUC {best_auc})", flush=True)
    print(f"Feature reduction: {len(feature_cols)} → {BEST_K}", flush=True)
    print(f"Macro retained: {len(macro_survived)}/{len(FRED_SERIES)} — {macro_survived}", flush=True)
    print(f"Improvement over scorecard: +{best_auc - lr_m['test']['AUC']:.4f} AUC", flush=True)
    print(f"Macro ranks: {macro_range}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
