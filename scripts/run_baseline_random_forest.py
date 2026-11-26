#!/usr/bin/env python
# Baseline é o Random Forest nos 50 CpGs mais variáveis (sem busca local).
# Serve como ponto de comparação ,se a busca local não superar isso, algo está errado.

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils import elapsed

DATA_PATH    = Path("data/processed/sarcoma_top5000_variable_cpgs_imputed.csv")
CLASSES_PATH = Path("data/processed/sarcoma_top5000_classes.csv")
SPLIT_OUT    = Path("data/processed/train_test_split_80_20.csv")
SUBSET_OUT   = Path("data/processed/baseline_top50_cpgs.csv")
REPORT_JSON  = Path("reports/baseline_rf_top50_report.json")

SUBSET_SIZE  = 50
TEST_SIZE    = 0.20
CV_FOLDS     = 5
N_ESTIMATORS = 300
RANDOM_STATE = 42


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )


def run_cv(x: pd.DataFrame, y: pd.Series) -> list[dict]:
    # limita o número de folds ao tamanho da menor classe, pq a gente tinha umas com só 7 exemplos
    n_splits = min(CV_FOLDS, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for fold, (tr, val) in enumerate(cv.split(x, y), 1):
        t0 = time.perf_counter()
        model = make_rf(RANDOM_STATE + fold)
        model.fit(x.iloc[tr], y.iloc[tr])
        pred = model.predict(x.iloc[val])
        results.append({
            "fold": fold,
            "macro_f1": float(f1_score(y.iloc[val], pred, average="macro")),
            "accuracy": float(accuracy_score(y.iloc[val], pred)),
            "elapsed": elapsed(t0),
        })
        print(f"  fold {fold}/{n_splits} | F1={results[-1]['macro_f1']:.4f} | "
              f"acc={results[-1]['accuracy']:.4f} | {results[-1]['elapsed']}", flush=True)
    return results


def main() -> None:
    run_start = time.perf_counter()

    print("Carregando dados...", flush=True)
    x_full = pd.read_csv(DATA_PATH, sep=";", index_col=0).T
    labels = pd.read_csv(CLASSES_PATH, sep=";").set_index("ID")["Methylation Class Name"]
    y_full = labels.loc[x_full.index]

    # divisão estratificada 80/20, salva o split pra reusar em todos os experimentos
    x_train, x_test, y_train, y_test = train_test_split(
        x_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
    )

    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["sample_id", "split"])
        for sid in x_train.index:
            w.writerow([sid, "train_valid"])
        for sid in x_test.index:
            w.writerow([sid, "test"])

    print(f"Treino: {len(x_train)} | Teste: {len(x_test)} | Classes: {y_full.nunique()}", flush=True)

    # top 50 CpGs por variância calculada apenas no treino, evita vazamento
    train_var = x_train.var(axis=0, ddof=0).sort_values(ascending=False)
    selected = train_var.head(SUBSET_SIZE).index.tolist()

    SUBSET_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "rank": range(1, SUBSET_SIZE + 1),
        "cpg_id": selected,
        "train_variance": train_var.loc[selected].values,
    }).to_csv(SUBSET_OUT, sep=";", index=False)

    print(f"\nRodando {CV_FOLDS}-fold CV nos {len(x_train)} exemplos de treino...", flush=True)
    cv_results = run_cv(x_train[selected], y_train)

    f1_vals  = [r["macro_f1"] for r in cv_results]
    acc_vals = [r["accuracy"]  for r in cv_results]
    print(f"\nCV Macro F1: {np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}", flush=True)

    # avaliação final no teste hold-out!!!!!
    final_model = make_rf(RANDOM_STATE)
    final_model.fit(x_train[selected], y_train)
    test_pred = final_model.predict(x_test[selected])
    test_f1  = float(f1_score(y_test, test_pred, average="macro"))
    test_acc = float(accuracy_score(y_test, test_pred))

    print(f"Teste final — Macro F1: {test_f1:.4f} | Accuracy: {test_acc:.4f}", flush=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "subset_size": SUBSET_SIZE, "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS, "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
        },
        "split": {
            "train_samples": len(x_train), "test_samples": len(x_test),
            "n_classes": int(y_full.nunique()),
            "min_train_class": int(y_train.value_counts().min()),
        },
        "cv": {
            "macro_f1_mean": float(np.mean(f1_vals)),
            "macro_f1_std":  float(np.std(f1_vals)),
            "accuracy_mean": float(np.mean(acc_vals)),
            "accuracy_std":  float(np.std(acc_vals)),
            "folds": cv_results,
        },
        "test": {
            "macro_f1": test_f1,
            "accuracy": test_acc,
            "classification_report": classification_report(
                y_test, test_pred, output_dict=True, zero_division=0
            ),
        },
        "total_elapsed": elapsed(run_start),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório salvo em {REPORT_JSON}")
    print(f"Tempo total: {elapsed(run_start)}")


if __name__ == "__main__":
    main()
