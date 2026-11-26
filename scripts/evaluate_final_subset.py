#!/usr/bin/env python
# Avaliação final: treina nos 80% completos e testa nos 20% que ficaram isolados.

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from utils import elapsed

DATA_PATH        = Path("data/processed/sarcoma_top5000_variable_cpgs_imputed.csv")
CLASSES_PATH     = Path("data/processed/sarcoma_top5000_classes.csv")
SPLIT_PATH       = Path("data/processed/train_test_split_80_20.csv")
BEST_SUBSET_PATH = Path("data/processed/local_search_best_subset.csv")
REPORT_JSON      = Path("reports/final_evaluation_report.json")

N_ESTIMATORS = 300
RANDOM_STATE = 42

# valores do baseline para a tabela de comparação no relatório
BASELINE_F1  = 0.817510
BASELINE_ACC = 0.888889


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_full = pd.read_csv(DATA_PATH, sep=";", index_col=0).T
    labels = pd.read_csv(CLASSES_PATH, sep=";").set_index("ID")["Methylation Class Name"]
    split  = pd.read_csv(SPLIT_PATH, sep=";")

    train_ids = set(split.loc[split["split"] == "train_valid", "sample_id"])
    test_ids  = set(split.loc[split["split"] == "test", "sample_id"])

    x_train = x_full.loc[x_full.index.isin(train_ids)].copy()
    x_test  = x_full.loc[x_full.index.isin(test_ids)].copy()
    return x_train, labels.loc[x_train.index], x_test, labels.loc[x_test.index]


def main() -> None:
    run_start = time.perf_counter()

    print("Carregando dados e split...", flush=True)
    x_train, y_train, x_test, y_test = load_splits()
    print(f"Treino: {len(x_train)} | Teste: {len(x_test)} | Classes: {y_train.nunique()}", flush=True)

    cpgs = pd.read_csv(BEST_SUBSET_PATH, sep=";")["cpg_id"].tolist()
    print(f"CpGs no subconjunto final: {len(cpgs)}", flush=True)

    # treina no treino completo (esse é o modelo final)
    print("\nTreinando modelo final...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced_subsample", max_features="sqrt",
    )
    rf.fit(x_train[cpgs], y_train)

    y_pred = rf.predict(x_test[cpgs])
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    accuracy  = float(accuracy_score(y_test, y_pred))

    print(f"\n{'='*60}", flush=True)
    print("RESULTADO FINAL (teste hold-out)", flush=True)
    print(f"  Macro F1 : {macro_f1:.6f}  (baseline: {BASELINE_F1:.6f}, delta={macro_f1-BASELINE_F1:+.6f})")
    print(f"  Accuracy : {accuracy:.6f}  (baseline: {BASELINE_ACC:.6f}, delta={accuracy-BASELINE_ACC:+.6f})")
    print(f"{'='*60}", flush=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "result": {
            "n_cpgs": len(cpgs), "test_samples": len(x_test),
            "n_classes": int(y_train.nunique()),
            "macro_f1": round(macro_f1, 6),
            "accuracy":  round(accuracy, 6),
        },
        "baseline_comparison": {
            "baseline_macro_f1": BASELINE_F1,
            "baseline_accuracy": BASELINE_ACC,
            "f1_delta":  round(macro_f1 - BASELINE_F1, 6),
            "acc_delta": round(accuracy  - BASELINE_ACC, 6),
        },
        "per_class_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "best_cpgs": cpgs,
        "total_elapsed": elapsed(run_start),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório salvo em {REPORT_JSON}")


if __name__ == "__main__":
    main()
