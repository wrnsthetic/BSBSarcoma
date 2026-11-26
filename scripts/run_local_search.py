#!/usr/bin/env python
# A cada iteração, sorteia K vizinhos (cada um troca 1 CpG do subconjunto atual 
# por um CpG candidato), avalia todos com 5-fold CV e aceita o melhor se melhorar o F1.
# O histórico é salvo a cada iteração, se interromper o progresso fica.

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from utils import elapsed, fmt_eta

DATA_PATH    = Path("data/processed/sarcoma_top5000_variable_cpgs_imputed.csv")
CLASSES_PATH = Path("data/processed/sarcoma_top5000_classes.csv")
SPLIT_PATH   = Path("data/processed/train_test_split_80_20.csv")
HISTORY_OUT  = Path("data/processed/local_search_history.json")
BEST_OUT     = Path("data/processed/local_search_best_subset.csv")
REPORT_JSON  = Path("reports/local_search_report.json")

SUBSET_SIZE  = 50
CV_FOLDS     = 5
N_ESTIMATORS = 300
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-neighbors",          type=int, default=20)
    p.add_argument("--max-iter",             type=int, default=100)
    p.add_argument("--patience",             type=int, default=20)
    p.add_argument("--initial-subset-path",  default=None,
                   help="CSV com subconjunto inicial. Se omitido, usa top variância.")
    return p.parse_args()


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    print("Carregando dados...", flush=True)
    x_full = pd.read_csv(DATA_PATH, sep=";", index_col=0).T
    labels = pd.read_csv(CLASSES_PATH, sep=";").set_index("ID")["Methylation Class Name"]

    split = pd.read_csv(SPLIT_PATH, sep=";")
    train_ids = set(split.loc[split["split"] == "train_valid", "sample_id"])

    x_train = x_full.loc[x_full.index.isin(train_ids)].copy()
    y_train = labels.loc[x_train.index]
    print(f"Treino: {len(x_train)} amostras | {x_train.shape[1]} CpGs candidatos | "
          f"{y_train.nunique()} classes", flush=True)
    return x_train, y_train


def evaluate(x_train: pd.DataFrame, y_train: pd.Series, cpgs: list[str]) -> float:
    x = x_train[cpgs]
    n_splits = min(CV_FOLDS, int(y_train.value_counts().min()))
    if n_splits < 2:
        return 0.0
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for k, (tr, val) in enumerate(cv.split(x, y_train)):
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE + k,
            n_jobs=-1, class_weight="balanced_subsample", max_features="sqrt",
        )
        rf.fit(x.iloc[tr], y_train.iloc[tr])
        scores.append(f1_score(y_train.iloc[val], rf.predict(x.iloc[val]),
                               average="macro", zero_division=0))
    return float(np.mean(scores))


def neighbors(current: list[str], pool: list[str], k: int, rng: random.Random) -> list[tuple]:
    outside = [c for c in pool if c not in set(current)]
    seen: set[tuple] = set()
    result = []
    for _ in range(k * 20):  # tentativas extras pra evitar repetição
        if len(result) >= k:
            break
        rm = rng.choice(current)
        add = rng.choice(outside)
        if (rm, add) in seen:
            continue
        seen.add((rm, add))
        new = [c for c in current if c != rm] + [add]
        result.append((rm, add, new))
    return result


def save_best(cpgs: list[str], f1: float) -> None:
    BEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rank": range(1, len(cpgs)+1), "cpg_id": cpgs, "best_f1": f1}).to_csv(
        BEST_OUT, sep=";", index=False
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(RANDOM_STATE)

    x_train, y_train = load_data()

    # subconjunto inicial: carrega de arquivo ou pega os top por variância
    train_var = x_train.var(axis=0, ddof=0).sort_values(ascending=False)
    pool = train_var.index.tolist()

    if args.initial_subset_path:
        print(f"Carregando subconjunto inicial de {args.initial_subset_path}", flush=True)
        current = pd.read_csv(args.initial_subset_path, sep=";")["cpg_id"].tolist()
    else:
        current = pool[:SUBSET_SIZE]

    print(f"\nParâmetros: {args.n_neighbors} vizinhos/iter | "
          f"máx {args.max_iter} iter | patience {args.patience}", flush=True)

    print("\nAvaliando subconjunto inicial...", flush=True)
    t0 = time.perf_counter()
    current_f1 = evaluate(x_train, y_train, current)
    print(f"F1 inicial: {current_f1:.6f} ({elapsed(t0)} por avaliação CV)", flush=True)

    best, best_f1 = list(current), current_f1
    history = []
    no_improve = 0
    eval_times: list[float] = []
    total_start = time.perf_counter()

    for it in range(1, args.max_iter + 1):
        print(f"\n{'─'*68}", flush=True)
        print(f"[Iter {it:>3}/{args.max_iter}] F1={current_f1:.6f} | "
              f"melhor={best_f1:.6f} | sem melhoria={no_improve}/{args.patience}", flush=True)

        nbrs = neighbors(current, pool, args.n_neighbors, rng)
        nbr_results = []

        for ni, (rm, add, cpgs) in enumerate(nbrs, 1):
            t_n = time.perf_counter()
            f1 = evaluate(x_train, y_train, cpgs)
            dt = time.perf_counter() - t_n
            eval_times.append(dt)

            avg = float(np.mean(eval_times[-20:]))
            remaining = (len(nbrs) - ni) + (args.max_iter - it) * args.n_neighbors
            delta = f1 - current_f1
            sign = "+" if delta >= 0 else ""
            print(f"  viz {ni:>2}/{len(nbrs)} | F1={f1:.6f} ({sign}{delta:.6f}) | "
                  f"rm={rm} add={add} | {elapsed(t_n)} | ETA={fmt_eta(remaining * avg)}", flush=True)
            nbr_results.append({"rm": rm, "add": add, "f1": f1, "dt": round(dt, 2)})

        best_nbr = max(nbr_results, key=lambda r: r["f1"])
        accepted = best_nbr["f1"] > current_f1

        if accepted:
            old = current_f1
            current_f1 = best_nbr["f1"]
            current = [c for c in current if c != best_nbr["rm"]] + [best_nbr["add"]]
            no_improve = 0
            if current_f1 > best_f1:
                best_f1, best = current_f1, list(current)
                save_best(best, best_f1)
            print(f"\n  >> ACEITO: {old:.6f} → {current_f1:.6f} (+{current_f1-old:.6f})", flush=True)
        else:
            no_improve += 1
            print(f"\n  >> REJEITADO (melhor viz={best_nbr['f1']:.6f}) | "
                  f"sem melhoria: {no_improve}/{args.patience}", flush=True)

        print(f"  iter: {elapsed(total_start)} total | melhor global: {best_f1:.6f}", flush=True)

        history.append({
            "iteration": it, "accepted": accepted,
            "current_f1": round(current_f1, 6), "best_f1": round(best_f1, 6),
            "no_improve": no_improve, "neighbors": nbr_results,
        })
        # salva histórico incrementalmente
        HISTORY_OUT.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_OUT.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        if no_improve >= args.patience:
            print(f"\n[PARADA] {args.patience} iterações sem melhoria.", flush=True)
            break

    print(f"\n{'='*68}", flush=True)
    print(f"Melhor Macro F1 (CV): {best_f1:.6f}", flush=True)
    print(f"Iterações: {len(history)} | Tempo total: {elapsed(total_start)}", flush=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "result": {
            "best_f1_cv": round(best_f1, 6),
            "iterations": len(history),
            "best_cpgs": best,
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Melhor subconjunto: {BEST_OUT}")
    print(f"Histórico: {HISTORY_OUT}")


if __name__ == "__main__":
    main()
