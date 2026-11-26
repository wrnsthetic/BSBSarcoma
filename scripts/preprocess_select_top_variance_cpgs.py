#!/usr/bin/env python
# Pré-processamento do dataset de metilação de sarcoma.
# Lê o CSV bruto em chunks
# descarta CpGs com muitos valores faltantes, imputa os demais pela mediana
# e exporta os N CpGs mais variáveis para uso nos experimentos.

from __future__ import annotations

import csv
import heapq
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import elapsed

# caminhos e parâmetros
DATA_PATH    = Path("all_samples_sarcoma.csv")
CLASSES_PATH = Path("class_samples_training.csv")
OUT_DATA     = Path("data/processed/sarcoma_top5000_variable_cpgs_imputed.csv")
OUT_CLASSES  = Path("data/processed/sarcoma_top5000_classes.csv")
OUT_STATS    = Path("data/processed/top5000_cpg_variance_stats.csv")
REPORT_JSON  = Path("reports/preprocess_top5000_report.json")

TOP_N             = 5000   # quantos CpGs mais variáveis manter
MISSING_THRESHOLD = 0.05   # CpGs com >= 5% de faltantes são descartados
CHUNKSIZE         = 512    # linhas por chunk
PROGRESS_EVERY    = 25     # imprime progresso a cada N chunks


def read_class_labels(path: Path) -> dict[str, str]:
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            sid = (row.get("ID") or "").strip()
            label = (row.get("Methylation Class Name") or "").strip()
            if sid and label:
                labels[sid] = label
    return labels


def get_sample_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f, delimiter=";"))
    return header[1:]


def process_chunks(data_path: Path, missing_limit: int) -> tuple[list[dict], dict]:
    # calcula variância e mediana de cada CpG, mantém os top-N num heap
    heap: list[tuple[float, int, dict]] = []
    seq = 0
    totals = dict(total=0, kept=0, removed=0, with_missing=0, imputed=0)

    reader = pd.read_csv(
        data_path, sep=";", index_col=0, chunksize=CHUNKSIZE,
        na_values=["", "NA", "NaN", "nan", "null", "NULL"],
        keep_default_na=True, low_memory=False,
    )

    for chunk_n, chunk in enumerate(reader, 1):
        arr = chunk.to_numpy(dtype=np.float32, copy=True)
        missing = np.isnan(arr)
        missing_counts = missing.sum(axis=1)

        keep = missing_counts < missing_limit
        totals["total"]        += arr.shape[0]
        totals["removed"]      += int((~keep).sum())
        totals["with_missing"] += int((missing_counts > 0).sum())
        totals["imputed"]      += int(((missing_counts > 0) & keep).sum())
        totals["kept"]         += int(keep.sum())

        if not keep.any():
            continue

        arr_k   = arr[keep]
        miss_k  = missing[keep]
        medians = np.nanmedian(arr_k, axis=1).astype(np.float32)

        # imputa os faltantes pela mediana do próprio CpG antes de calcular variância
        rows, _ = np.where(miss_k)
        if rows.size:
            arr_k[miss_k] = medians[rows]

        variances = arr_k.var(axis=1, ddof=0)
        ids = chunk.index.to_numpy()[keep]

        for i, (cpg_id, var, med, mc) in enumerate(
            zip(ids, variances, medians, missing_counts[keep])
        ):
            stat = {"cpg_id": str(cpg_id), "variance": float(var),
                    "median": float(med), "missing_count": int(mc)}
            item = (float(var), seq + i, stat)
            if len(heap) < TOP_N:
                heapq.heappush(heap, item)
            elif float(var) > heap[0][0]:
                heapq.heapreplace(heap, item)

        seq += int(keep.sum())

        if PROGRESS_EVERY and chunk_n % PROGRESS_EVERY == 0:
            print(f"  [pass 1] chunk {chunk_n} | total={totals['total']} "
                  f"removidos={totals['removed']} elapsed={elapsed(run_start)}", flush=True)

    top = [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]
    return top, totals


def write_reduced_dataset(data_path: Path, top_stats: list[dict]) -> int:
    # relê o arquivo e grava apenas os CpGs selecionados, já imputados
    selected_ids = {s["cpg_id"] for s in top_stats}
    rank_map = {s["cpg_id"]: i for i, s in enumerate(top_stats)}

    reader = pd.read_csv(
        data_path, sep=";", index_col=0, chunksize=CHUNKSIZE,
        na_values=["", "NA", "NaN", "null", "NULL"],
        keep_default_na=True, low_memory=False,
    )

    frames = []
    found: set[str] = set()
    for chunk_n, chunk in enumerate(reader, 1):
        mask = chunk.index.isin(selected_ids)
        if not mask.any():
            continue
        sel = chunk.loc[mask].copy()
        arr = sel.to_numpy(dtype=np.float32, copy=True)
        medians = np.nanmedian(arr, axis=1).astype(np.float32)
        miss = np.isnan(arr)
        rows, _ = np.where(miss)
        if rows.size:
            arr[miss] = medians[rows]
        imputed = pd.DataFrame(arr, index=sel.index, columns=sel.columns)
        imputed["_rank"] = [rank_map[str(i)] for i in imputed.index]
        frames.append(imputed)
        found.update(str(i) for i in imputed.index)

        if PROGRESS_EVERY and chunk_n % PROGRESS_EVERY == 0:
            print(f"  [pass 2] chunk {chunk_n} | encontrados={len(found)}", flush=True)

    if not frames:
        raise RuntimeError("Nenhum CpG selecionado encontrado no arquivo.")

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(frames).sort_values("_rank").drop(columns=["_rank"])
    result.index.name = "ID_REF"
    result.to_csv(OUT_DATA, sep=";", float_format="%.8g")
    return len(found)


def main() -> None:
    global run_start
    run_start = time.perf_counter()

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(CLASSES_PATH)

    sample_ids = get_sample_ids(DATA_PATH)
    n_samples = len(sample_ids)
    missing_limit = math.ceil(n_samples * MISSING_THRESHOLD)

    print(f"Amostras: {n_samples} | limiar de missing: {missing_limit} | top-N: {TOP_N}", flush=True)

    print("\nPassagem 1: calculando variâncias e selecionando top CpGs...", flush=True)
    top_stats, totals = process_chunks(DATA_PATH, missing_limit)

    if len(top_stats) < TOP_N:
        raise RuntimeError(f"Só {len(top_stats)} CpGs passaram o filtro; esperava {TOP_N}.")

    # salva estatísticas dos CpGs selecionados
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_STATS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank","cpg_id","variance","missing_count","median"], delimiter=";")
        w.writeheader()
        for rank, s in enumerate(top_stats, 1):
            w.writerow({"rank": rank, **{k: s[k] for k in ["cpg_id","variance","missing_count","median"]}})

    print(f"\nPassagem 2: gravando dataset reduzido em {OUT_DATA}...", flush=True)
    n_written = write_reduced_dataset(DATA_PATH, top_stats)

    # alinha o arquivo de classes com as amostras do dataset
    labels = read_class_labels(CLASSES_PATH)
    OUT_CLASSES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CLASSES.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["ID", "Methylation Class Name"])
        for sid in sample_ids:
            if sid in labels:
                w.writerow([sid, labels[sid]])

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "data": str(DATA_PATH), "top_n": TOP_N,
            "missing_threshold": MISSING_THRESHOLD,
            "missing_threshold_count": missing_limit,
        },
        "summary": {
            "total_cpgs_read": totals["total"],
            "removed_by_missing": totals["removed"],
            "kept": totals["kept"],
            "imputed": totals["imputed"],
            "selected": n_written,
            "elapsed": elapsed(run_start),
        },
        "top20_preview": [{"rank": i+1, **s} for i, s in enumerate(top_stats[:20])],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nCpGs removidos (>= {MISSING_THRESHOLD:.0%} faltantes): {totals['removed']}")
    print(f"CpGs imputados por mediana: {totals['imputed']}")
    print(f"CpGs selecionados por variância: {n_written}")
    print(f"Dataset reduzido: {OUT_DATA}")
    print(f"Classes alinhadas: {OUT_CLASSES}")
    print(f"Tempo total: {elapsed(run_start)}")


if __name__ == "__main__":
    main()
