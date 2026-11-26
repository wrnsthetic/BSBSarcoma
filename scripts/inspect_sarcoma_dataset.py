#!/usr/bin/env python
# Inspeção exploratória do dataset bruto — lê o CSV sem carregar tudo na memória.
# Não faz parte do pipeline principal, foi usado só na fase inicial pra entender os dados.

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_PATH    = Path("all_samples_sarcoma.csv")
CLASSES_PATH = Path("class_samples_training.csv")
MAX_ROWS     = 1000   # quantas linhas de CpG inspecionar (None = todas)
REPORT_JSON  = Path("reports/dataset_inspection.json")


def human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_human": human_bytes(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def read_classes(path: Path) -> dict[str, Any]:
    labels: dict[str, str] = {}
    duplicates: list[str] = []
    rows_missing = 0
    total_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames or []
        if "ID" not in fieldnames or "Methylation Class Name" not in fieldnames:
            raise ValueError(f"Colunas esperadas 'ID' e 'Methylation Class Name' nao encontradas: {fieldnames}")

        for row in reader:
            total_rows += 1
            sid   = (row.get("ID") or "").strip()
            label = (row.get("Methylation Class Name") or "").strip()
            if not sid or not label:
                rows_missing += 1
                continue
            if sid in labels:
                duplicates.append(sid)
            labels[sid] = label

    dist = Counter(labels.values())
    return {
        "n_rows": total_rows,
        "n_labeled_samples": len(labels),
        "n_classes": len(dist),
        "class_distribution": dict(dist.most_common()),
        "n_duplicate_ids": len(duplicates),
        "rows_with_missing_values": rows_missing,
        "labels_by_sample": labels,
    }


def count_lines(path: Path) -> int:
    # conta linhas sem carregar o arquivo inteiro na memória
    count = 0
    last = b""
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        count += 1
    return max(count - 1, 0)  # desconta o header


def update_stats(stats: dict[str, Any], value_text: str) -> None:
    if value_text == "":
        stats["missing_values"] += 1
        return
    try:
        v = float(value_text)
    except ValueError:
        stats["non_numeric_values"] += 1
        return
    if math.isnan(v):
        stats["nan_values"] += 1
        return

    stats["numeric_values"] += 1
    stats["min_value"] = v if stats["min_value"] is None else min(stats["min_value"], v)
    stats["max_value"] = v if stats["max_value"] is None else max(stats["max_value"], v)
    stats["sum_value"]    += v
    stats["sum_sq_value"] += v * v
    if v < 0: stats["values_below_0"] += 1
    if v > 1: stats["values_above_1"] += 1


def inspect_methylation(path: Path, max_rows: int | None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "inspected_cpg_rows": 0,
        "first_cpg_ids": [],
        "n_duplicate_cpg_ids": 0,
        "rows_with_wrong_column_count": 0,
        "numeric_values": 0,
        "missing_values": 0,
        "non_numeric_values": 0,
        "nan_values": 0,
        "values_below_0": 0,
        "values_above_1": 0,
        "min_value": None,
        "max_value": None,
        "sum_value": 0.0,
        "sum_sq_value": 0.0,
    }
    seen: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Arquivo de metilacao vazio: {path}")

        sample_ids = [c.strip() for c in header[1:]]
        expected_cols = len(header)

        for i, row in enumerate(reader, 1):
            if max_rows is not None and i > max_rows:
                break
            stats["inspected_cpg_rows"] += 1
            cpg_id = row[0].strip() if row else ""
            if len(stats["first_cpg_ids"]) < 10:
                stats["first_cpg_ids"].append(cpg_id)
            if cpg_id in seen:
                stats["n_duplicate_cpg_ids"] += 1
            seen.add(cpg_id)
            if len(row) != expected_cols:
                stats["rows_with_wrong_column_count"] += 1
                continue
            for val in row[1:]:
                update_stats(stats, val.strip())

    n = stats["numeric_values"]
    mean = stats["sum_value"] / n if n else None
    var  = (stats["sum_sq_value"] / n - mean * mean) if n and mean is not None else None

    return {
        "n_sample_columns": len(sample_ids),
        "first_sample_ids": sample_ids[:10],
        "scan_mode": "full" if max_rows is None else f"first_{max_rows}_rows",
        "value_summary": {
            **{k: v for k, v in stats.items()},
            "mean_value": mean,
            "std_value": math.sqrt(max(var, 0.0)) if var is not None else None,
        },
        "sample_ids": sample_ids,
        "header_first_col": header[0] if header else None,
    }


def compare_samples(sample_ids: list[str], labels: dict[str, str]) -> dict[str, Any]:
    m_set = set(sample_ids)
    c_set = set(labels)
    return {
        "n_samples_in_methylation": len(sample_ids),
        "n_samples_in_classes": len(labels),
        "n_matching": len(m_set & c_set),
        "n_missing_class_labels": len(m_set - c_set),
        "n_labels_without_data": len(c_set - m_set),
        "n_duplicated_columns": sum(1 for _, n in Counter(sample_ids).items() if n > 1),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATA_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Arquivo de classes nao encontrado: {CLASSES_PATH}")

    print("Lendo classes...", flush=True)
    classes = read_classes(CLASSES_PATH)
    labels_by_sample = classes.pop("labels_by_sample")

    print(f"Inspecionando metilacao (primeiras {MAX_ROWS} linhas)...", flush=True)
    methylation = inspect_methylation(DATA_PATH, MAX_ROWS)
    comparison  = compare_samples(methylation["sample_ids"], labels_by_sample)

    # conta total de linhas sem carregar o arquivo — util pra confirmar quantos CpGs tem
    print("Contando total de linhas do arquivo...", flush=True)
    total_cpg_rows = count_lines(DATA_PATH)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": {
            "methylation": file_info(DATA_PATH),
            "classes":     file_info(CLASSES_PATH),
        },
        "total_cpg_rows": total_cpg_rows,
        "methylation": methylation,
        "classes": classes,
        "sample_comparison": comparison,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAmostras no arquivo de metilacao : {comparison['n_samples_in_methylation']}")
    print(f"Amostras com classe definida     : {classes['n_labeled_samples']}")
    print(f"Amostras correspondentes         : {comparison['n_matching']}")
    print(f"Classes distintas                : {classes['n_classes']}")
    print(f"Total de CpGs no arquivo         : {total_cpg_rows}")
    print(f"CpGs inspecionados               : {methylation['value_summary']['inspected_cpg_rows']}")
    print(f"Intervalo de valores             : {methylation['value_summary']['min_value']} a {methylation['value_summary']['max_value']}")
    print(f"\nRelatorio salvo em {REPORT_JSON}")


if __name__ == "__main__":
    main()
