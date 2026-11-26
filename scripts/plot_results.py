"""Gera gráficos a partir dos relatórios JSON em reports/.

Não precisa do dataset bruto nem de data/processed/ — usa só os JSONs
já gerados pelo pipeline. Saída: reports/figures/*.png

    python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela, só salva os arquivos
import matplotlib.pyplot as plt

REPORTS = Path("reports")
OUT = REPORTS / "figures"

# Paleta: baseline em cinza (referência), Hill Climbing em azul (destaque)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a8984"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"


def load(name):
    with open(REPORTS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def style(ax):
    """Eixos discretos: grade leve, sem bordas superiores/direitas."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def short(cls):
    return cls.replace("methylation class ", "")


def plot_comparison(base, final):
    """Baseline vs Hill Climbing no conjunto de teste."""
    metrics = ["Macro F1", "Accuracy"]
    b = [base["test"]["macro_f1"], base["test"]["accuracy"]]
    h = [final["result"]["macro_f1"], final["result"]["accuracy"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = range(len(metrics))
    w = 0.36
    bars_b = ax.bar([i - w / 2 - 0.01 for i in x], b, w, color=GRAY, label="Baseline (top 50 variância)")
    bars_h = ax.bar([i + w / 2 + 0.01 for i in x], h, w, color=BLUE, label="Hill Climbing (50 CpGs)")
    for bars in (bars_b, bars_h):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(list(x), metrics)
    ax.set_ylim(0, 1.08)
    ax.set_title("Desempenho no teste (216 amostras, 65 classes)", color=TEXT, loc="left", pad=28)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "1_baseline_vs_hill_climbing.png", dpi=150)
    plt.close(fig)


def plot_convergence(local, base):
    """Evolução do Macro F1 (CV) ao longo das iterações do Hill Climbing."""
    hist = local["history_summary"]
    it = [r["iteration"] for r in hist]
    best = [r["best_f1"] for r in hist]
    accepted = [(r["iteration"], r["best_f1"]) for r in hist if r["accepted"]]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.step(it, best, where="post", color=BLUE, linewidth=2, label="Melhor Macro F1 (CV)")
    if accepted:
        ax.scatter(*zip(*accepted), s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
                   label="Vizinho aceito")
    ax.axhline(base["cv"]["macro_f1_mean"], color=GRAY, linestyle="--", linewidth=1.5,
               label="Baseline (CV)")
    ax.set_xlabel("Iteração", color=TEXT2)
    ax.set_ylabel("Macro F1 (5-fold CV)", color=TEXT2)
    ax.set_title("Convergência do Hill Climbing (execução continuada)", color=TEXT, loc="left")
    ax.legend(frameon=False, loc="center right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "2_hill_climbing_convergencia.png", dpi=150)
    plt.close(fig)


def plot_per_class(base, final):
    """F1 por classe: baseline vs Hill Climbing (dumbbell), ordenado pelo ganho."""
    skip = {"accuracy", "macro avg", "weighted avg"}
    b = base["test"]["classification_report"]
    h = final["per_class_report"]
    classes = [c for c in h if c not in skip and c in b]
    classes.sort(key=lambda c: h[c]["f1-score"] - b[c]["f1-score"])

    fig, ax = plt.subplots(figsize=(9, 15))
    for i, c in enumerate(classes):
        fb, fh = b[c]["f1-score"], h[c]["f1-score"]
        color = BLUE if fh > fb else ORANGE if fh < fb else GRAY
        ax.plot([fb, fh], [i, i], color=color, linewidth=2, alpha=0.6)
    ax.scatter([b[c]["f1-score"] for c in classes], range(len(classes)), s=36, color=GRAY,
               edgecolor="white", linewidth=1, zorder=3, label="Baseline")
    ax.scatter([h[c]["f1-score"] for c in classes], range(len(classes)), s=36, color=BLUE,
               edgecolor="white", linewidth=1, zorder=3, label="Hill Climbing")
    ax.set_ylim(-0.8, len(classes) - 0.2)
    ax.set_yticks(range(len(classes)),
                  [f"{short(c)} (n={int(h[c]['support'])})" for c in classes], fontsize=8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("F1-score no teste", color=TEXT2)
    ax.set_title("F1 por classe — ordenado pelo ganho do Hill Climbing", color=TEXT, loc="left", pad=24)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "3_f1_por_classe.png", dpi=150)
    plt.close(fig)


def plot_cv_folds(base):
    """Variação do baseline entre os folds da CV."""
    folds = base["cv"]["folds"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    f = [d["fold"] for d in folds]
    ax.plot(f, [d["macro_f1"] for d in folds], marker="o", color=BLUE, linewidth=2, label="Macro F1")
    ax.plot(f, [d["accuracy"] for d in folds], marker="o", color=ORANGE, linewidth=2, label="Accuracy")
    ax.set_xticks(f)
    ax.set_xlabel("Fold", color=TEXT2)
    ax.set_title("Baseline — métricas por fold (5-fold CV)", color=TEXT, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "4_baseline_cv_folds.png", dpi=150)
    plt.close(fig)


def plot_top_variance(pre):
    """Variância dos CpGs mais variáveis (preview do pré-processamento)."""
    top = pre["top_cpgs_preview"]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(top))))
    ids = [t["cpg_id"] for t in top][::-1]
    var = [t["variance"] for t in top][::-1]
    ax.barh(ids, var, color=BLUE, height=0.7)
    ax.set_xlabel("Variância do beta-value", color=TEXT2)
    ax.set_title(f"Top {len(top)} CpGs por variância", color=TEXT, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "5_top_cpgs_variancia.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load("baseline_rf_top50_report")
    final = load("final_evaluation_report")
    local = load("local_search_report")
    pre = load("preprocess_top5000_report")

    plot_comparison(base, final)
    plot_convergence(local, base)
    plot_per_class(base, final)
    plot_cv_folds(base)
    plot_top_variance(pre)
    print(f"Gráficos salvos em {OUT}/")


if __name__ == "__main__":
    main()
