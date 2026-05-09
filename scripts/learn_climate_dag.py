"""Learn the consensus DAG over climate variables driving ETo.

Uses the EVAOnline-style 8 variables (Tmax, Tmin, Tmean, RH, u2, Rs, pr,
ETo) on the pooled MATOPIBA daily series 1961-2025, and writes:

* ``output/paper_tables/table_dag_relevance.csv``  — variable relevance
  ranking (direct parent / child / co-parent / distal w.r.t. ETo).
* ``output/paper_tables/table_dag_relevance.tex`` — paper table.
* ``figures/paper/fig_climate_dag.png`` — graphviz-style DAG figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bwb.data.loaders import load_city_series  # noqa: E402
from bwb.inference.dag_learning import (  # noqa: E402
    learn_dag,
    variable_relevance_to_target,
)

CITIES = [
    "Baixa_Grande_do_Ribeiro", "Balsas", "Barreiras", "Bom_Jesus",
    "Campos_Lindos", "Correntina", "Formosa_do_Rio_Preto",
    "Luis_Eduardo_Magalhaes", "Tasso_Fragoso", "Urucui",
]
VARS = ["Tmax", "Tmin", "RH", "u2", "Rs", "pr", "ETo"]

OUT = ROOT / "output" / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = ROOT / "figures" / "paper"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    print("[1/3] loading 10 MATOPIBA cities, 1961-2025")
    pieces = []
    for c in CITIES:
        df = load_city_series(c)[["date"] + [v for v in VARS if v != "ETo"] + ["ETo"]]
        df["city"] = c
        pieces.append(df)
    full = pd.concat(pieces, ignore_index=True).dropna(subset=VARS)
    print(f"      pooled rows: {len(full):,}")

    print("[2/3] learning DAG (HillClimbSearch with BDeu and BIC, 5 bins, "
          "consensus edges)")
    result = learn_dag(full, columns=VARS, n_bins=5, max_indegree=4,
                       random_seed=42)
    print(f"      consensus edges: {len(result.edges)}")
    for src, dst in result.edges:
        print(f"        {src} -> {dst}")

    print("[3/3] computing variable relevance to ETo")
    rel = variable_relevance_to_target(result, target="ETo")
    print(rel.to_string(index=False))

    rel.to_csv(OUT / "table_dag_relevance.csv", index=False)

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Variable-relevance ranking with respect to ETo derived "
        r"from the consensus DAG learnt by hill-climbing search over the "
        r"pooled 1961--2025 MATOPIBA daily series ("
        f"$n={result.n_samples:,}$). "
        r"Edges retained only when BDeu \citep{Buntine1991} and BIC "
        r"\citep{Schwarz1978} scorers agree, with quantile binning at "
        f"{result.n_bins} bins per variable. ``Direct parent'' nodes "
        r"point at ETo in the DAG; ``co-parent'' nodes share a child "
        r"with ETo (Markov blanket); ``distal'' nodes are conditionally "
        r"independent of ETo given the parents.}",
        r"\label{tab:dag-relevance}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Variable & Relation to ETo \\",
        r"\midrule",
    ]
    pretty = {
        "Tmax": r"$T_{\mathrm{max}}$", "Tmin": r"$T_{\mathrm{min}}$",
        "RH": r"$\overline{\mathrm{RH}}$", "u2": r"$u_2$",
        "Rs": r"$R_{\mathrm{s}}$", "pr": r"$P$",
    }
    for _, row in rel.iterrows():
        v = pretty.get(row["variable"], row["variable"])
        tex_lines.append(f"{v} & {row['relation']} \\\\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = OUT / "table_dag_relevance.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"\nWrote {tex_path.relative_to(ROOT)}")

    # ==========================================
    # CORREÇÃO DA FIGURA (matplotlib + networkx)
    # ==========================================
    import matplotlib.pyplot as plt
    import networkx as nx
    
    G = nx.DiGraph()
    G.add_nodes_from(VARS)
    
    # Adiciona as arestas aprendidas pelo modelo
    for src, dst in result.edges:
        G.add_edge(src, dst)

    # Highlight ETo
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pos = nx.spring_layout(G, seed=42, k=2.0)
    node_colors = ["#fdae6b" if n == "ETo" else "#74a9cf" for n in G.nodes]
    
    NODE_SIZE = 1800 

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=NODE_SIZE, edgecolors="#404040", ax=ax)
                           
    # Correção 1: arrows=True e node_size=NODE_SIZE para a seta não ficar escondida sob o nó
    nx.draw_networkx_edges(G, pos, edge_color="#404040",
                           arrowsize=22, width=1.6,
                           connectionstyle="arc3,rad=0.08", 
                           node_size=NODE_SIZE, arrows=True, ax=ax)
                           
    # Correção 2: Substituir visualmente "pr" por "P"
    labels_map = {n: "P" if n == "pr" else n for n in G.nodes}
    
    nx.draw_networkx_labels(
        G, pos, labels=labels_map, font_size=11, ax=ax,
    )
    
    ax.set_title("Consensus DAG over climate variables driving ETo\n"
                 "(BDeu + BIC, 5-bin discretisation, 1961-2025 pooled)",
                 fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig_path = FIGS / "fig_climate_dag.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()