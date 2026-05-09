"""DAG structure learning over climate variables driving ETo.

This module implements a Bayesian-network learner over the eight
climate variables typically required by FAO-56 Penman-Monteith
(Tmax, Tmin, Tmean, RH, u2, Rs, pr) plus the response variable
ETo. The learnt directed acyclic graph (DAG) makes explicit which
variables are direct parents of ETo and which are conditionally
independent of it given other parents — answering the same question
that \\citet{Ribeiro2024} addresses through a discrete BN, but with
two methodological refinements:

1. **Bin choice is documented**: each continuous variable is
   discretised by quantile binning (default 5 bins ≈ 20% quantiles).
   The number of bins is exposed as a parameter and reported in the
   paper, so the learnt DAG is fully reproducible.
2. **Score sensitivity test**: the learner is run with two
   independent scoring functions (BDeu \\citep{Buntine1991} and BIC
   \\citep{Schwarz1978}) and the structures are compared. A claim
   that a node is a parent of ETo is retained only if both scorers
   agree.

The learner is a thin wrapper around ``pgmpy.estimators.HillClimbSearch``
because that combination is the most widely cited in the climate-BN
literature (Hui et al. 2022, Sobie et al. 2020, Ribeiro et al. 2023)
and is fast enough to run on the 23,741-day MATOPIBA dataset on a
laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

# Lazy imports so the rest of the bwb package does not depend on pgmpy.
def _imports():
    from pgmpy.estimators import HillClimbSearch
    # pgmpy 1.1+ moved scoring functions to ``structure_score``
    try:
        from pgmpy.structure_score import BDeu as BDeuScore  # type: ignore
        from pgmpy.structure_score import BIC as BicScore    # type: ignore
    except ImportError:  # pragma: no cover
        from pgmpy.estimators import BDeuScore, BicScore     # type: ignore
    from pgmpy.models import DiscreteBayesianNetwork
    return HillClimbSearch, BDeuScore, BicScore, DiscreteBayesianNetwork


@dataclass
class DagLearningResult:
    edges: list[tuple[str, str]]
    parents_of: dict[str, list[str]]
    score_bdeu: float
    score_bic: float
    bin_edges: dict[str, np.ndarray]
    n_bins: int
    n_samples: int


def _quantile_bin(series: pd.Series, n_bins: int) -> tuple[pd.Series, np.ndarray]:
    """Equal-frequency binning with monotone integer labels."""
    s = series.dropna()
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(s.to_numpy(), qs)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    edges = np.unique(edges)
    if len(edges) - 1 < 2:
        # Degenerate: variable has too little spread; fall back to a single bin.
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index), edges
    labels = pd.cut(series, bins=edges, labels=False, include_lowest=True)
    return labels.fillna(-1).astype(int), edges


def discretise(df: pd.DataFrame, columns: Iterable[str], n_bins: int = 5
               ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Quantile-bin every requested column; return discretised frame + bin edges."""
    out = pd.DataFrame(index=df.index)
    bin_edges = {}
    for c in columns:
        out[c], bin_edges[c] = _quantile_bin(df[c], n_bins)
    return out, bin_edges


def learn_dag(df: pd.DataFrame, *, columns: Iterable[str],
              n_bins: int = 5, max_indegree: int = 4,
              random_seed: int = 42) -> DagLearningResult:
    """Run hill-climbing structure search with both BDeu and BIC scores.

    Edges accepted only when both scorers agree on the direction.
    """
    HillClimbSearch, BDeuScore, BicScore, _ = _imports()

    cols = list(columns)
    discrete, bin_edges = discretise(df.dropna(subset=cols), cols, n_bins=n_bins)

    rng = np.random.default_rng(random_seed)
    discrete = discrete.iloc[rng.permutation(len(discrete))].reset_index(drop=True)

    # pgmpy 1.1+ infers continuous vs discrete from dtype; integer columns
    # are treated as continuous unless explicitly cast to ``category``.
    for c in cols:
        discrete[c] = discrete[c].astype("category")

    # pgmpy 1.1+ accepts scoring methods either as strings or as instances.
    # Strings cover the common cases without needing to instantiate the
    # scoring class manually.
    hc_bdeu = HillClimbSearch(discrete)
    model_bdeu = hc_bdeu.estimate(
        scoring_method="bdeu",
        max_indegree=max_indegree, show_progress=False,
    )
    hc_bic = HillClimbSearch(discrete)
    model_bic = hc_bic.estimate(
        scoring_method="bic-d",
        max_indegree=max_indegree, show_progress=False,
    )
    edges_bdeu = set(model_bdeu.edges())
    edges_bic = set(model_bic.edges())
    consensus_edges = sorted(edges_bdeu & edges_bic)

    parents_of: dict[str, list[str]] = {c: [] for c in cols}
    for src, dst in consensus_edges:
        parents_of[dst].append(src)

    score_bdeu = float(BDeuScore(discrete, equivalent_sample_size=10).score(model_bdeu))
    score_bic = float(BicScore(discrete).score(model_bic))

    return DagLearningResult(
        edges=consensus_edges,
        parents_of=parents_of,
        score_bdeu=score_bdeu,
        score_bic=score_bic,
        bin_edges=bin_edges,
        n_bins=n_bins,
        n_samples=len(discrete),
    )


def variable_relevance_to_target(result: DagLearningResult,
                                  target: str = "ETo") -> pd.DataFrame:
    """Rank variables by Markov-blanket distance to the target node.

    A variable is **direct parent** if it points to ``target`` in the
    consensus DAG; **child** if ``target`` points to it; **co-parent**
    if it shares a child with target; otherwise **distal**.
    """
    parents_of = result.parents_of
    direct_parents = parents_of.get(target, [])
    children = [c for c, ps in parents_of.items() if target in ps]
    co_parents = []
    for child in children:
        for p in parents_of.get(child, []):
            if p != target and p not in direct_parents and p not in co_parents:
                co_parents.append(p)
    rows = []
    all_vars = sorted(parents_of.keys())
    for v in all_vars:
        if v == target:
            continue
        if v in direct_parents:
            cls = "direct parent"
            rank = 1
        elif v in children:
            cls = "direct child"
            rank = 2
        elif v in co_parents:
            cls = "co-parent (Markov blanket)"
            rank = 3
        else:
            cls = "distal"
            rank = 4
        rows.append({"variable": v, "relation": cls, "rank": rank})
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


__all__ = ["learn_dag", "discretise", "variable_relevance_to_target",
           "DagLearningResult"]
