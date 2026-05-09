"""Generate a publication-grade methodology flowchart in matplotlib.

Outputs:
* ``figures/paper/methodology_flowchart.png``  (high-resolution raster)
* ``figures/paper/methodology_flowchart.pdf``  (vector for LaTeX inclusion)

Color scheme follows the advisor's recommendation:
* gray   — static/exogenous data
* blue   — computational steps
* green  — output / decision support
* orange — Bayesian conjugate update (highlighted)

Reference: docs/methodology_diagram.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
C_DATA = "#bdbdbd"
C_COMPUTE = "#74a9cf"
C_BAYES = "#fdae6b"
C_OUTPUT = "#74c476"
C_TEXT = "#2c2c2c"
EDGE = "#404040"


def _box(ax, x, y, w, h, text, facecolor, *, fontsize=9, fontweight="normal",
         text_color=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=facecolor, edgecolor=EDGE, linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color or C_TEXT,
            wrap=True)
    return (x, y, w, h)


def _arrow(ax, src, dst, *, label=None, ls="-", color=None, fontsize=8,
           connection="arc3,rad=0.0"):
    src_x, src_y, src_w, src_h = src
    dst_x, dst_y, dst_w, dst_h = dst
    sx = src_x + src_w / 2
    sy = src_y
    dx = dst_x + dst_w / 2
    dy = dst_y + dst_h
    a = FancyArrowPatch(
        (sx, sy), (dx, dy),
        arrowstyle="->,head_length=0.012,head_width=0.010",
        connectionstyle=connection,
        linewidth=1.2, color=color or EDGE, linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        ax.text((sx + dx) / 2 + 0.005, (sy + dy) / 2,
                label, fontsize=fontsize, color=color or EDGE,
                style="italic")


def _hline_arrow(ax, src_xy, dst_xy, *, label=None, ls="-", color=None,
                 fontsize=8):
    a = FancyArrowPatch(
        src_xy, dst_xy,
        arrowstyle="->,head_length=0.012,head_width=0.010",
        connectionstyle="arc3,rad=0.0",
        linewidth=1.2, color=color or EDGE, linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        mx, my = (src_xy[0] + dst_xy[0]) / 2, (src_xy[1] + dst_xy[1]) / 2
        ax.text(mx, my + 0.012, label, fontsize=fontsize, ha="center",
                color=color or EDGE)


def main():
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Top: INPUTS row ----
    inputs = _box(
        ax, x=0.02, y=0.86, w=0.30, h=0.10,
        facecolor=C_DATA, fontweight="bold",
        text="USER INPUTS\n• location (lat, lon, elevation)\n"
             "• crop name + planting_date\n"
             "• forecast_date  ($t$)  + horizon $H$",
    )
    static_lookup = _box(
        ax, x=0.36, y=0.86, w=0.30, h=0.10,
        facecolor=C_DATA, fontweight="bold",
        text="STATIC LOOKUP TABLES\n• AWC$_c$ (mm)  ← SNIRH/ANA per município\n"
             "• Kc curve  ← FAO-56 Cap. 6 (5-stage step)\n"
             "• $Z_r$, MAD  ← FAO-56 Tab. 22",
    )
    history = _box(
        ax, x=0.70, y=0.86, w=0.28, h=0.10,
        facecolor=C_DATA, fontweight="bold",
        text="HISTORICAL CLIMATE\nDaily P, ETo  1961..($t-1$)\n"
             "← Xavier (BR) / NASA-POWER\n"
             "/ ERA5 / OpenMeteo (global)",
    )

    # ---- Step 1: Climatology classification ----
    step1 = _box(
        ax, x=0.04, y=0.665, w=0.45, h=0.13,
        facecolor=C_COMPUTE, fontweight="bold",
        text=("STEP 1 — Climatology classification\n"
              "for each year $y \\in \\{1961..t-1\\}$:\n"
              "  $D_y = \\Sigma P - \\Sigma ETo$  (90 days)\n"
              "  SPEI$_y$ = $\\Phi^{-1}(F_{LL}(D_y))$\n"
              "  $\\kappa_y$ = tercile(SPEI$_y$) ∈ {dry, normal, wet}\n"
              "→ counts $(n_d, n_n, n_w)$"),
    )

    # ---- Step 2: Bayesian prior (highlighted) ----
    step2 = _box(
        ax, x=0.51, y=0.665, w=0.45, h=0.13,
        facecolor=C_BAYES, fontweight="bold",
        text=("STEP 2 — Bayesian prior  (CONJUGATE, no MCMC)\n"
              "$\\alpha_0 = (n_d, n_n, n_w) + \\mathbf{1}$\n"
              "$w \\sim \\mathrm{Dir}(\\alpha_0)$\n"
              "Sequential update across cycles:\n"
              "$\\alpha_{c+1} = \\alpha_c + e_{\\kappa^*}$"),
    )

    # ---- Step 3: state reconstruction ----
    step3 = _box(
        ax, x=0.04, y=0.435, w=0.92, h=0.13,
        facecolor=C_COMPUTE, fontweight="bold",
        text=(
            "STEP 3 — Soil-water state reconstruction (deterministic FAO-56, planting $\\to t$)\n"
            "FAO-56 daily forward run with observed P, ETo, Kc[d]:\n"
            "  $SW_d = \\mathrm{clip}(SW_{d-1} + 0.8\\,P_d - K_{c,d}\\,ETo_d,\\ 0,\\ AWC)$;  "
            "if  $SW_d < AWC\\,(1 - MAD)$:  refill to AWC, accumulate $I_d$.\n"
            "$\\to$ output: current state  $SW(t)$,  $I_{\\mathrm{to\\ date}}$"
        ),
    )

    # ---- Step 4: Horizon Monte-Carlo ----
    step4_clim = _box(
        ax, x=0.04, y=0.205, w=0.45, h=0.13,
        facecolor=C_COMPUTE,
        text=("STEP 4a — Horizon MC (climatology mode, default)\n"
              "for $i$ in $1..N$:\n"
              "  $w_i \\sim \\mathrm{Dir}(\\alpha_c)$  [analytical]\n"
              "  $\\kappa_i \\sim \\mathrm{Cat}(w_i)$\n"
              "  $y_i \\sim \\mathrm{Unif}(\\mathrm{years\\ of\\ class\\ }\\kappa_i)$\n"
              "  P, ETo  ← season $y_i$, days  $[t+1..t+H]$\n"
              "  FAO-56 forward from  $SW(t)$"),
    )
    step4_nwp = _box(
        ax, x=0.51, y=0.205, w=0.45, h=0.13,
        facecolor=C_COMPUTE,
        text=("STEP 4b — Horizon MC (NWP override, operational)\n"
              "given  ECMWF / GFS / OpenMeteo ensemble\n"
              "$\\{(P_{m,h}, ETo_{m,h})\\}_{m=1..M, h=1..H}$:\n"
              "for $i$ in $1..N$:\n"
              "  $m \\sim \\mathrm{Unif}(1..M)$\n"
              "  P, ETo  ← member $m$, full horizon\n"
              "  FAO-56 forward from  $SW(t)$"),
    )

    # ---- Step 5: Aggregation + Output ----
    step5 = _box(
        ax, x=0.04, y=0.020, w=0.92, h=0.13,
        facecolor=C_OUTPUT, fontweight="bold",
        text=(
            "STEP 5 — Aggregation $\\to$ OUTPUT  (decision support)\n"
            "$P(I_1 > 0)$,  $q_{05}, q_{50}, q_{95}$ of $I_h$ per day,  "
            "$q_{05}, q_{50}, q_{95}$ of  $\\Sigma_{h=1}^{H} I_h$\n"
            "$\\to$  \"Irrigate tomorrow with $X$ mm\"  or  \"no action\";    "
            "Verification (CRPS, KGE, PBIAS, coverage 90%) when $t$ in past."
        ),
    )

    # ---- Arrows ----
    # Inputs row → Step 1 / Step 2 / Step 3
    _arrow(ax, history, step1)
    _arrow(ax, static_lookup, step2)
    _arrow(ax, history, step3)
    _arrow(ax, inputs, step3)
    _arrow(ax, static_lookup, step3)

    # Step 1 → Step 2 (counts feed prior)
    _hline_arrow(ax,
                 (step1[0] + step1[2], step1[1] + step1[3] / 2),
                 (step2[0], step2[1] + step2[3] / 2),
                 label=r"$\alpha_0$",
                 color="#d2691e")

    # Step 2 → Step 4a (alpha drives Dirichlet sampling)
    _arrow(ax, step2, step4_clim, color="#d2691e")
    # Step 3 → Step 4a, Step 4b (SW(t) is initial condition)
    _arrow(ax, step3, step4_clim, label=r"$SW(t)$")
    _arrow(ax, step3, step4_nwp, label=r"$SW(t)$")

    # Step 4 → Step 5
    _arrow(ax, step4_clim, step5)
    _arrow(ax, step4_nwp, step5)

    # Title
    fig.text(0.5, 0.985, "Rolling 5-day operational irrigation forecast — methodology",
             ha="center", fontsize=13, fontweight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_DATA, label="Input / static lookup"),
        mpatches.Patch(color=C_COMPUTE, label="Computation"),
        mpatches.Patch(color=C_BAYES, label="Conjugate Bayesian update"),
        mpatches.Patch(color=C_OUTPUT, label="Output / decision"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005))

    plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0.04)

    out_png = OUT_DIR / "methodology_flowchart.png"
    out_pdf = OUT_DIR / "methodology_flowchart.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"  -> {out_png.relative_to(ROOT)}")
    print(f"  -> {out_pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
