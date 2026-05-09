"""Generate the two-panel study-area figure for the manuscript.

Panel A (left): Brazil with state boundaries and the MATOPIBA polygon
outlined in red.

Panel B (right): MATOPIBA region with the per-pixel volumetric AWC
choropleth (ANA/SNIRH soil-physics dataset; the same source feeding
``tab:soils-matopiba``) overlaid on the state boundaries, with the ten
validation cities labelled and numbered alphabetically.

Inputs
------
* ``data_raw/geojson/BR_UF_2024.geojson``   -- Brazilian state boundaries
* ``data_raw/geojson/Matopiba_Perimetro.geojson`` -- MATOPIBA polygon
* ``data_raw/shapefile/AWC.shp``            -- ANA AWC (volumetric m^3/m^3)
* ``src/bwb/config/regional/matopiba.toml`` -- the 10 city centroids

Outputs
-------
* ``figures/paper/study_area_map.png`` (300 dpi)
* ``figures/paper/study_area_map.pdf``

Modelled after ``1_generate_matopiba_map.py`` (provided as a reference)
but with two simplifications: (i) we replace the full Köppen climate
shapefile (which we do not ship) with the ANA AWC choropleth, since
AWC is the variable the manuscript actually uses to elicit the
hydraulic priors; (ii) we drop the heavy three-column climate legend
and use a single AWC colourbar.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEOJSON_BR = ROOT / "data_raw" / "geojson" / "BR_UF_2024.geojson"
GEOJSON_MATOPIBA = ROOT / "data_raw" / "geojson" / "Matopiba_Perimetro.geojson"
SHAPE_AWC = ROOT / "data_raw" / "shapefile" / "AWC.shp"
TOML_PROFILE = ROOT / "src" / "bwb" / "config" / "regional" / "matopiba.toml"

OUT_DIR = ROOT / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG = OUT_DIR / "study_area_map.png"
PDF = OUT_DIR / "study_area_map.pdf"


def load_cities() -> pd.DataFrame:
    """Read the ten validation cities from the matopiba.toml profile."""
    cfg = tomllib.loads(TOML_PROFILE.read_text(encoding="utf-8"))
    cities = cfg["cities"]
    rows = []
    for name, (lat, lon) in cities.items():
        pretty = name.replace("_", " ")
        if pretty == "Luis Eduardo Magalhaes":
            pretty = "Luís Eduardo Magalhães"
        if pretty == "Urucui":
            pretty = "Urucuí"
        rows.append({"name": pretty, "code": name, "lat": lat, "lon": lon})
    df = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    df["number"] = df.index + 1
    return df


def main() -> None:
    print("Loading geospatial data...")
    gdf_br = gpd.read_file(GEOJSON_BR).to_crs(epsg=4326)
    gdf_matopiba = gpd.read_file(GEOJSON_MATOPIBA).to_crs(epsg=4326)
    gdf_awc = gpd.read_file(SHAPE_AWC).to_crs(epsg=4326)
    print(f"  BR_UF: {len(gdf_br)} polygons")
    print(f"  MATOPIBA: bounds {gdf_matopiba.total_bounds}")
    print(f"  AWC: {len(gdf_awc)} polygons (m^3/m^3)")

    matopiba_bounds = gdf_matopiba.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = matopiba_bounds

    # Clip AWC to a box around MATOPIBA (faster + cleaner panel)
    bbox = (minx - 1.0, miny - 1.0, maxx + 1.0, maxy + 1.0)
    gdf_awc_clip = gpd.clip(gdf_awc, mask=gpd.GeoDataFrame(
        {"geometry": [
            __import__("shapely.geometry", fromlist=["box"]).box(*bbox)
        ]}, crs="EPSG:4326",
    )).copy()
    print(f"  AWC clipped to MATOPIBA bbox: {len(gdf_awc_clip)} polygons")

    # Cities (one distinct colour per hub from tab20)
    cities = load_cities()
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(cities)))
    cities["color"] = list(palette)
    print(f"  Cities: {len(cities)}")

    # Figure layout
    #   row 0: panels A and B (maps)
    #   row 1: AWC colourbar (only under panel B, separated by buffer row)
    # Outer constrained_layout=False to control spacing manually.
    fig = plt.figure(figsize=(15, 8.5), dpi=130)
    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[40, 1],
        width_ratios=[1, 1.35],
        hspace=0.30, wspace=0.18,
        left=0.05, right=0.98, top=0.92, bottom=0.08,
    )
    ax_br = fig.add_subplot(gs[0, 0])
    ax_mp = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 1])

    # ----- Panel A: Brazil -----
    gdf_br.plot(
        ax=ax_br, facecolor="#F5F5F5", edgecolor="black",
        linewidth=0.6, zorder=2,
    )
    gdf_matopiba.plot(
        ax=ax_br, facecolor="#FFE4E1", edgecolor="red",
        linewidth=1.5, alpha=0.7, zorder=3,
    )
    ax_br.set_xlim(-75, -33)
    ax_br.set_ylim(-35, 6)
    ax_br.set_aspect("equal")
    ax_br.set_title("(A) Brazil and the MATOPIBA region", fontsize=12,
                    fontweight="bold")
    ax_br.set_xlabel("Longitude")
    ax_br.set_ylabel("Latitude")
    ax_br.grid(linestyle=":", color="grey", alpha=0.4)

    # State labels for the four MATOPIBA states
    state_labels = {
        "MA": (-45.5, -5.5), "TO": (-48.5, -10.5),
        "PI": (-43.0, -8.0), "BA": (-42.0, -12.5),
    }
    for sigla, (lon, lat) in state_labels.items():
        ax_br.text(lon, lat, sigla, fontsize=9, fontweight="bold",
                   ha="center", va="center", color="#222222",
                   path_effects=[PathEffects.withStroke(
                       linewidth=2, foreground="white")])

    # Inset rectangle showing where panel B zooms
    rect = mpatches.Rectangle(
        (minx, miny), maxx - minx, maxy - miny,
        fill=False, edgecolor="red", linewidth=1.5, zorder=4,
    )
    ax_br.add_patch(rect)

    # ----- Panel B: MATOPIBA + AWC + cities -----
    awc_min, awc_max = 0.025, 0.21
    cmap = plt.get_cmap("YlGnBu")
    gdf_awc_clip.plot(
        ax=ax_mp, column="AWC", cmap=cmap,
        vmin=awc_min, vmax=awc_max,
        linewidth=0, alpha=0.85, zorder=1,
    )
    gdf_br.plot(
        ax=ax_mp, facecolor="none", edgecolor="black",
        linewidth=0.6, zorder=3,
    )
    gdf_matopiba.plot(
        ax=ax_mp, facecolor="none", edgecolor="red",
        linewidth=2.0, zorder=4,
    )

    ax_mp.set_xlim(minx - 1.0, maxx + 0.6)
    ax_mp.set_ylim(miny - 1.0, maxy + 0.6)
    ax_mp.set_aspect("equal")
    ax_mp.set_title(
        "(B) MATOPIBA: ANA volumetric AWC and the 10 validation cities",
        fontsize=12, fontweight="bold",
    )
    ax_mp.set_xlabel("Longitude")
    ax_mp.set_ylabel("Latitude")
    ax_mp.grid(linestyle=":", color="grey", alpha=0.4)

    # Cities: one distinct colour per hub, with a numeric label offset to
    # avoid overlap with the marker.
    for _, row in cities.iterrows():
        ax_mp.scatter(
            row["lon"], row["lat"],
            s=260, marker="o", facecolor=row["color"],
            edgecolor="black", linewidth=1.4, zorder=10,
        )
        ax_mp.text(
            row["lon"] + 0.22, row["lat"] + 0.12,
            str(int(row["number"])),
            ha="left", va="bottom", fontsize=11, fontweight="bold",
            color="black", zorder=11,
            path_effects=[PathEffects.withStroke(
                linewidth=3.0, foreground="white")],
        )

    # Numbered legend with the same colour as the marker
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=row["color"], markeredgecolor="black",
               markersize=13, linestyle="None",
               label=f"{int(row['number'])}. {row['name']}")
        for _, row in cities.iterrows()
    ]
    leg = ax_mp.legend(
        handles=legend_handles, title="Validation cities",
        loc="upper left", fontsize=10, title_fontsize=11,
        frameon=True, framealpha=0.95, edgecolor="black",
        ncol=1, bbox_to_anchor=(1.02, 1.0),
        handletextpad=0.6, labelspacing=0.7, borderpad=0.8,
    )
    leg.get_title().set_fontweight("bold")

    # AWC colourbar (separate row, fully under panel B)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=awc_min, vmax=awc_max),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(
        r"AWC (volumetric m$^{3}$ m$^{-3}$, ANA/SNIRH;  "
        r"AWC$_{\mathrm{mm}}$ = AWC $\times Z_r \times 10$)",
        fontsize=10,
    )
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Study area: ten MATOPIBA cropping hubs validated in this study",
        fontsize=13, fontweight="bold", y=0.97,
    )

    plt.savefig(PNG, dpi=300, bbox_inches="tight")
    plt.savefig(PDF, bbox_inches="tight")
    print(f"OK: wrote {PNG.name} and {PDF.name} into {OUT_DIR}")


if __name__ == "__main__":
    main()
