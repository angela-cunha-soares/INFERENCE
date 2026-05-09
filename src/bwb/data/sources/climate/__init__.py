"""Climate data sources for the bwb framework.

Three downloaders + one fuser, all returning DataFrames in the project's
canonical schema:

    ``date, Rs, u2, Tmax, Tmin, RH, pr, ETo``

This is the same schema used by :func:`bwb.data.loaders.load_city_series`,
so external sources can be used as drop-in alternatives to (or augmentations
of) the curated MATOPIBA Xavier reanalysis dataset.

* :func:`download_nasa_power` — NASA POWER MERRA-2 (1990 → today − 7 days,
  global 0.5° × 0.625°).
* :func:`download_openmeteo_archive` — Open-Meteo / ERA5 reanalysis
  (1940 → today − 2 days).
* :func:`download_openmeteo_forecast` — Open-Meteo NWP forecast
  (today − 29 days → today + 16 days).
* :func:`fuse_climate_sources` — multi-source fusion using the
  per-variable weights calibrated by EVAOnline against BR-DWGD.

Operational deployment for 2026 should fuse all three sources; backtests
covering 1961-2019 are restricted to NASA POWER + Archive (Forecast does
not exist that far back).
"""

from bwb.data.sources.climate.eto_alternatives import (
    compute_eto_benavides_lopez,
    compute_eto_hargreaves_samani,
)
from bwb.data.sources.climate.fusion import (
    FUSED_VARS,
    HIST_WEIGHTS_NASA,
    fuse_climate_sources,
)
from bwb.data.sources.climate.nasa_power import (
    compute_eto_fao56_pm,
    download_nasa_power,
    wind_10m_to_2m_fao56,
)
from bwb.data.sources.climate.openmeteo import (
    download_openmeteo_archive,
    download_openmeteo_forecast,
)

__all__ = [
    "download_nasa_power",
    "download_openmeteo_archive",
    "download_openmeteo_forecast",
    "fuse_climate_sources",
    "compute_eto_fao56_pm",
    "compute_eto_hargreaves_samani",
    "compute_eto_benavides_lopez",
    "wind_10m_to_2m_fao56",
    "HIST_WEIGHTS_NASA",
    "FUSED_VARS",
]
