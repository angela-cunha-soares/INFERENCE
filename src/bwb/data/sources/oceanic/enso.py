# src/bwb/data/sources/oceanic/enso.py
import pandas as pd
from pathlib import Path

ONI_URL = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"

class OceanicIndex:
    """Adapter unificado para índices oceânicos da NOAA."""
    
    def fetch(self, index: str, cache_dir: Path) -> pd.Series:
        cache = cache_dir / f"{index}.parquet"
        if cache.exists() and self._is_fresh(cache, days=30):
            return pd.read_parquet(cache)["value"]
        # download + parse específico de cada índice
        ...