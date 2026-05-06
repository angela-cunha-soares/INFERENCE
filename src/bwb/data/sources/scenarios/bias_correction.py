# src/bwb/data/sources/scenarios/bias_correction.py
import xarray as xr
from typing import Literal

Method = Literal["quantile_mapping", "linear_scaling", "isimip3b"]

class BiasCorrector:
    """Correção de viés CMIP6 contra baseline observacional."""
    
    def __init__(self, method: Method = "isimip3b"):
        self.method = method
    
    def fit(self, model_hist: xr.DataArray, obs_hist: xr.DataArray):
        """Ajusta função de transferência sobre período histórico comum."""
        ...
    
    def transform(self, model_future: xr.DataArray) -> xr.DataArray:
        """Aplica correção a dados de cenário futuro."""
        ...