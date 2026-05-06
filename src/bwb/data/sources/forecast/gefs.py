# src/bwb/data/sources/forecast/gefs.py
import xarray as xr

GEFS_AWS = "s3://noaa-gefs-pds"   # bucket AWS público

class GEFSEnsemble:
    """31-member GEFS ensemble, 0.25°, lead time 1-16 days."""
    
    def get_forecast(self, lat: float, lon: float, 
                     variables: list[str], lead_days: int = 5) -> xr.Dataset:
        # retorna (ensemble_member, time, var)
        ...