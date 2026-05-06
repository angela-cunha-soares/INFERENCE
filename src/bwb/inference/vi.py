# src/bwb/inference/vi.py
import pymc as pm
import arviz as az
from typing import Literal

class VariationalInference:
    """Fast-path para preview interativo (sub-segundo)."""
    
    def __init__(self, method: Literal["advi", "fullrank_advi", "svgd"] = "advi"):
        self.method = method
    
    def fit(self, model: pm.Model, n_iter: int = 30_000) -> az.InferenceData:
        with model:
            approx = pm.fit(n=n_iter, method=self.method, progressbar=False)
            idata = approx.sample(2000)
        return idata