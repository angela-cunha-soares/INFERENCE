# src/bwb/inference/ppc.py
import arviz as az
import pymc as pm

def run_ppc(model: pm.Model, idata: az.InferenceData, 
            var_names: list[str]) -> az.InferenceData:
    with model:
        idata.extend(pm.sample_posterior_predictive(idata, var_names=var_names))
    return idata

def ppc_summary(idata: az.InferenceData, observed_var: str) -> dict:
    """Retorna métricas de calibração: bayesian p-value, T-stat, etc."""
    ...