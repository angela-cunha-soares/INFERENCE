"""Smoke test that the public API imports cleanly."""

from __future__ import annotations


def test_bwb_imports():
    import bwb
    assert bwb.__version__


def test_validation_module_imports():
    from bwb.validation import (
        bias, compute_all_metrics, coverage, crps_ensemble, crps_mean,
        interval_score, kge, mae, nse, pbias, pit, pit_alpha_reliability,
        rmse,
        ValidationConfig, create_validation_config, generate_validation_grid,
        run_single_validation, run_validation_pipeline,
        SensitivityConfig, generate_saltelli_samples, compute_sobol_indices,
    )
    assert callable(kge)
    assert callable(crps_mean)


def test_data_module_imports():
    from bwb.data.loaders import (
        load_balsas_historical, list_available_cities, load_city_series,
        extract_crop_cycle, load_oceanic_indices, load_climatological_priors,
    )
    from bwb.data.adapters import (
        build_water_balance_inputs, run_deterministic_baseline, build_kc_curve,
    )
    assert callable(load_city_series)
    assert callable(build_water_balance_inputs)


def test_config_module_imports():
    from bwb.config.profiles import load_profile, list_profiles
    from bwb.config.settings import get_settings
    assert "matopiba" in list_profiles()
    assert get_settings().region == "matopiba"


def test_forecast_module_imports():
    from bwb.forecast.ensemble import (
        PosteriorParameters, propagate_ensemble, ensemble_quantiles,
        decision_from_ensemble, synthetic_perturbed_ensemble,
        posterior_from_idata,
    )
    assert callable(propagate_ensemble)


def test_models_module_imports():
    from bwb.models.water_balance import (
        WaterBalanceInputs, build_model, fit_model, apply_decision_rule,
        diagnostics_summary,
    )
    from bwb.models.van_genuchten import (
        SoilParameters, water_retention, get_soil_parameters,
        compute_field_capacity, compute_wilting_point,
    )
    assert callable(build_model)
    assert callable(water_retention)
