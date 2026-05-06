"""
crop_loader.py
==============

Loader, schema validator e construtor de priors PyMC para a biblioteca
FAO-56 Table 12 de coeficientes de cultura (Kc).

Capacidades
-----------
1. Carrega e valida o JSON `fao56_table12_kc.json` via Pydantic v2.
2. Resolve valores escalares ou ranges {low, high} de forma uniforme.
3. Gera a curva Kc diária seguindo a interpolação linear FAO-56 (Eq. 66).
4. Constrói priors PyMC automaticamente a partir de cada cultura:
   - valor escalar → TruncatedNormal(mu=valor, sigma=...) com lower=0
   - range {low, high} → Uniform(low, high)

Uso típico
----------
    from crop_loader import load_fao56_table12, kc_daily_curve, make_kc_priors

    fao = load_fao56_table12("data/crops/fao56_table12_kc.json")
    soja = fao.get_crop("soybeans")

    # Curva determinística (média dos ranges)
    kc_curve = kc_daily_curve(
        kc_ini=soja.kc.ini.resolve(),
        kc_mid=soja.kc.mid.resolve(),
        kc_end=soja.kc.end.resolve(),
        L_ini=15, L_dev=15, L_mid=40, L_late=20,
    )

    # Priors bayesianos (dentro de pm.Model)
    import pymc as pm
    with pm.Model() as model:
        kc_priors = make_kc_priors(soja, sigma_scalar=0.05)
        # kc_priors["ini"], ["mid"], ["end"] são RVs do PyMC
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional, Union

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# ---------------------------------------------------------------------------
# Schema (Pydantic v2)
# ---------------------------------------------------------------------------


class Range(BaseModel):
    """Faixa de valores [low, high] da FAO-56 (e.g., Kc_mid = 1.15-1.20)."""

    model_config = ConfigDict(frozen=True)

    low: float
    high: float

    def resolve(self, mode: str = "mean") -> float:
        """Resolve para um único valor escalar.

        Parameters
        ----------
        mode : {'mean', 'low', 'high'}
            'mean' devolve a média; 'low'/'high' devolvem o respectivo extremo.
        """
        if mode == "mean":
            return (self.low + self.high) / 2.0
        if mode == "low":
            return self.low
        if mode == "high":
            return self.high
        raise ValueError(f"mode inválido: {mode!r}; use 'mean', 'low' ou 'high'.")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Range({self.low}-{self.high})"


def _parse_value_or_range(v):
    """Coerção robusta para float | Range | None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Range):
        return v
    if isinstance(v, dict) and "low" in v and "high" in v:
        return Range(low=v["low"], high=v["high"])
    raise ValueError(f"Não foi possível interpretar valor Kc/altura: {v!r}")


ValueOrRange = Annotated[
    Optional[Union[float, Range]], BeforeValidator(_parse_value_or_range)
]


def resolve(v: Optional[Union[float, Range]], mode: str = "mean") -> Optional[float]:
    """Resolve um valor escalar ou Range para float. None passa direto."""
    if v is None:
        return None
    if isinstance(v, Range):
        return v.resolve(mode=mode)
    return float(v)


def is_range(v) -> bool:
    """True se o valor é um Range; False para escalar ou None."""
    return isinstance(v, Range)


class KcCoefficients(BaseModel):
    """Coeficientes Kc nos três estágios canônicos da FAO-56."""

    model_config = ConfigDict(frozen=True)

    ini: ValueOrRange = None
    mid: ValueOrRange = None
    end: ValueOrRange = None


class Crop(BaseModel):
    """Entrada de uma cultura na Tabela 12 da FAO-56."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    scientific_name: Optional[str] = None
    category: str
    kc: KcCoefficients
    max_height_m: ValueOrRange = None
    notes: Optional[str] = None


class CategoryInfo(BaseModel):
    """Metadados de cada categoria FAO-56 (a-p)."""

    model_config = ConfigDict(frozen=True)

    name: str
    group_kc_ini: Optional[float] = None


class FAOCropLibrary(BaseModel):
    """Biblioteca completa carregada e validada do JSON FAO-56 Tabela 12."""

    model_config = ConfigDict(extra="allow")

    metadata: dict = Field(default_factory=dict)
    categories: dict[str, CategoryInfo] = Field(default_factory=dict)
    crops: dict[str, Crop]

    # --- API de consulta ---------------------------------------------------

    def get_crop(self, crop_id: str) -> Crop:
        """Retorna a entrada da cultura ou lança KeyError com sugestão."""
        if crop_id in self.crops:
            return self.crops[crop_id]
        available = sorted(self.crops.keys())
        sample = ", ".join(available[:8])
        raise KeyError(
            f"Cultura {crop_id!r} não encontrada na biblioteca FAO-56. "
            f"Total disponível: {len(available)}. Exemplos: {sample}, ..."
        )

    def list_by_category(self, category: str) -> list[str]:
        """Lista IDs de culturas em uma categoria FAO (e.g., 'e_legumes')."""
        return [k for k, v in self.crops.items() if v.category == category]

    def search(self, query: str) -> list[str]:
        """Busca textual em name e scientific_name (case-insensitive)."""
        q = query.lower()
        return [
            k
            for k, v in self.crops.items()
            if q in v.name.lower()
            or (v.scientific_name and q in v.scientific_name.lower())
            or q in k.lower()
        ]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_fao56_table12(path: Union[str, Path]) -> FAOCropLibrary:
    """Carrega e valida o JSON FAO-56 Tabela 12.

    Parameters
    ----------
    path : str | Path
        Caminho para `fao56_table12_kc.json`.

    Returns
    -------
    FAOCropLibrary
        Objeto validado. Acesso direto via `.crops['soybeans']` ou
        `.get_crop('soybeans')`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON FAO-56 não encontrado em: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return FAOCropLibrary.model_validate(data)


# ---------------------------------------------------------------------------
# Curva Kc diária (FAO-56 Eq. 66 - interpolação linear por estágio)
# ---------------------------------------------------------------------------


def kc_daily_curve(
    kc_ini: float,
    kc_mid: float,
    kc_end: float,
    L_ini: int,
    L_dev: int,
    L_mid: int,
    L_late: int,
) -> np.ndarray:
    """Gera série diária de Kc seguindo a FAO-56 (Eq. 66).

    Estágios:
        - initial    : Kc constante = Kc_ini  (duração L_ini dias)
        - development: ramp linear Kc_ini → Kc_mid (L_dev dias)
        - mid-season : Kc constante = Kc_mid  (L_mid dias)
        - late-season: ramp linear Kc_mid → Kc_end (L_late dias)

    Returns
    -------
    np.ndarray
        Vetor de comprimento L_ini + L_dev + L_mid + L_late.
    """
    if min(L_ini, L_dev, L_mid, L_late) < 0:
        raise ValueError("Durações de estágio não podem ser negativas.")

    n_days = L_ini + L_dev + L_mid + L_late
    if n_days == 0:
        return np.array([], dtype=float)

    kc = np.empty(n_days, dtype=float)

    # 1) Initial — constante
    kc[:L_ini] = kc_ini

    # 2) Development — ramp linear ini → mid
    if L_dev > 0:
        f = np.arange(1, L_dev + 1) / L_dev
        kc[L_ini : L_ini + L_dev] = kc_ini + f * (kc_mid - kc_ini)

    # 3) Mid-season — constante
    if L_mid > 0:
        s = L_ini + L_dev
        kc[s : s + L_mid] = kc_mid

    # 4) Late-season — ramp linear mid → end
    if L_late > 0:
        f = np.arange(1, L_late + 1) / L_late
        kc[L_ini + L_dev + L_mid :] = kc_mid + f * (kc_end - kc_mid)

    return kc


def kc_daily_from_crop(
    crop: Crop,
    L_ini: int,
    L_dev: int,
    L_mid: int,
    L_late: int,
    range_mode: str = "mean",
) -> np.ndarray:
    """Atalho: gera curva diária a partir de um objeto `Crop`."""
    return kc_daily_curve(
        kc_ini=resolve(crop.kc.ini, mode=range_mode),
        kc_mid=resolve(crop.kc.mid, mode=range_mode),
        kc_end=resolve(crop.kc.end, mode=range_mode),
        L_ini=L_ini,
        L_dev=L_dev,
        L_mid=L_mid,
        L_late=L_late,
    )


# ---------------------------------------------------------------------------
# Construtor de priors PyMC
# ---------------------------------------------------------------------------


def make_kc_priors(
    crop: Union[Crop, KcCoefficients],
    *,
    name_prefix: str = "Kc",
    sigma_scalar: float = 0.05,
    sigma_relative: bool = False,
    lower_bound: float = 0.0,
    upper_bound: Optional[float] = None,
) -> dict:
    """Constrói priors PyMC para Kc_ini, Kc_mid, Kc_end automaticamente.

    Regra:
        - valor escalar → TruncatedNormal(mu=valor, sigma, lower=lower_bound, upper=upper_bound)
        - range {low, high} → Uniform(low, high)

    Parameters
    ----------
    crop : Crop | KcCoefficients
        Objeto de cultura ou apenas o bloco Kc.
    name_prefix : str
        Prefixo para nomes das variáveis (ex.: 'Kc' → 'Kc_ini', 'Kc_mid', 'Kc_end').
        Use prefixos distintos se houver múltiplas culturas no mesmo modelo.
    sigma_scalar : float
        Desvio-padrão do prior Normal truncado para valores escalares.
        Se sigma_relative=True, é interpretado como fração da média.
    sigma_relative : bool
        Se True, sigma efetivo = mu * sigma_scalar (e.g., 0.10 = ±10% da média).
    lower_bound : float
        Truncamento inferior (default 0; Kc não pode ser negativo).
    upper_bound : float | None
        Truncamento superior opcional.

    Returns
    -------
    dict
        {'ini': RV, 'mid': RV, 'end': RV} — cada um pode ser None se a cultura
        não definir o valor (e.g., open water não tem Kc_ini).

    Notas
    -----
    Deve ser chamado dentro de um contexto `with pm.Model():`.
    """
    try:
        import pymc as pm
    except ImportError as e:
        raise ImportError(
            "PyMC não está instalado. Instale com: pip install pymc"
        ) from e

    kc = crop.kc if hasattr(crop, "kc") else crop

    priors: dict = {}
    for stage in ("ini", "mid", "end"):
        v = getattr(kc, stage)
        if v is None:
            priors[stage] = None
            continue

        var_name = f"{name_prefix}_{stage}"

        if isinstance(v, Range):
            priors[stage] = pm.Uniform(var_name, lower=v.low, upper=v.high)
        else:
            mu = float(v)
            sigma = mu * sigma_scalar if sigma_relative else sigma_scalar
            priors[stage] = pm.TruncatedNormal(
                var_name,
                mu=mu,
                sigma=sigma,
                lower=lower_bound,
                upper=upper_bound,
            )

    return priors


# ---------------------------------------------------------------------------
# Demo / smoke test (executável)
# ---------------------------------------------------------------------------


def _demo(json_path: Path) -> None:
    """Demonstração de uso. Roda quando o módulo é executado diretamente."""
    print("=" * 70)
    print(" Demo crop_loader — FAO-56 Table 12")
    print("=" * 70)

    fao = load_fao56_table12(json_path)
    print(f"\nBiblioteca carregada: {len(fao.crops)} culturas, "
          f"{len(fao.categories)} categorias")
    print(f"Versão: {fao.metadata.get('version')}")

    # 1) Acesso direto a uma cultura
    print("\n--- Soja (FAO-56 Table 12) ---")
    soja = fao.get_crop("soybeans")
    print(f"Nome           : {soja.name}")
    print(f"Espécie        : {soja.scientific_name}")
    print(f"Categoria      : {soja.category}")
    print(f"Kc_ini         : {resolve(soja.kc.ini)}")
    print(f"Kc_mid         : {resolve(soja.kc.mid)}")
    print(f"Kc_end         : {resolve(soja.kc.end)}")
    print(f"Altura máxima  : {resolve(soja.max_height_m)} m "
          f"(range? {is_range(soja.max_height_m)})")

    # 2) Cultura com range (algodão)
    print("\n--- Algodão (com Kc_mid em range) ---")
    cot = fao.get_crop("cotton")
    print(f"Kc_mid raw     : {cot.kc.mid}")
    print(f"Kc_mid mean    : {resolve(cot.kc.mid, mode='mean'):.3f}")
    print(f"Kc_mid low     : {resolve(cot.kc.mid, mode='low'):.3f}")
    print(f"Kc_mid high    : {resolve(cot.kc.mid, mode='high'):.3f}")

    # 3) Curva diária para soja precoce (90 dias)
    print("\n--- Curva Kc diária: soja precoce 90 dias ---")
    print("    Estágios: 15 ini + 15 dev + 40 mid + 20 late")
    curve = kc_daily_from_crop(soja, L_ini=15, L_dev=15, L_mid=40, L_late=20)
    print(f"    Total de dias        : {len(curve)}")
    print(f"    Kc no dia 1          : {curve[0]:.3f}  (= Kc_ini)")
    print(f"    Kc no dia 15         : {curve[14]:.3f}  (final do initial)")
    print(f"    Kc no dia 30         : {curve[29]:.3f}  (final do development)")
    print(f"    Kc no dia 50         : {curve[49]:.3f}  (meio do mid-season)")
    print(f"    Kc no dia 70         : {curve[69]:.3f}  (final do mid-season)")
    print(f"    Kc no dia 90         : {curve[89]:.3f}  (final do late = Kc_end)")
    print(f"    Kc médio (integrado) : {curve.mean():.3f}")

    # 4) Busca e listagem
    print("\n--- Busca textual: 'maize' ---")
    for hit in fao.search("maize"):
        c = fao.get_crop(hit)
        print(f"    {hit:35s} ({c.scientific_name})")

    print("\n--- Categoria e_legumes ---")
    for crop_id in fao.list_by_category("e_legumes"):
        print(f"    {crop_id}")

    # 5) Geração de priors PyMC (opcional — pula se PyMC não instalado)
    print("\n--- Priors PyMC para soja ---")
    try:
        import pymc as pm

        with pm.Model() as model:
            priors = make_kc_priors(soja, sigma_scalar=0.05)
            print(f"    Kc_ini prior : {priors['ini']}")
            print(f"    Kc_mid prior : {priors['mid']}")
            print(f"    Kc_end prior : {priors['end']}")
        print("    Modelo PyMC montado com sucesso.")
    except ImportError:
        print("    (PyMC não instalado — pulando esta seção)")
    except Exception as e:
        print(f"    Aviso: erro ao montar priors: {e}")

    print("\nDemo concluído.")


if __name__ == "__main__":
    import sys

    default_path = Path("data/crops/fao56_table12_kc.json")
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    elif default_path.exists():
        json_path = default_path
    else:
        # tentativa: mesmo diretório do script
        json_path = Path(__file__).parent / "fao56_table12_kc.json"

    _demo(json_path)