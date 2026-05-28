"""
build_ml_dataset_v5.py
----------------------
Transforma varios CSVs de MintEDGE en un único dataset homogéneo para ML.

Cambios respecto a v4:
  - Targets globales ampliados: añadidos satisfied_rate y rejection_rate.
  - Targets POR SERVICIO: unsatisfied_rate_{svc} y satisfied_rate_{svc}
    para cada servicio detectado en el dataset (t+1 y t+5).
  - Corrección satisfied_rate_{svc}: denominador corregido a requests+rejected.
  - Features temporales reducidas a 9 columnas clave con mayor granularidad:
    lags [1,2,4,6,8,10], rolling [2,4,6,8,10], diffs [1,2,4,6,8,10].

Uso:
    python build_ml_dataset_v5.py --input data/raw/results_maastricht_kpn_2500car_500people_s1.csv
                                           data/raw/results_elburg_kpn_2000car_500people_s1.csv
                                  --output data/processed/dataset_v4_maastricht_elburg.csv

Argumentos opcionales:
    --delay_p95_threshold FLOAT   Umbral de delay_p95_all para target de clasificación.
                                  Si no se pasa, se usa el percentil 95 del dataset completo.
    --keep_rows_without_target    Si se pasa, se conservan las últimas filas sin target futuro.
                                  Por defecto se eliminan.
"""

import argparse
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path


# ===========================================================================
# CONFIGURACIÓN DE TARGETS
# ===========================================================================
# Cambiar CREATE_TARGETS a False para desactivar la creación de targets.
CREATE_TARGETS = True

# Horizontes temporales (en segundos hacia el futuro).
# Para añadir más, agregar valores a la lista: [1, 5, 10, 30]
TARGET_HORIZONS = [1, 5]

# Targets de regresión GLOBALES a crear.
TARGETS_TO_CREATE = [
    "unsatisfied_rate_t_plus_{h}",       # tasa de peticiones que incumplen QoS
    "satisfied_rate_over_total_t_plus_{h}",    # tasa satisfechas sobre total peticiones
    "satisfied_rate_over_accepted_t_plus_{h}", # tasa satisfechas sobre peticiones aceptadas
    "qos_failure_rate_t_plus_{h}",       # tasa global de fallo (rechazadas + insatisfechas)
    "rejection_rate_t_plus_{h}",         # tasa de rechazo (ACTIVADO v4)
    "delay_p95_all_t_plus_{h}",          # latencia alta representativa
    "max_delay_p95_all_t_plus_{h}",      # peor delay representativo
]

# Targets POR SERVICIO (NUEVO v4).
# Se generan para cada servicio detectado en el dataset.
# Columnas fuente: unsatisfied_rate_{svc} y satisfied_rate_{svc}.
TARGETS_PER_SERVICE = [
    "unsatisfied_rate_{svc}_t_plus_{h}",
    "satisfied_rate_over_total_{svc}_t_plus_{h}",
    "satisfied_rate_over_accepted_{svc}_t_plus_{h}",
]

# Target de clasificación (desactivado).
CREATE_QOS_DEGRADATION_TARGET = False

UNSATISFIED_RATE_THRESHOLD = 0.01
REJECTION_RATE_THRESHOLD   = 0.01
# DELAY_P95_THRESHOLD: si None, se calcula como percentil 95 del dataset completo.
DELAY_P95_THRESHOLD = None

# Columnas base para features temporales.
# v4: se mantienen las 14 de v3 y se añaden requests_per_server y W_links (pedidas por Gabriel).
TEMPORAL_FEATURE_COLS = [
    "total_requests",
    "active_users",
    "rejection_rate",
    "unsatisfied_rate",
    "qos_failure_rate",
    "satisfied_rate_over_total",
    "satisfied_rate_over_accepted",
    "server_util_max",
    "delay_p95_all",
    "max_delay_all",
    "max_delay_p95_all",
    "delay_mean_weighted_all",
    "congestion_index",
    "server_pressure_index",
    "backhaul_pressure",
    "requests_per_server",   # NUEVO v4
    "W_links",               # NUEVO v4
]

# Mayor granularidad temporal que v3.
LAG_STEPS    = [1, 2, 4, 6, 8, 10]
ROLL_WINDOWS = [5, 10, 30]
DIFF_STEPS = [1, 5, 10]

# Valor con el que rellenar NaN al inicio de cada ciudad en features temporales.
TEMPORAL_NAN_FILL = 0


# ===========================================================================
# 1. ARGUMENTOS
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Construye un dataset homogéneo para ML a partir de CSVs de MintEDGE."
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="CSV(s) de entrada."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path del CSV de salida."
    )
    parser.add_argument(
        "--delay_p95_threshold", type=float, default=None,
        help=(
            "Umbral de delay_p95_all para el target de clasificación. "
            "Si no se pasa, se usa el percentil 95 del dataset completo."
        )
    )
    parser.add_argument(
        "--keep_rows_without_target", action="store_true",
        help="Si se pasa, conserva las últimas filas de cada ciudad sin target futuro."
    )
    return parser.parse_args()


# ===========================================================================
# 2. UTILIDADES
# ===========================================================================

def extract_city_id(filename: str) -> str:
    """
    Extrae el city_id del nombre del fichero.
    Ejemplo: results_maastricht_s1.csv -> maastricht
    Patrón: results_<city>_s<seed>.csv
    """
    stem = Path(filename).stem
    match = re.match(r"results?_(.+?)_s\d+", stem, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback: usar el nombre del fichero sin extensión
    warnings.warn(f"No se pudo extraer city_id de '{filename}'. Usando el nombre completo.")
    return stem.lower()


def detect_separator(filepath: str) -> str:
    """Detecta si el CSV usa coma o punto y coma."""
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def detect_services(df: pd.DataFrame) -> list:
    """Detecta los servicios presentes en el CSV mediante las columnas requests_BSx_<service>."""
    services = set()
    pattern = re.compile(r"^requests_BS\d+_(.+)$")
    for col in df.columns:
        m = pattern.match(col)
        if m:
            services.add(m.group(1))
    return sorted(services)


def detect_base_stations(df: pd.DataFrame) -> set:
    """
    Detecta los IDs de estaciones base presentes en el CSV.
    Busca en columnas: requests_BSx_*, rejected_req_BSx_*, delay_BSx_*, max_delay_BSx_*
    """
    bs_ids = set()
    pattern = re.compile(r"(?:requests|rejected_req|delay|max_delay)_BS(\d+)_")
    for col in df.columns:
        m = pattern.match(col)
        if m:
            bs_ids.add(int(m.group(1)))
    return bs_ids


def detect_server_columns(df: pd.DataFrame) -> list:
    """Devuelve las columnas server_util_BSx presentes en el CSV."""
    return [c for c in df.columns if re.match(r"^server_util_BS\d+$", c)]


def safe_divide(numerator, denominator):
    """División segura: devuelve 0 donde el denominador es 0 o NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (denominator == 0) | np.isnan(denominator),
            0.0,
            numerator / denominator
        )
    return result


# ===========================================================================
# 3-13. CONSTRUCCIÓN DE FEATURES
# ===========================================================================

def add_basic_features(df: pd.DataFrame, city_id: str,
                        bs_ids: set, server_cols: list) -> pd.DataFrame:
    """Bloque 1: identificación, tiempo y estructura de la ciudad."""
    out = pd.DataFrame()
    out["city_id"]  = pd.array([city_id] * len(df), dtype=object)
    out["time"]     = df["time"].values
    out["time_sin"] = np.sin(2 * np.pi * df["time"].values / 86400)
    out["time_cos"] = np.cos(2 * np.pi * df["time"].values / 86400)

    num_bs      = len(bs_ids)
    num_servers = len(server_cols)

    out["num_base_stations"] = num_bs
    out["num_servers"]       = num_servers
    out["server_ratio"]      = safe_divide(num_servers, num_bs)

    # Métricas globales originales
    for col in ["active_users", "total_requests", "total_rejected",
                "dynamic_W_servers", "idle_W_servers", "W_links"]:
        out[col] = df[col].values if col in df.columns else 0.0

    return out


def add_global_derived_features(out: pd.DataFrame) -> pd.DataFrame:
    """Bloque 2: métricas globales derivadas."""
    tr  = out["total_requests"].values
    taj = out["total_rejected"].values
    au  = out["active_users"].values
    nb  = out["num_base_stations"].values
    ns  = out["num_servers"].values
    dw  = out["dynamic_W_servers"].values
    iw  = out["idle_W_servers"].values
    wl  = out["W_links"].values

    total_W = dw + iw + wl

    out["total_W"]              = total_W
    out["requests_per_user"]    = safe_divide(tr, au)
    out["requests_per_bs"]      = safe_divide(tr, nb)
    out["requests_per_server"]  = safe_divide(tr, ns)
    out["rejection_rate"]       = safe_divide(taj, tr)
    out["W_per_request"]        = safe_divide(total_W, tr)
    return out


def add_service_request_features(df: pd.DataFrame, out: pd.DataFrame,
                                  services: list) -> pd.DataFrame:
    """Bloque 3a: agregaciones de requests por servicio."""
    tr = out["total_requests"].values
    for svc in services:
        cols = [c for c in df.columns if re.match(rf"^requests_BS\d+_{re.escape(svc)}$", c)]
        if not cols:
            warnings.warn(f"Sin columnas requests_BS*_{svc}. Columnas puestas a 0.")
            for stat in ["total","mean","std","max","p95","share"]:
                out[f"requests_{svc}_{stat}"] = 0.0
            continue
        sub = df[cols]
        total = sub.sum(axis=1).values
        out[f"requests_{svc}_total"] = total
        out[f"requests_{svc}_mean"]  = sub.mean(axis=1).values
        out[f"requests_{svc}_std"]   = sub.std(axis=1, ddof=0).values
        out[f"requests_{svc}_max"]   = sub.max(axis=1).values
        out[f"requests_{svc}_p95"]   = np.nanpercentile(sub.values, 95, axis=1)
        out[f"requests_{svc}_share"] = safe_divide(total, tr)
    return out


def add_service_rejection_features(df: pd.DataFrame, out: pd.DataFrame,
                                    services: list) -> pd.DataFrame:
    """Bloque 3b: agregaciones de rejected_req por servicio."""
    for svc in services:
        cols = [c for c in df.columns if re.match(rf"^rejected_req_BS\d+_{re.escape(svc)}$", c)]
        if not cols:
            warnings.warn(f"Sin columnas rejected_req_BS*_{svc}. Columnas puestas a 0.")
            for stat in ["total","mean","std","max","p95"]:
                out[f"rejected_{svc}_{stat}"] = 0.0
            out[f"rejection_rate_{svc}"] = 0.0
            continue
        sub   = df[cols]
        total = sub.sum(axis=1).values
        req_total = out[f"requests_{svc}_total"].values if f"requests_{svc}_total" in out else np.ones(len(out))
        out[f"rejected_{svc}_total"]     = total
        out[f"rejected_{svc}_mean"]      = sub.mean(axis=1).values
        out[f"rejected_{svc}_std"]       = sub.std(axis=1, ddof=0).values
        out[f"rejected_{svc}_max"]       = sub.max(axis=1).values
        out[f"rejected_{svc}_p95"]       = np.nanpercentile(sub.values, 95, axis=1)
        out[f"rejection_rate_{svc}"]     = safe_divide(total, req_total)
    return out


def add_service_delay_features(df: pd.DataFrame, out: pd.DataFrame,
                                services: list) -> pd.DataFrame:
    """Bloque 3c-d: agregaciones de delay y max_delay por servicio + hot_bs_ratio."""
    for metric in ["delay", "max_delay"]:
        for svc in services:
            cols = [c for c in df.columns
                    if re.match(rf"^{metric}_BS\d+_{re.escape(svc)}$", c)]
            if not cols:
                warnings.warn(f"Sin columnas {metric}_BS*_{svc}. Columnas puestas a 0.")
                for stat in ["mean","std","max","p95"]:
                    out[f"{metric}_{svc}_{stat}"] = 0.0
                continue
            sub = df[cols]
            out[f"{metric}_{svc}_mean"] = sub.mean(axis=1).values
            out[f"{metric}_{svc}_std"]  = sub.std(axis=1, ddof=0).values
            out[f"{metric}_{svc}_max"]  = sub.max(axis=1).values
            out[f"{metric}_{svc}_p95"]  = np.nanpercentile(sub.values, 95, axis=1)

    # hot_bs_ratio_{svc}: fracción de BSs donde max_delay supera la media del servicio
    # Detecta hotspots localizados — BSs concretas tirando del máximo hacia arriba
    nb = out["num_base_stations"].values
    for svc in services:
        cols = [c for c in df.columns
                if re.match(rf"^max_delay_BS\d+_{re.escape(svc)}$", c)]
        if not cols:
            out[f"hot_bs_ratio_{svc}"] = 0.0
            continue
        sub = df[cols]
        svc_mean = sub.mean(axis=1).values.reshape(-1, 1)
        hot_count = (sub.values > svc_mean).sum(axis=1)
        out[f"hot_bs_ratio_{svc}"] = safe_divide(hot_count, nb)

    return out


def add_unsatisfied_features(df: pd.DataFrame, out: pd.DataFrame,
                              services: list) -> pd.DataFrame:
    """Bloque 3e: peticiones insatisfechas por servicio."""
    for svc in services:
        raw_col = f"unsatisf_req_{svc}"
        if raw_col not in df.columns:
            warnings.warn(f"Columna '{raw_col}' no encontrada. Puesta a 0.")
            out[f"unsatisfied_{svc}"]      = 0.0
            out[f"unsatisfied_rate_{svc}"] = 0.0
            continue
        val = df[raw_col].values
        req = out[f"requests_{svc}_total"].values if f"requests_{svc}_total" in out else np.ones(len(out))
        out[f"unsatisfied_{svc}"]      = val
        out[f"unsatisfied_rate_{svc}"] = safe_divide(val, req)
    return out


def add_global_qos_features(df: pd.DataFrame, out: pd.DataFrame,
                              services: list) -> pd.DataFrame:
    """Bloque 4: métricas globales de QoS."""
    tr = out["total_requests"].values

    # Totales de insatisfechas
    unsatisfied_cols = [f"unsatisfied_{svc}" for svc in services if f"unsatisfied_{svc}" in out]
    unsatisfied_total = out[unsatisfied_cols].sum(axis=1).values if unsatisfied_cols else np.zeros(len(out))
    out["unsatisfied_total"] = unsatisfied_total

    taj = out["total_rejected"].values

    # v8: alineacion temporal con pd.Series.shift(1).
    # unsatisfied_t pertenece a peticiones de t-1, por ello se usan tr_{t-1} y taj_{t-1}.
    tr_prev  = pd.Series(tr).shift(1).fillna(0).values.astype(float)
    taj_prev = pd.Series(taj).shift(1).fillna(0).values.astype(float)

    accepted_prev      = tr_prev - taj_prev
    satisfied_requests = np.clip(accepted_prev - unsatisfied_total, 0, None)

    out["satisfied_requests"] = satisfied_requests
    out["unsatisfied_rate"]   = np.clip(safe_divide(unsatisfied_total, tr_prev), 0, 1)
    out["satisfied_rate_over_total"]    = np.clip(safe_divide(satisfied_requests, tr_prev), 0, 1)
    accepted_prev_clipped = np.where(accepted_prev <= 0, np.nan, accepted_prev)
    out["satisfied_rate_over_accepted"] = np.clip(safe_divide(satisfied_requests, accepted_prev_clipped), 0, 1)
    out["qos_failure_rate"]   = np.clip(safe_divide(taj_prev + unsatisfied_total, tr_prev), 0, 1)

    # Métricas globales de delay (fila a fila sobre todas las BSs y servicios)
    all_delay_cols     = [c for c in df.columns if re.match(r"^delay_BS\d+_", c)]
    all_max_delay_cols = [c for c in df.columns if re.match(r"^max_delay_BS\d+_", c)]

    if all_delay_cols:
        sub = df[all_delay_cols]
        out["delay_mean_all"] = sub.mean(axis=1).values
        out["delay_p95_all"]  = np.nanpercentile(sub.values, 95, axis=1)
        out["delay_max_all"]  = sub.max(axis=1).values
    else:
        warnings.warn("Sin columnas delay_BS*_*. delay_*_all puestas a 0.")
        out["delay_mean_all"] = 0.0
        out["delay_p95_all"]  = 0.0
        out["delay_max_all"]  = 0.0

    if all_max_delay_cols:
        sub = df[all_max_delay_cols]
        out["max_delay_all"]      = sub.max(axis=1).values
        out["max_delay_p95_all"]  = np.nanpercentile(sub.values, 95, axis=1)
    else:
        warnings.warn("Sin columnas max_delay_BS*_*. max_delay_*_all puestas a 0.")
        out["max_delay_all"]     = 0.0
        out["max_delay_p95_all"] = 0.0

    # Media ponderada global de delay
    try:
        weighted_vals = []
        for col in all_delay_cols:
            m = re.match(r"^delay_(BS\d+)_(.+)$", col)
            if not m:
                continue
            bs_id, svc = m.group(1), m.group(2)
            req_col = f"requests_{bs_id}_{svc}"
            if req_col not in df.columns:
                continue
            weighted_vals.append(df[col].values * df[req_col].values)
        total_req_all = df[all_delay_cols].shape[1]  # fallback
        if weighted_vals:
            numerator = np.sum(weighted_vals, axis=0)
            # denominador: suma de todos los requests usados como pesos
            req_weights = []
            for col in all_delay_cols:
                m = re.match(r"^delay_(BS\d+)_(.+)$", col)
                if not m:
                    continue
                req_col = f"requests_{m.group(1)}_{m.group(2)}"
                if req_col in df.columns:
                    req_weights.append(df[req_col].values)
            denominator = np.sum(req_weights, axis=0) if req_weights else np.ones(len(out))
            out["delay_mean_weighted_all"] = safe_divide(numerator, denominator)
        else:
            warnings.warn("No se pudo calcular delay_mean_weighted_all. Puesta a 0.")
            out["delay_mean_weighted_all"] = 0.0
    except Exception as e:
        warnings.warn(f"Error calculando delay_mean_weighted_all: {e}. Puesta a 0.")
        out["delay_mean_weighted_all"] = 0.0

    # Media ponderada por servicio
    for svc in services:
        try:
            delay_cols = [c for c in df.columns if re.match(rf"^delay_BS\d+_{re.escape(svc)}$", c)]
            req_cols   = [c for c in df.columns if re.match(rf"^requests_BS\d+_{re.escape(svc)}$", c)]
            if delay_cols and req_cols:
                numerator   = sum(df[d].values * df[r].values for d, r in zip(delay_cols, req_cols))
                denominator = sum(df[r].values for r in req_cols)
                out[f"delay_{svc}_weighted_mean"] = safe_divide(numerator, denominator)
            else:
                warnings.warn(f"No se pudo calcular delay_{svc}_weighted_mean. Puesta a 0.")
                out[f"delay_{svc}_weighted_mean"] = 0.0
        except Exception as e:
            warnings.warn(f"Error en delay_{svc}_weighted_mean: {e}. Puesta a 0.")
            out[f"delay_{svc}_weighted_mean"] = 0.0

    # satisfied_rate por servicio (v8: shift() + clamp)
    for svc in services:
        req_svc = out[f"requests_{svc}_total"].values  if f"requests_{svc}_total"  in out.columns else np.ones(len(out))
        rej_svc = out[f"rejected_{svc}_total"].values  if f"rejected_{svc}_total"  in out.columns else np.zeros(len(out))
        uns_svc = out[f"unsatisfied_{svc}"].values     if f"unsatisfied_{svc}"     in out.columns else np.zeros(len(out))
        req_svc_prev = pd.Series(req_svc).shift(1).fillna(0).values.astype(float)
        rej_svc_prev = pd.Series(rej_svc).shift(1).fillna(0).values.astype(float)
        accepted_svc_prev = req_svc_prev - rej_svc_prev
        satisfied_svc     = np.clip(accepted_svc_prev - uns_svc, 0, None)
        out[f"satisfied_rate_over_total_{svc}"]    = np.clip(safe_divide(satisfied_svc, req_svc_prev), 0, 1)
        accepted_svc_prev_clipped = np.where(accepted_svc_prev <= 0, np.nan, accepted_svc_prev)
        out[f"satisfied_rate_over_accepted_{svc}"] = np.clip(safe_divide(satisfied_svc, accepted_svc_prev_clipped), 0, 1)

    return out


def add_server_util_features(df: pd.DataFrame, out: pd.DataFrame,
                              server_cols: list) -> pd.DataFrame:
    """Bloque 5: utilización de servidores."""
    ns = out["num_servers"].values

    if not server_cols:
        warnings.warn("Sin columnas server_util_BS*. Métricas de servidor puestas a 0.")
        for col in ["server_util_mean","server_util_std","server_util_max","server_util_p90",
                    "servers_over_80","servers_over_90","share_servers_over_80","share_servers_over_90"]:
            out[col] = 0.0
        return out

    sub = df[server_cols]
    out["server_util_mean"]  = sub.mean(axis=1).values
    out["server_util_std"]   = sub.std(axis=1, ddof=0).values
    out["server_util_max"]   = sub.max(axis=1).values
    out["server_util_p90"]   = np.nanpercentile(sub.values, 90, axis=1)

    over_80 = (sub > 0.80).sum(axis=1).values
    over_90 = (sub > 0.90).sum(axis=1).values
    out["share_servers_over_80"]  = safe_divide(over_80, ns)
    out["share_servers_over_90"]  = safe_divide(over_90, ns)
    return out


def add_derived_interaction_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Bloque de variables derivadas de interacción.
    Se calculan a partir de columnas ya existentes en out.
    No requieren el CSV crudo — solo columnas ya agregadas.
    """
    su_max = out["server_util_max"].values
    su_mean = out["server_util_mean"].values
    rps    = out["requests_per_server"].values
    rr     = out["rejection_rate"].values
    dp95   = out["delay_p95_all"].values
    wl     = out["W_links"].values
    tr     = out["total_requests"].values
    dw     = out["dynamic_W_servers"].values
    tw     = out["total_W"].values

    # Margen restante hasta saturación del servidor más cargado
    out["capacity_margin"]         = 1.0 - su_max

    # Desequilibrio de carga: diferencia entre servidor más cargado y la media
    out["load_imbalance"]          = su_max - su_mean

    # Presión por servidor: peticiones × utilización
    out["server_pressure_index"]   = safe_divide(rps * su_max, 1.0)

    # Saturación y delay ocurriendo a la vez
    out["server_delay_interaction"] = su_max * dp95

    # Índice de congestión: rechazos × servidores llenos
    out["congestion_index"]        = rr * su_max

    # Tráfico de backhaul por petición
    out["backhaul_pressure"]       = safe_divide(wl, tr)

    # Ratio backhaul / cómputo
    out["backhaul_to_compute_ratio"] = safe_divide(wl, dw)

    # Fracción del consumo total que va a backhaul
    out["backhaul_energy_share"]   = safe_divide(wl, tw)

    return out


def add_post_temporal_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Variables que dependen de los diffs temporales — se calculan DESPUÉS de add_temporal_features.
    delay_acceleration y unsatisfied_acceleration miden la segunda derivada.
    """
    out = out.sort_values(["city_id", "time"]).reset_index(drop=True).copy()

    # delay_acceleration: velocidad de cambio del delay (segunda derivada)
    if "max_delay_all_diff_1" in out.columns:
        out["delay_acceleration"] = (
            out.groupby("city_id")["max_delay_all_diff_1"]
            .diff(1).fillna(0).values
        )

    # unsatisfied_acceleration: velocidad de crecimiento de la insatisfacción
    if "unsatisfied_rate_diff_1" in out.columns:
        out["unsatisfied_acceleration"] = (
            out.groupby("city_id")["unsatisfied_rate_diff_1"]
            .diff(1).fillna(0).values
        )

    return out


def add_temporal_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Bloque 6: lags, rolling means y diffs.
    Se calculan DENTRO de cada city_id, ordenando por time.
    Nunca mezcla ciudades.
    """
    out = out.sort_values(["city_id", "time"]).reset_index(drop=True).copy()

    for col in TEMPORAL_FEATURE_COLS:
        if col not in out.columns:
            warnings.warn(f"Columna temporal '{col}' no encontrada. Se omite.")
            continue

        grp = out.groupby("city_id")[col]

        for lag in LAG_STEPS:
            out[f"{col}_lag_{lag}"] = grp.shift(lag).fillna(TEMPORAL_NAN_FILL).values

        for window in ROLL_WINDOWS:
            roll_col = f"{col}_roll_mean_{window}"
            roll_series = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
            out[roll_col] = roll_series.fillna(TEMPORAL_NAN_FILL)

        for diff in DIFF_STEPS:
            out[f"{col}_diff_{diff}"] = grp.diff(diff).fillna(TEMPORAL_NAN_FILL).values

    return out


# ===========================================================================
# 14. TARGETS OPCIONALES
# ===========================================================================

def add_targets(out: pd.DataFrame, delay_p95_threshold: float,
                keep_rows_without_target: bool, services: list = None) -> pd.DataFrame:
    """
    Bloque 7: targets opcionales.
    Se calculan DENTRO de cada city_id, ordenando por time.
    Los targets son valores FUTUROS (shift negativo).
    v4: añade targets por servicio (unsatisfied_rate_{svc} y satisfied_rate_{svc}).
    """
    if not CREATE_TARGETS:
        return out

    out = out.sort_values(["city_id", "time"]).reset_index(drop=True).copy()

    source_map = {
        "max_delay_all":          "max_delay_all",
        "delay_p95_all":          "delay_p95_all",
        "max_delay_p95_all":      "max_delay_p95_all",
        "unsatisfied_rate":       "unsatisfied_rate",
        "rejection_rate":         "rejection_rate",
        "qos_failure_rate":       "qos_failure_rate",
        "satisfied_rate_over_total":    "satisfied_rate_over_total",
        "satisfied_rate_over_accepted": "satisfied_rate_over_accepted",
        "delay_mean_weighted_all":"delay_mean_weighted_all",
    }

    created_targets = []

    # --- Targets globales ---
    for h in TARGET_HORIZONS:
        for target_template in TARGETS_TO_CREATE:
            target_name = target_template.format(h=h)
            source_key  = re.sub(r"_t_plus_\d+$", "", target_name)
            source_col  = source_map.get(source_key)

            if source_col is None or source_col not in out.columns:
                warnings.warn(f"Columna fuente '{source_col}' no encontrada para target '{target_name}'.")
                continue

            col_name = f"target_{target_name}"
            out[col_name] = out.groupby("city_id")[source_col].shift(-h)
            created_targets.append(col_name)

        # Target de clasificación (desactivado por defecto)
        if CREATE_QOS_DEGRADATION_TARGET:
            t_unsatisfied = f"target_unsatisfied_rate_t_plus_{h}"
            t_rejection   = f"target_rejection_rate_t_plus_{h}"
            t_delay_p95   = f"target_delay_p95_all_t_plus_{h}"

            if all(c in out.columns for c in [t_unsatisfied, t_rejection, t_delay_p95]):
                deg_col = f"target_qos_degradation_t_plus_{h}"
                out[deg_col] = (
                    (out[t_unsatisfied] > UNSATISFIED_RATE_THRESHOLD) |
                    (out[t_rejection]   > REJECTION_RATE_THRESHOLD)   |
                    (out[t_delay_p95]   > delay_p95_threshold)
                ).astype(float)
                mask_nan = out[t_delay_p95].isna()
                out.loc[mask_nan, deg_col] = np.nan
                created_targets.append(deg_col)
            else:
                warnings.warn(
                    f"No se puede crear target_qos_degradation_t_plus_{h}: "
                    "faltan targets de regresión previos."
                )

    # --- Targets por servicio (NUEVO v4) ---
    # Usar la lista de servicios recibida directamente (ya detectada en build_dataset).
    # No detectar desde columnas para evitar capturar features temporales derivadas.
    detected_services = sorted(services) if services else []

    if detected_services and TARGETS_PER_SERVICE:
        print(f"  Creando targets por servicio para: {detected_services}")
        for h in TARGET_HORIZONS:
            for template in TARGETS_PER_SERVICE:
                for svc in detected_services:
                    target_name = template.format(svc=svc, h=h)
                    # Columna fuente: quitar el sufijo _t_plus_N para obtener el nombre base
                    source_col = re.sub(r"_t_plus_\d+$", "", target_name)

                    if source_col not in out.columns:
                        warnings.warn(
                            f"Columna fuente '{source_col}' no encontrada para "
                            f"target '{target_name}'. Se omite."
                        )
                        continue

                    col_name = f"target_{target_name}"
                    out[col_name] = out.groupby("city_id")[source_col].shift(-h)
                    created_targets.append(col_name)

    # Eliminar filas sin target si se indica
    if not keep_rows_without_target and created_targets:
        target_cols_present = [c for c in created_targets if c in out.columns]
        if target_cols_present:
            mask = out[target_cols_present].isna().any(axis=1)
            n_dropped = mask.sum()
            out = out[~mask]
            print(f"  Filas eliminadas por falta de target futuro: {n_dropped}")

    return out, created_targets


# ===========================================================================
# 15-16. PROCESADO POR FICHERO Y DATASET COMPLETO
# ===========================================================================

def process_file(path: Path, all_services: list) -> tuple:
    """
    Procesa un único CSV de MintEDGE.
    Devuelve (DataFrame homogéneo, city_id, num_bs, num_servers, services).
    """
    city_id     = extract_city_id(path.name)
    sep         = detect_separator(str(path))
    df          = pd.read_csv(path, sep=sep)
    services    = detect_services(df)
    bs_ids      = detect_base_stations(df)
    server_cols = detect_server_columns(df)

    print(f"\n  Ciudad: {city_id}")
    print(f"    Filas: {len(df)} | Columnas originales: {len(df.columns)}")
    print(f"    Servicios: {services}")
    print(f"    Estaciones base: {len(bs_ids)} | Servidores: {len(server_cols)}")

    out = add_basic_features(df, city_id, bs_ids, server_cols)
    out = add_global_derived_features(out)

    # Usar la lista global de servicios para que todas las ciudades tengan las mismas columnas
    for svc in all_services:
        if svc not in services:
            warnings.warn(f"Servicio '{svc}' no encontrado en {city_id}. Columnas puestas a 0.")
    target_services = all_services

    out = add_service_request_features(df, out, target_services)
    out = add_service_rejection_features(df, out, target_services)
    out = add_service_delay_features(df, out, target_services)
    out = add_unsatisfied_features(df, out, target_services)
    out = add_global_qos_features(df, out, target_services)
    out = add_server_util_features(df, out, server_cols)
    out = add_derived_interaction_features(out)

    # Limpieza: infinitos → NaN → 0 (solo columnas numéricas)
    out = out.copy()  # desfragmentar
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return out, city_id, len(bs_ids), len(server_cols), services


def build_dataset(input: list, output: str, delay_p95_threshold: float,
                  keep_rows_without_target: bool):
    """Pipeline completo: lee todos los CSVs, construye y guarda el dataset."""

    csv_files = [Path(p) for p in input]
    missing = [p for p in csv_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Ficheros no encontrados: {missing}")

    print(f"\n{'='*60}")
    print(f"CSVs encontrados: {len(csv_files)}")
    for f in csv_files:
        print(f"  {f.name}")

    # Primera pasada: detectar todos los servicios posibles entre todas las ciudades
    all_services = set()
    for f in csv_files:
        sep = detect_separator(str(f))
        df_tmp = pd.read_csv(f, sep=sep, nrows=1)
        all_services |= set(detect_services(df_tmp))
    all_services = sorted(all_services)
    print(f"\nServicios detectados globalmente: {all_services}")

    # Segunda pasada: procesar cada fichero
    dfs = []
    summary = {}
    for f in csv_files:
        df_city, city_id, num_bs, num_srv, svcs = process_file(f, all_services)
        summary[city_id] = {"num_bs": num_bs, "num_servers": num_srv, "services": svcs}
        dfs.append(df_city)

    # Concatenar y añadir features temporales
    print("\nConcatenando datasets...")
    combined = pd.concat(dfs, ignore_index=True)
    combined = add_temporal_features(combined)
    combined = add_post_temporal_features(combined)

    temporal_features_created = []
    for col in TEMPORAL_FEATURE_COLS:
        if col in combined.columns:
            for lag in LAG_STEPS:
                temporal_features_created.append(f"{col}_lag_{lag}")
            for w in ROLL_WINDOWS:
                temporal_features_created.append(f"{col}_roll_mean_{w}")
            for d in DIFF_STEPS:
                temporal_features_created.append(f"{col}_diff_{d}")

    # Calcular umbral de delay_p95 si no se pasó
    if delay_p95_threshold is None:
        delay_p95_threshold = combined["delay_p95_all"].quantile(0.95)
        print(f"\nUmbral delay_p95_all calculado automáticamente (p95): {delay_p95_threshold:.6f}")
    else:
        print(f"\nUmbral delay_p95_all recibido por argumento: {delay_p95_threshold:.6f}")

    # Targets
    created_targets = []
    if CREATE_TARGETS:
        print("Creando targets...")
        combined, created_targets = add_targets(combined, delay_p95_threshold, keep_rows_without_target, all_services)

    # Orden final: por city_id y time
    combined = combined.sort_values(["city_id", "time"]).reset_index(drop=True)

    # Guardar
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"  CSVs procesados:      {len(csv_files)}")
    print(f"  Ciudades detectadas:  {list(summary.keys())}")
    print(f"  Servicios detectados: {all_services}")
    for city, info in summary.items():
        print(f"  {city}: {info['num_bs']} BSs, {info['num_servers']} servidores")
    print(f"  Filas totales:        {len(combined)}")
    print(f"  Columnas totales:     {len(combined.columns)}")
    print(f"  Features temporales:  {len(temporal_features_created)}")
    print(f"  Targets creados:      {created_targets}")
    print(f"  Dataset guardado en:  {output}")
    print(f"{'='*60}\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args = parse_args()

    # Sobreescribir umbral desde argumento si se pasó
    global DELAY_P95_THRESHOLD
    if args.delay_p95_threshold is not None:
        DELAY_P95_THRESHOLD = args.delay_p95_threshold

    build_dataset(
        input=args.input,
        output=args.output,
        delay_p95_threshold=DELAY_P95_THRESHOLD,
        keep_rows_without_target=args.keep_rows_without_target,
    )


if __name__ == "__main__":
    main()
