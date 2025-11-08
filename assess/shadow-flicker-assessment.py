# -*- coding: utf-8 -*-
"""
Shadow Flicker – WORST + REALISTIC
Amb DEM screening (Numba), raster i receptors, dies afectats i minuts/dia.
Inclou: CSV #META, timing, curtailment mensual, Weibull vent, reponderació direccional (mitjana=1),
SOL per turbina (opcional), tall 10×D, tolerància azimutal.
"""
import os, math, time, yaml, argparse, sys, json, csv, datetime as dt
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
#from matplotlib.colors import LightSource
import rasterio
from rasterio.transform import Affine
import rasterio.transform
from dataclasses import dataclass
from shapely.geometry import Point, box, Polygon, MultiPolygon, shape
from typing import Dict, Any, List, Tuple
from functools import lru_cache
from zoneinfo import ZoneInfo
import pandas as pd

try:
    from numba import njit, prange
except Exception:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap
    def prange(*args, **kwargs):
        return range(*args)  # fallback segur
    
cfg = None

def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def norm_name(s): 
    return str(s).strip().upper()

def make_time_index_utc(year: int, step_min: int = 1) -> pd.DatetimeIndex:
    """Index continu en UTC per a tot l'any [inici, any+1)."""
    return pd.date_range(
        start=pd.Timestamp(year=year,  month=1, day=1, tz="UTC"),
        end=pd.Timestamp(year=year+1, month=1, day=1, tz="UTC"),
        freq=f"{step_min}min",
        inclusive="left",
    )    
    
def ensure_output_dir(output_dir: str | None) -> str | None:
    """
    Si s'especifica output_dir, assegura que existeix (mkdir -p).
    Retorna la ruta normalitzada o None si no s'ha indicat.
    """
    if output_dir:
        out = os.path.abspath(output_dir)
        os.makedirs(out, exist_ok=True)        
        return out
    return None

@dataclass
class Config:
    """Lleugera 'wrapper' per a la configuració carregada del YAML.
    Manté compatibilitat: seguim assignant globals com abans, però retornem l'objecte per traçabilitat/tests.
    """
    data: dict
        
def load_config(yaml_args, g: dict):
    """
    Carrega el YAML i reassigna seccions a variables globals per compatibilitat
    amb codi existent (globals()).
    - WIND_WEIBULL, CURTAIL_MONTH_PROFILES, AVAIL_MONTH_PROFILES, SUNSHINE_FRAC_MONTH_PROFILES
    - WIND_DIR_MONTH, WIND_ROSE
    - XMIN/YMIN/XMAX/YMAX (domain)
    - YEAR, SITE_LAT/LON, DEM_PATH
    - MIN_ELEV, MIN_GHI, ANNUAL_LIMIT_ASTR/REAL, DAILY_LIMIT_MIN
    - TURBINES, TURBINE_BY_ID
    - RECEPTORS, RECEPTOR_BY_ID
    Retorna també l’objecte cfg estructurat.
    """    
    try:
        with open(yaml_args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(f"[ERROR] Fitxer de configuració no trobat: {path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"[ERROR] Error al parsejar el YAML {path}: {e}")

                  
    # ---- VENT / WEIBULL / ROSA / DIRECCIONS ----
    if "wind_weibull" in cfg:
        g["WIND_WEIBULL"] = {int(m): (float(v["k"]), float(v["c"])) for m, v in cfg["wind_weibull"].items()}

    if "curtail_month_profiles" in cfg:
        g["CURTAIL_MONTH_PROFILES"] = cfg["curtail_month_profiles"]

    if "avail_month_profiles" in cfg:
        g["AVAIL_MONTH_PROFILES"] = cfg["avail_month_profiles"]

    if "sunshine_frac_profiles" in cfg:
        g["SUNSHINE_FRAC_MONTH_PROFILES"] = cfg["sunshine_frac_profiles"]

    if "wind" in cfg:
        w = cfg["wind"]
        if "monthly_dir" in w:
            # claus de mes → int
            g["WIND_DIR_MONTH"] = {int(m): {k: float(v) for k, v in sect.items()}
                                   for m, sect in w["monthly_dir_rose"].items()}
        if "wind_rose" in w:
            g["WIND_ROSE"] = list(w["wind_rose"])

    # ---- DOMINI ----
    if "domain" in cfg:
        d = cfg["domain"]
        if "bbox" in d and isinstance(d["bbox"], (list, tuple)) and len(d["bbox"]) == 4:
            g["XMIN"], g["YMIN"], g["XMAX"], g["YMAX"] = map(float, d["bbox"])
        else:
            g["XMIN"] = float(d["xmin"]); g["YMIN"] = float(d["ymin"])
            g["XMAX"] = float(d["xmax"]); g["YMAX"] = float(d["ymax"])

    # ---- PROJECTE ----
    if "project" in cfg:
        g["YEAR"] = int(cfg["project"].get("year", g.get("YEAR", 0)))
        g["PROJECT"] = cfg["project"].get("name")
        g["EXPORT_SUFFIX"] = cfg["project"].get("export")        
        g["ESIA"] = cfg["project"].get("esia")
        g["SITE_LAT"] = _num(cfg["project"].get("site_lat"))
        g["SITE_LON"] = _num(cfg["project"].get("site_lon"))
        g["DEM_PATH"] = cfg["project"].get("dem_path")
        g["OUTPUT_DIR"] = cfg["project"].get("output_dir")

    # ---- SHADOW FLICKER / LLINDARS ----
    if "shadow_flicker" in cfg:
        sf = cfg["shadow_flicker"]
        g["MIN_ELEV"] = _num(sf.get("min_elev_deg"), 3.0)
        g["MIN_GHI"]  = _num(sf.get("min_ghi_wm2"), 120.0)
        g["ANNUAL_LIMIT_ASTR"] = _num(sf.get("annual_limit_hours_astr"), 30.0)
        g["ANNUAL_LIMIT_REAL"] = _num(sf.get("annual_limit_hours_real"), 8.0)
        g["DAILY_LIMIT_MIN"]   = _num(sf.get("daily_limit_min"), 30.0)
        # (Opcional) altres paràmetres que tinguis al bloc:
        if (yaml_args.fast and (yaml_args.fast.upper() == "Y" or yaml_args.fast.upper() == "YES")):
            g["GRID_STEP_M"]  = 25.0
            g["H_REC_RASTER"] = 4.0
            g["TIME_STEP_MIN"] = 5.0
            g["RASTER_TOL_M"] = 1.0 
            g["MAX_CHECKS_STEP"] = 4000
            g["N_SAMP_PER_KM"] = 60
            g["TOL_AZDEG"] = 10
        else:
            g["GRID_STEP_M"]  = _num(sf.get("grid_step_m"), g.get("GRID_STEP_M"))
            g["H_REC_RASTER"] = _num(sf.get("h_rec_raster_m"), g.get("H_REC_RASTER"))
            g["TIME_STEP_MIN"] = _num(sf.get("time_step_min"), g.get("TIME_STEP_MIN"))
            g["RASTER_TOL_M"] = _num(sf.get("raster_tol_m"), g.get("RASTER_TOL_M"))
            g["MAX_CHECKS_STEP"] = _num(sf.get("max_checks_step"), g.get("MAX_CHECKS_STEP"))
            g["N_SAMP_PER_KM"] = _num(sf.get("n_samp_per_km"), g.get("N_SAMP_PER_KM"))
            g["TOL_AZDEG"] = _num(sf.get("tol_azdeg"), g.get("TOL_AZDEG"))

    # ---- TURBINES (llista i índex per id) ----
    legacy = []
    turbine_by_id = {}
    for t in cfg.get("turbines", []):
        tid   = norm_name(str(t["id"]))
        x     = _num(t.get("x"))
        y     = _num(t.get("y"))
        lat   = _num(t.get("lat"))
        lon   = _num(t.get("lon"))        
        plat_h   = _num(t.get("plat_h"))
        hub   = _num(t.get("hub_height_m"))
        rotor = _num(t.get("rotor_diameter_m"))
        shade = _num(t.get("shade_corr"), 0.48)   # default if not provided
        cutin = _num(t.get("cut_in"), 3.0)
        cutout= _num(t.get("cut_out"), 25.0)
        model = t.get("model", "")
        p_MW  = _num(t.get("power_MW"))

        # Optional platform offset kept separately
        #turbine_by_id[tid] = _num(t.get("platform_offset_m"), 2.0)

        # Build the legacy tuple (order matters!)
        legacy.append((tid, x, y, lat, lon, plat_h, hub, rotor, shade, cutin, cutout, model, p_MW))    

    g["TURBINES"] = legacy
    #g["TURBINE_BY_ID"] = {t["id"]: t for t in turbines_out}

    # ---- RECEPTORS (llista i índex per id) ----
    legacy = []
    receptor_by_id = {}
    for t in cfg.get("receptors", []):
        id   = str(t["id"])
        town = str(t.get("town"))
        x     = _num(t.get("x"))
        y     = _num(t.get("y"))
        height_m   = _num(t.get("height_m"))

        # Optional platform offset kept separately
        #receptor_by_id[tid] = _num(t.get("platform_offset_m"), 2.0)

        # Build the legacy tuple (order matters!)
        legacy.append((town, id,  x, y, height_m))    
    
    g["RPTS"] = legacy
    
    return cfg
        

# ---------- (1) CONFIG BÀSICA ----------
YEAR = 2025
PROJECT = ""
ESIA = ""
EXPORT_SUFFIX = ""
OUTPUT_DIR = "SCENES"
TZ_LOCAL = ZoneInfo("Europe/Madrid")
# ------------ LLINDARS -----------------
# Llindars (bones pràctiques europees)
MAX_MIN_PER_DAY = 30.0  # minuts/dia (p.ex. criteri alemany)

# Àmbit cartogràfic (UTM)
XMIN, YMIN, XMAX, YMAX = 344500, 4595500, 353500, 4600500

# SOL per turbina (opcional)
SITE_LAT, SITE_LON = 41.515244, 1.194816

# Turbines: [(name, xUTM, yUTM, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power MW)]
# hub_h: we add 2m from ground level as a save assumption for platform
# shade_corr provides possibility to adapt rotor_shade_h = hub + (rotor_d * shade_corr)
# Otherwise we would need more technical data about the dimensions of the wings to calculate the shading factor
# as the flicker potential is lower for the extremes of the rotor wings

TURBINE_DATA_STRUCT= [("name", "xUTM", "yUTM", "lat", "lon", "plat_h", "hub_h", "rotor_d", "shade_corr", "cut_in", "cut_out", "model", "powerMW")]

TURBINES = []
#    ("YA3", 347327, 4598444, 41.52286141, 1.170478, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
#    ("Y09", 348363, 4598715, 41.525559, 1.182837, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
#    ("Y05", 348718, 4598000, 41.519005, 1.186957, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
#    ("Y06", 349377, 4597537, 41.515277, 1.194854, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
#    ("Y07", 350011, 4597190, 41.512064, 1.202922, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
#    ("Y8B", 350526, 4596948, 41.510136, 1.209102, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
#]

# Caché local per horitzons (independent a cada procés)
HORIZON_CACHE = {}

def get_horizon_for(tname, tx, ty, rotor_shade_h):
    key = norm_name(tname)
    H = HORIZON_CACHE.get(key)
    if H is None:
        H = build_horizon_table(tx, ty, DEM_Z, TA, TB, TC, TD, TE, TF, rotor_shade_h=rotor_shade_h)
        HORIZON_CACHE[key] = H
    return H

USE_PER_TURBINE_SUNPOS = True

# Receptors: (Nucli, ReceptorID, X, Y, h_rec_m)
# Posa tants receptors per nucli com vulguis
RPTS = [
    ("La Pobla de Ferran", "PBF_1", 349013, 4599960, 2.0),
    ("Passanant",          "PAS_1", 349455, 4599284, 2.0),
    ("Glorieta",           "GLO_1", 350170, 4598024, 2.0),
    #("Glorieta",           "GLO_2", 350206, 4597984, 4.0),
    ("La Sala de Comalats","SAL_1", 351251, 4597853, 2.0),
    ("Belltall",           "BEL_1", 348542, 4596638, 2.0),
]

# DEM
DEM_PATH = "DEM_clip_5m-rpglobal.tif"   # DEM 5 m existent

# Paràmetres geomètrics turbin a
#HUB_H   = 112.0        # altura del buc (m)
#ROTOR_D = 172.0        # diàmetre (m)
#ROTOR_R = ROTOR_D/2.0

# Tall físic shadow flicker
MAX_SF_DIST = None  # calculat després com rotor_d*10

# Resolució temporal i espacial
TIME_STEP_MIN = 10.0            # puja a 5–10 min si vols calcular més ràpid per proves, però cal fixar-ho en 1 (1 minut) per càlculs exactes
GRID_STEP_M   = 50.0           # pas del raster XY, 25 millor, 50 per proves
H_REC_RASTER  = 2.0            # alçada del receptor “raster” (mapa d’envolvent)
RASTER_TOL_M       = 0.5     # tolerància d'excedència (impuresa) DEM per línia SOL→píxel (m)
MAX_CHECKS_STEP = 4000
N_SAMP_PER_KM = 48
# No cal mirar per elevació per sota de 3º
MIN_ELEV = 3.0
TOL_AZDEG = 10

# Procurar més concordància de resultats de receptors i terrain screening
REC_PIX_RING = 2 # per fer un screening extra al voltant dels receptors pel terrain screening

# Garantir concordància amb resultats 
BLOCK_MIN_LEN_M = 25.0 # llargada mínima de bloqueig

R_EARTH = 6371000.0
K_FACTOR = 4.0/3.0
R_EFF = R_EARTH * K_FACTOR


# Precalc of turbine horizons
HORIZON = {}
# Tolerància azimutal (Sol entre turbina i receptor)
# TOL_AZ_DEG = 11.0 available for SCENE CONFIG now

# Direcció de vent (reponderació, mitjana = 1)
USE_DIR_REWEIGHT = True   # si True, aplica rosa del vent per realistic mensual amb mitjana 1

#NSECT = 8  # N,NE,E,SE,S,SW,W,NW
# --- Direcció del vent COM A PROBABILITAT CONDICIONADA ---
# IMPORTANT: aquestes taules han de sumar 1.0 cada mes.
# Són p(sector | v in [CUT_IN, CUT_OUT]) — ja condicionades a velocitat operable.
# No hem d'aplicar factor WEIBULL adicional sería ponderar el vent dues vegades i un doble comptatge subestimant el factor vent
# Exemple de sectors (N, NE, E, ...), percentatges que sumen 1 per mes
# (aproximació local... caldría mirar si podem millorar els valors per la rosa dels vents que condiciona el yaw (el giro) del rotor)
WIND_DIR_MONTH = {
    1: {"N":0.15,"NE":0.08,"E":0.05,"SE":0.08,"S":0.10,"SW":0.15,"W":0.21,"NW":0.18},
    2: {"N":0.15,"NE":0.08,"E":0.05,"SE":0.08,"S":0.10,"SW":0.15,"W":0.21,"NW":0.18},
    3: {"N":0.12,"NE":0.07,"E":0.06,"SE":0.09,"S":0.12,"SW":0.17,"W":0.22,"NW":0.15},
    4: {"N":0.12,"NE":0.07,"E":0.06,"SE":0.09,"S":0.13,"SW":0.17,"W":0.21,"NW":0.15},
    5: {"N":0.11,"NE":0.06,"E":0.06,"SE":0.09,"S":0.15,"SW":0.20,"W":0.19,"NW":0.14},
    6: {"N":0.10,"NE":0.06,"E":0.08,"SE":0.12,"S":0.16,"SW":0.22,"W":0.16,"NW":0.10},
    7: {"N":0.10,"NE":0.06,"E":0.08,"SE":0.12,"S":0.16,"SW":0.22,"W":0.16,"NW":0.10},
    8: {"N":0.10,"NE":0.06,"E":0.08,"SE":0.12,"S":0.16,"SW":0.22,"W":0.16,"NW":0.10},
    9: {"N":0.11,"NE":0.08,"E":0.09,"SE":0.11,"S":0.15,"SW":0.20,"W":0.14,"NW":0.12},
    10: {"N":0.12,"NE":0.09,"E":0.10,"SE":0.11,"S":0.14,"SW":0.19,"W":0.13,"NW":0.12},
    11: {"N":0.13,"NE":0.08,"E":0.07,"SE":0.09,"S":0.12,"SW":0.17,"W":0.19,"NW":0.15},
    12: {"N":0.15,"NE":0.08,"E":0.05,"SE":0.08,"S":0.10,"SW":0.15,"W":0.21,"NW":0.18}    
}

WIND_ROSE = ["N","NE","E","SE","S","SW","W","NW"]

def sector_of_az(az_deg):
    idx = int(((az_deg % 360) + 22.5) // 45) % 8
    #idx_flipped = int(((az_deg % 360) + 202.5) // 45) % 8
    return WIND_ROSE[idx]

def sector_index(s: str) -> int:
    return WIND_ROSE.index(s)

def sector_by_index(i: int) -> str:
    return WIND_ROSE[i % 8]

def opposite_sector(s: str) -> str:
    return sector_by_index(sector_index(s) + 4)

def left_sector(s: str) -> str:
    return sector_by_index(sector_index(s) - 1)

def right_sector(s: str) -> str:
    return sector_by_index(sector_index(s) + 1)

def wind_dir_month_basic(month: int, sector: str) -> float:
    pm = WIND_DIR_MONTH.get(month, {})
    if not pm:
        return 1.0/8.0
    s = sum(pm.values())
    if s <= 0:
        return 1.0/8.0
    return max(0.0, pm.get(sector, 0.0) / s)

# wind dir establishes rotor position (yaw) intercepting sun in relation to receivers
# Adjectent, wind orientations are ponderated into the equation with a factor of 0.35 of their value
# this allows us to consider shadow flicker even if the rotor is not perfectly exposed and intercepting 
def wind_dir_yaw(month: int, sector: str, alpha_adj: float = 0.35, clamp: bool = True) -> float:
    # Prob{yaw adient per flicker} amb simetria 180° i esmorteïment dels veïns.
    s0  = sector
    s0L = left_sector(s0);  s0R = right_sector(s0)
    sO  = opposite_sector(s0)
    sOL = left_sector(sO);  sOR = right_sector(sO)

    p = (
        wind_dir_month_basic(month, s0) +
        wind_dir_month_basic(month, sO) +
        alpha_adj * (
            wind_dir_month_basic(month, s0L) + wind_dir_month_basic(month, s0R) +
            wind_dir_month_basic(month, sOL) + wind_dir_month_basic(month, sOR)
        )
    )
    if clamp:
        p = min(1.0, max(0.0, p))
    return p


# SUNSHINE_FRAC_MONTH = {m: 0.6 for m in range(1,13)}  # fracció d’hores de sol efectives
SUNSHINE_FRAC_MONTH_PROFILES = {  # aprox. inland català i apujant una mica perquè influeix molt la costa aquí per a realistic
    "WORST":   {m:1.0  for m in range(1,13)},
    #"REALISTIC": {m:0.6 for m in range(1,13)},
    "REALISTIC": {1:0.52, 2:0.52, 3:0.55, 4:0.60, 5:0.70, 6:0.75, 7:0.80, 8:0.80, 9:0.70,10:0.65,11:0.55,12:0.52}
}


def sunshinefrac_fn_factory(profile):
    def fn(m): return float(profile.get(m, 0.98))
    return fn


def avail_fn_factory(profile):
    def fn(m): return float(profile.get(m, 0.98))
    return fn

AVAIL_MONTH_PROFILES = {
    "WORST":   {m:1.0  for m in range(1,13)},
    "REALISTIC": {m:0.975 for m in range(1,13)}, # vent, gairebé sempre n'hi ha 2.5% està bé
}

def curtail_fn_factory(profile):
    maint = profile.get("maint", {})
    ree   = profile.get("ree", {})
    def fn(m):
        fm = 1.0 - float(maint.get(m,0.0))
        fr = 1.0 - float(ree.get(m,0.0))
        return max(0.0, min(1.0, fm*fr))
    return fn

# ree includes all external factors, curtailments by REE and curtailments by other external resason like bird-life curtailments
# maint includes all internal factors... 
# AVAIL_MONTH_PROFILES is not used as maint should count up for all internal reasons of curtailment
# Curtailment mensual (fracció de temps *retallat* 0..1). Es transformen a factor (1 - fracció).
CURTAIL_MONTH_PROFILES = {
    "WORST":   {"maint":{m:0.0 for m in range(1,13)}, "ree":{m:0.0 for m in range(1,13)}},
    "REALISTIC": {
        "maint": {1:0.0,2:0.0,3:0.02,4:0.02,5:0.02,6:0.02,7:0.02,8:0.02,9:0.02,10:0.02,11:0.02,12:0.0}, # prop de 2% pel conjunt està bé
        "ree":   {1:0.01,2:0.01,3:0.015,4:0.015,5:0.02,6:0.03,7:0.03,8:0.03,9:0.02,10:0.01,11:0.01,12:0.01} # més per estiu, menys a l'hivern
    },
}

# Weibull vent (per mesos, si vols)
WIND_WEIBULL_NO = {
    # k (shape), c (scale m/s) a altura del buc
    m: {"k":2.0, "c":7.5} for m in range(1,13)
}

# Weibull mensual (exemple; posa valors locals si en tens)
# k = shape, c = scale (m/s)
WIND_WEIBULL = {
    1:(2.0, 7.3), 2:(2.0, 7.6), 3:(2.0, 7.8), 4:(2.0, 8.1),
    5:(2.0, 8.2), 6:(2.0, 8.0), 7:(2.0, 7.6), 8:(2.0, 7.5),
    9:(2.0, 7.4),10:(2.0, 7.7),11:(2.0, 7.6),12:(2.0, 7.3),
}

def weibull_cdf(v, k, c):  # F(v)
    return 1.0 - np.exp(-(v/c)**k)

def wind_oper_prob_month(m, cut_in, cut_out):
    k, c = WIND_WEIBULL[m]
    # Probabilitat d’estar en servei: v ∈ [cut-in, cut-out]
    # we only need to know if we have enough wind for the turbine running
    weibull_weighted = max(0.0, min(1.0, (weibull_cdf(cut_out, k, c) - weibull_cdf(cut_in, k, c))))
    if weibull_weighted > 0.0:
        return 1
    else:
        return 0

def solar_pos_utc(dt_utc, lat_deg, lon_deg):
    """
    dt_utc: datetime timezone-aware en UTC (o naive assumint UTC).
    Retorna (elev_deg, azim_deg) on azimut 0°=N, 90°=E, 180°=S, 270°=W.
    Precisió suficient per a càlculs de shadow-flicker i geometria solar.
    """

    # --- 1) Julian Date (UTC) ---
    # Si reps naive, assumeix UTC
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=dt.timezone.utc)
    dt_utc = dt_utc.astimezone(dt.timezone.utc)

    Y = dt_utc.year
    M = dt_utc.month
    D = dt_utc.day + (dt_utc.hour + (dt_utc.minute + dt_utc.second/60)/60)/24

    if M <= 2:
        Y -= 1
        M += 12

    A = math.floor(Y/100)
    B = 2 - A + math.floor(A/4)

    JD = math.floor(365.25*(Y + 4716)) + math.floor(30.6001*(M + 1)) + D + B - 1524.5
    T = (JD - 2451545.0)/36525.0  # segles julians des de J2000.0

    # --- 2) GMST en graus (IAU aproximat) ---
    GMST = (280.46061837 +
            360.98564736629*(JD - 2451545.0) +
            0.000387933*T*T -
            (T*T*T)/38710000.0) % 360.0

    # --- 3) LST (deg) ---
    LST = (GMST + lon_deg) % 360.0

    # --- 4) Posició solar eclíptica -> RA/Dec ---
    # Longitud mitjana i anomalia (deg)
    L = (280.460 + 0.9856474*(JD - 2451545.0)) % 360.0
    g = math.radians((357.528 + 0.9856003*(JD - 2451545.0)) % 360.0)

    # Longitud eclíptica aparent (rad)
    lam = math.radians(L + 1.915*math.sin(g) + 0.020*math.sin(2*g))
    # Obliqüitat (rad)
    eps = math.radians(23.439 - 0.0000004*(JD - 2451545.0))

    # Ascensió recta (alpha) i declinació (delta_s)
    alpha = math.atan2(math.cos(eps)*math.sin(lam), math.cos(lam))  # rad, -pi..pi
    delta_s = math.asin(math.sin(eps)*math.sin(lam))                # rad

    # --- 5) Angle horari H (rad, -pi..pi) ---
    alpha_deg = (math.degrees(alpha) % 360.0)
    H_deg = ((LST - alpha_deg + 540.0) % 360.0) - 180.0  # -180..180
    H = math.radians(H_deg)

    # --- 6) Altitud i azimut ---
    lat = math.radians(lat_deg)

    sin_alt = math.sin(lat)*math.sin(delta_s) + math.cos(lat)*math.cos(delta_s)*math.cos(H)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)

    # Azimut amb atan2 (0=N cap a E)
    y = -math.sin(H)*math.cos(delta_s)
    x =  math.sin(delta_s)*math.cos(lat) - math.cos(delta_s)*math.sin(lat)*math.cos(H)
    az = math.atan2(y, x)  # -pi..pi, 0 = Nord
    az_deg = (math.degrees(az) + 360.0) % 360.0

    elev_deg = math.degrees(alt)
    return elev_deg, az_deg

# ---------- (3) DEM: lectura i affine ----------
# ---------------------- DEM & utilitats UTM ----------------------
with rasterio.open(DEM_PATH) as dem:
    Z = dem.read(1).astype("float32")
    T: Affine = dem.transform
    BOUNDS = dem.bounds
    XRES, YRES = T.a, -T.e

def sample_dem(x, y):
    col = (x - T.c)/T.a
    row = (y - T.f)/T.e
    r0 = np.floor(row).astype(int); c0 = np.floor(col).astype(int)
    r1 = r0+1; c1 = c0+1
    r0c = np.clip(r0, 0, Z.shape[0]-1); r1c = np.clip(r1, 0, Z.shape[0]-1)
    c0c = np.clip(c0, 0, Z.shape[1]-1); c1c = np.clip(c1, 0, Z.shape[1]-1)
    fr = row - r0; fc = col - c0
    z00 = Z[r0c, c0c]; z10 = Z[r1c, c0c]; z01 = Z[r0c, c1c]; z11 = Z[r1c, c1c]
    z0 = z00*(1-fr) + z10*fr
    z1 = z01*(1-fr) + z11*fr
    return z0*(1-fc) + z1*fc

# """"""
def load_dem(dem_path):
    ds = rasterio.open(dem_path)
    Z = ds.read(1).astype("float32")
    T = ds.transform
    bounds = ds.bounds
    # IMPORTANT: treballarem amb espaiats POSITIUS en metres
    xres = abs(T.a)
    yres = abs(T.e)
    return ds, Z, T, bounds, xres, yres

def make_hillshade(Z, xres, yres, az_deg=315.0, alt_deg=45.0, valleys_light=True):
    # Hillshade normalitzat [0..1] amb nord amunt.
    # - Usem gradients amb espaiats POSITIUS (cap inversió de signes).
    # - Plotjarem després amb origin='upper' perquè la primera fila del raster
    #   ja correspon al nord geogràfic (top del raster).
    # Gradients en coordenades de mapa (metres/píxel positius)
    gy, gx = np.gradient(Z, yres, xres)

    # Càlcul clàssic de hillshade (variant de Horn)
    slope  = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)  # convenció per azimut 0=N, 90=E

    az  = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)

    hs = np.sin(alt)*np.cos(slope) + np.cos(alt)*np.sin(slope)*np.cos(az - aspect)
    hs = (hs - np.nanmin(hs)) / (np.nanmax(hs) - np.nanmin(hs) + 1e-12)

    # Valls clares / carenes fosques (ja OK així; si vols invertir, fes: 1.0 - hs)
    return hs if valleys_light else (1.0 - hs)

def normalize01(A):
    a = A.astype("float32")
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)

# --- Carrega DEM una sola vegada
VALLEYS_LIGHT = True

# DEM_DS, DEM_Z, DEM_T, DEM_BOUNDS, DEM_XRES, DEM_YRES = load_dem(DEM_PATH)
# Constants de transformada (affine) per a Numba
# TA = float(DEM_T.a); TB = float(DEM_T.b); TC = float(DEM_T.c)
# TD = float(DEM_T.d); TE = float(DEM_T.e); TF = float(DEM_T.f)

# DEM_NORM  = normalize01(DEM_Z)
# DEM_SHADE = make_hillshade(DEM_Z, DEM_XRES, DEM_YRES, az_deg=315.0, alt_deg=45.0, valleys_light=VALLEYS_LIGHT)

# ----------- Intensitat lúminca ----------
# Cel clar: DNI ≈ 900, DHI ≈ 50 → cal sin ℎ ≥ (120−50) / 900 = 0.078 → h ≈ 4.5°
# Lleu calitja: DNI ≈ 700, DHI ≈ 30 → sin ℎ ≥ 0.129 → h ≈ 7.4°
# Terbolesa forta: DNI ≈ 500, DHI ≈ 20 → sin ℎ ≥ 0.20 → h ≈ 11.5°
# Tenim ja probabilitat de sol ponderat, podem posar cel clar
def elev_min_deg_for_ghi_watts(given_d: float, target_ghi: float = 120.0, dni: float = 900.0, dhi: float = 50.0) -> float:
    
    # Retorna l'elevació solar mínima (en graus) perquè GHI >= target_ghi,
    # assumint la descomposició: GHI ≈ DNI*sin(h) + DHI.

    # target_ghi: objectiu de GHI [W/m²], p.ex. 120
    # dni: Direct Normal Irradiance [W/m²]
    # dhi: Diffuse Horizontal Irradiance [W/m²] (opc.)
    
    # Si la difusa sola ja supera l’objectiu, amb elevació >0° ja ho tens
    if dhi >= target_ghi:
        return 0.0

    # Necessitem sin(h) >= (target_ghi - DHI)/DNI
    if dni <= 0:
        return float('inf')  # impossible sense component directa

    s = (target_ghi - dhi) / dni
    if s <= 0:
        return 0.0  # ja superes l'objectiu sense elevar gaire
    if s >= 1:
        return float('inf')  # ni al zenit

    if given_d >= math.degrees(math.asin(s)):
        return 1
    else:
        return 0

def dynamic_tol_azdeg(dist_h, rotor_d, az_drift_per_step_deg, k=1.2, min_deg=3.0, max_deg=12.0):
    # dist_h en metres; rotor_d en metres
    if dist_h <= 0:
        return max_deg
    half = 0.5 * rotor_d
    ratio = min(1.0, (k * half) / float(dist_h))
    base = math.degrees(math.asin(ratio)) if ratio > 0 else 0.0
    # Afegim mig de la deriva d’azimut del timestep (anti-aliasing)
    tol = base + 0.5 * max(0.0, float(az_drift_per_step_deg))
    return max(min_deg, min(max_deg, tol))

# ---------- (4) Numba screening ----------
def build_receptor_pixels_for_screening_on_grid(RPTS, xx, yy, ring_pix=1):
    """
    Retorna índexs locals (i,j) de la malla XX,YY al voltant dels receptors.
    ring_pix=1 → quadrat 3x3, ring_pix=2 → 5x5, etc.
    """
    pix = set()
    nx = xx.size
    ny = yy.size
    if nx == 0 or ny == 0:
        return np.empty((0,2), dtype=np.int64)

    # Suposem grid regular: pas constant
    dx = xx[1] - xx[0] if nx > 1 else 1.0
    dy = yy[1] - yy[0] if ny > 1 else 1.0

    for (nucli, rid, rx, ry, hrec) in RPTS:
        # Índex local aproximat
        j0 = int(round((rx - xx[0]) / dx))
        i0 = int(round((ry - yy[0]) / dy))

        if j0 < 0 or j0 >= nx or i0 < 0 or i0 >= ny:
            continue  # receptor fora de la malla

        for di in range(-ring_pix, ring_pix+1):
            for dj in range(-ring_pix, ring_pix+1):
                i = i0 + di
                j = j0 + dj
                if 0 <= i < ny and 0 <= j < nx:
                    pix.add((i, j))

    if not pix:
        return np.empty((0,2), dtype=np.int64)

    return np.array(sorted(pix), dtype=np.int64)  # (i_local, j_local)

# Globals (perquè els faràs servir des de raster_step_with_screening_numba)
REC_PIX_EXTRA = None

@njit(cache=True, fastmath=True)
def _dem_bilinear_scalar(Z, x, y, TA, TB, TC, TD, TE, TF):
    # Inversa afí (GDAL/rasterio): 
    # x = TA*col + TB*row + TC
    # y = TD*col + TE*row + TF
    dX = x - TC
    dY = y - TF
    det = TA*TE - TB*TD
    if det == 0.0 or abs(det) < 1e-20:
        return float('nan')
    col = ( dX*TE - dY*TB) / det
    row = (-dX*TD + dY*TA) / det

    nrows, ncols = Z.shape
    # clamp suau als marges, amb NN a la vora
    if not (0.0 <= col <= ncols-1 and 0.0 <= row <= nrows-1):
        # tolera errors numèrics de pocs nanòmetres
        eps = 1e-9
        if col < -eps or row < -eps or col > (ncols-1)+eps or row > (nrows-1)+eps:
            return float('nan')
        col = min(max(col, 0.0), ncols-1.0)
        row = min(max(row, 0.0), nrows-1.0)

    c0 = int(math.floor(col)); r0 = int(math.floor(row))
    if c0 == ncols-1 or r0 == nrows-1:
        return float(Z[r0, c0])  # nearest a la vora
    c1 = c0 + 1; r1 = r0 + 1
    tx = col - c0; ty = row - r0
    z00 = Z[r0, c0]; z10 = Z[r0, c1]
    z01 = Z[r1, c0]; z11 = Z[r1, c1]
    z0 = z00*(1.0-ty) + z01*ty
    z1 = z10*(1.0-ty) + z11*ty
    return float(z0*(1.0-tx) + z1*tx)

@njit(cache=True, fastmath=True)
def _terrain_screen_ok_path_raster(tx, ty, rotor_shade_h,
                                   px, py, h_rec_raster,
                                   DEM_Z, TA, TB, TC, TD, TE, TF,
                                   n_samp_per_km, tol_m):
    """
    LOS raster: fa servir el mateix criteri que los_clear_turbine_to_point
    dels receptors, però per a un píxel (px,py) i alçada h_rec_raster.
    """

    # Altura a la turbina
    z_t = _dem_bilinear_scalar(DEM_Z, tx, ty, TA, TB, TC, TD, TE, TF)
    if np.isnan(z_t):
        # fora del DEM → millor considerar-ho clar per no sobrebloquejar
        return True
    z_t += rotor_shade_h

    # Altura al píxel (receptor raster)
    z_p = _dem_bilinear_scalar(DEM_Z, px, py, TA, TB, TC, TD, TE, TF)
    if np.isnan(z_p):
        return True
    z_p += h_rec_raster

    # Mostreig de la línia
    samples = max(1, int(n_samp_per_km))
    step_m = 1000.0 / float(samples)
    tol = float(max(0.0, tol_m))

    # Ús del mateix motor LOS que pels receptors
    return _los_clear_tol_affine(tx, ty, z_t,
                                 px, py, z_p,
                                 DEM_Z, TA, TB, TC, TD, TE, TF,
                                 step_m, tol)

@njit(cache=True, fastmath=True, parallel=True)
def terrain_screen_ok_batch(tx, ty, rotor_shade_h, h_rec_raster,
                            rows, cols, XX, YY,
                            DEM_Z, TA, TB, TC, TD, TE, TF,
                            n_samp_per_km, tol_m):
    """
    Comprova LOS per un lot de píxels (rows, cols) de la malla XX,YY
    utilitzant el mateix criteri que als receptors.
    Retorna un array uint8 de 0/1.
    """
    n = rows.size
    out = np.ones(n, dtype=np.uint8)

    for k in range(n):
        i = rows[k]
        j = cols[k]
        px = XX[i, j]
        py = YY[i, j]

        if not _terrain_screen_ok_path_raster(tx, ty, rotor_shade_h,
                                              px, py, h_rec_raster,
                                              DEM_Z, TA, TB, TC, TD, TE, TF,
                                              n_samp_per_km, tol_m):
            out[k] = 0

    return out

# ---------- LOS robust amb tolerància ----------
@njit(cache=True, fastmath=True)
def _world_to_pixel(x, y, TA, TB, TC, TD, TE, TF):
    """
    Inversa de l'afí GDAL:
      X = TA + col*TB + row*TC
      Y = TD + col*TE + row*TF
    Retorna (col, row) en floats.
    """
    dX = x - TA
    dY = y - TD
    det = TB*TF - TC*TE
    if det == 0.0:
        return np.nan, np.nan
    col = ( dX*TF - dY*TC) / det
    row = (-dX*TE + dY*TB) / det
    return col, row

@njit(cache=True, fastmath=True)
def _bilinear_dem_affine(xi, yi, DEM_Z, TA, TB, TC, TD, TE, TF):
    """
    Interpolació bilinear directa en píxels, tolerant rotació.
    """
    col, row = _world_to_pixel(xi, yi, TA, TB, TC, TD, TE, TF)
    if not (col == col and row == row):  # NaN
        return np.nan

    c0 = int(np.floor(col)); r0 = int(np.floor(row))
    c1 = c0 + 1;             r1 = r0 + 1
    nrows, ncols = DEM_Z.shape

    if c0 < 0 or r0 < 0 or c1 >= ncols or r1 >= nrows:
        return np.nan

    tx = col - c0
    ty = row - r0
    z00 = DEM_Z[r0, c0]; z10 = DEM_Z[r0, c1]
    z01 = DEM_Z[r1, c0]; z11 = DEM_Z[r1, c1]
    return (1.0-ty)*((1.0-tx)*z00 + tx*z10) + ty*((1.0-tx)*z01 + tx*z11)

@njit(cache=True, fastmath=True)
def _los_clear_tol_affine(x0,y0,z0, x1,y1,z1,
                          DEM_Z, TA, TB, TC, TD, TE, TF,
                          step_m, tol):
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    dist = (dx*dx + dy*dy) ** 0.5
    if dist <= step_m:
        return True
    n = max(1, int(dist / step_m))
    for i in range(1, n):
        t = i / n
        xi = x0 + t*dx
        yi = y0 + t*dy
        zi = z0 + t*dz

        zdem = _bilinear_dem_affine(xi, yi, DEM_Z, TA, TB, TC, TD, TE, TF)
        if zdem == zdem and zdem > zi + tol:  # tolerància vertical
            return False
    return True

def _terrain_screen_ok_path(tx, ty, rx, ry, elev_t, hrec,
                            DEM_Z, TA, TB, TC, TD, TE, TF,
                            n_samp_per_km, RASTER_TOL_M):
    """
    - Suporta la rotació que tenim al DEM.
    - step_m = 1000 / n_samp_per_km
    - tol vertical = RASTER_TOL_M
    """
    # Ensure Z is float64 C-contiguous (Numba-friendly)
    DEM_Z = np.ascontiguousarray(Z, dtype=np.float64)

    # Ensure TA..TF are plain floats (not numpy scalars)
    TA = float(TA); TB = float(TB); TC = float(TC)
    TD = float(TD); TE = float(TE); TF = float(TF)

    # Quick sanity check on determinant (once)
    det = TB*TF - TC*TE
    if det == 0.0 or abs(det) < 1e-20:
        raise ValueError("Affine determinant is zero/near-zero; raster geotransform is degenerate.")
    
    samples = max(1, int(n_samp_per_km))
    step_m = 1000.0 / float(samples)
    tol = float(max(0.0, RASTER_TOL_M))

    return _los_clear_tol_affine(tx, ty, elev_t, rx, ry, hrec,
                                 DEM_Z, TA, TB, TC, TD, TE, TF,
                                 step_m, tol)


def draw_dem_hillshade(ax, alpha_hs=0.85, alpha_dem=0.45, cmap_dem='Greys'):    
    # Dibuixa el DEM en gris (suau) + Hillshade per sota de qualsevol capa.
    # - alpha_dem: opacitat del DEM base
    # - alpha_hs : opacitat del hillshade (llum de 315°/45° per defecte)
    ax.imshow(
        DEM_NORM, cmap="gray",
        extent=(DEM_BOUNDS.left, DEM_BOUNDS.right, DEM_BOUNDS.bottom, DEM_BOUNDS.top),
        origin="upper", alpha=0.55
    )
    # 2) Hillshade per sobre (volum)
    ax.imshow(
        DEM_SHADE, cmap="gray",
        extent=(DEM_BOUNDS.left, DEM_BOUNDS.right, DEM_BOUNDS.bottom, DEM_BOUNDS.top),
        origin="upper", alpha=0.80            
    )
    
# ---------- (5) Raster step (Numba) + timing ----------
ENABLE_TIMING = True
TARGET_MS_PER_TURB = 15.0
def _new_timing_bucket():
    return {"total_calls":0, "total_ms":0.0, "per_turb":{}, "per_step":[]}

def raster_step_with_screening_numba(XX, YY, acc_min, elev, az, w, dt_min, h_rec_raster,
                                     tx, ty, max_sf_dist, rotor_shade_h, rotor_d, rotor_tol_deg=8.0,
                                     n_samp_per_km=48, tol_m=0.5,
                                     max_checks_per_step=4000,
                                     timing_bucket=None, tname="TURB",
                                     throttle=True,
                                     daily_hit_mask=None):
    t0 = time.perf_counter()
    # fix error, this is not correct: DX = XX - tx; DY = YY - ty
    # Correct:
    DX = tx - XX
    DY = ty - YY    
    dist_h = np.hypot(DX, DY)
    #az_rt  = (np.degrees(np.arctan2(DX, DY)) % 360.0 + 360.0) % 360.0
    # ho mateix, més curt
    az_rt = (np.degrees(np.arctan2(DX, DY)) + 360.0) % 360.0

    L = (rotor_shade_h - h_rec_raster) / max(np.tan(np.radians(elev)), 1e-6)

    R = 0.5 * rotor_d
    ratio = np.clip(R / np.maximum(dist_h, 1e-6), 0.0, 1.0)
    theta_geom = np.degrees(np.arcsin(ratio))           # només geometria del disc

    # marge addicional opcional (configurable). Per WINDPRO-strict → 0.0
    theta_extra = rotor_tol_deg  # + 0.5*az_drift_per_step_deg
    theta_allow = theta_geom + theta_extra

    # diferència d’azimut mínima (−180..+180)
    daz = np.abs(((az_rt - az + 180.0) % 360.0) - 180.0)

    mask_cand = (
        (dist_h <= np.minimum(L, max_sf_dist)) &
        (daz <= theta_allow)
    )    
    
    rr = np.array([])   
    cc = np.array([])   
    
    idx = np.argwhere(mask_cand)
    if idx.size == 0:
        elapsed = (time.perf_counter() - t0)*1000.0
        return elapsed, max_checks_per_step, 0, rr, cc

    used_max = max_checks_per_step
    idx_sel = idx
    if idx.shape[0] > max_checks_per_step:
        sel = np.linspace(0, idx.shape[0]-1, max_checks_per_step).astype(np.int64)
        idx_sel = idx[sel]

    # fem un extra screening a prop de receptors si cal
    if REC_PIX_EXTRA is not None and REC_PIX_EXTRA.shape[0] > 0:
        s_cand = set((int(r), int(c)) for r, c in idx)        
        s_sel  = set((int(r), int(c)) for r, c in idx_sel)

        forced_pix = []
        for r, c in REC_PIX_EXTRA:
            key = (int(r), int(c))
            if key not in s_cand:
                continue         # no és candidat geomètric
            #if key in s_cand and key not in s_sel:
            if key not in s_sel:
                forced_pix.append(key)

        if forced_pix:
            forced_pix = np.array(forced_pix, dtype=np.int64)
            idx_sel = np.vstack([idx_sel, forced_pix])
            idx_sel = np.unique(idx_sel, axis=0)            
            print(f"[DEBUG] Extra receptor pixels afegits: {len(forced_pix)}")
            
        
    rows_sel = idx_sel[:,0].astype(np.int64)
    cols_sel = idx_sel[:,1].astype(np.int64)
    
    ok_mask = terrain_screen_ok_batch(
        float(tx), float(ty), float(rotor_shade_h), float(h_rec_raster),
        rows_sel, cols_sel, XX, YY,
        DEM_Z, TA, TB, TC, TD, TE, TF,
        int(n_samp_per_km), float(tol_m)
    )    

    updated = 0
    if np.any(ok_mask):
        rr = rows_sel[ok_mask.astype(bool)]
        cc = cols_sel[ok_mask.astype(bool)]
        # acc_min[rr, cc] += TIME_STEP_MIN * w        
        acc_min[rr, cc] += dt_min * w
        if daily_hit_mask is not None:
            daily_hit_mask[rr, cc] = True
        updated = rr.size

    elapsed = (time.perf_counter() - t0)*1000.0

    # Throttle simple
    if throttle and elapsed > TARGET_MS_PER_TURB and used_max > 1000:
        factor = TARGET_MS_PER_TURB / max(elapsed, 1e-3)
        factor = max(0.4, min(1.0, factor))
        used_max = max(1000, int(used_max * factor))

    if ENABLE_TIMING and timing_bucket is not None:
        timing_bucket["total_calls"] += 1
        timing_bucket["total_ms"] += elapsed
        d = timing_bucket["per_turb"].setdefault(tname, {"calls":0, "ms":0.0})
        d["calls"] += 1; d["ms"] += elapsed

    return elapsed, used_max, updated, rr, cc

# ---------- (6) CSV META ----------
def _profile_to_json_str(obj) -> str:
    try: return json.dumps(obj, ensure_ascii=False, separators=(",",":"))
    except Exception: return str(obj)

def write_csv_with_meta(path, header_cols, rows, scen_name):
    params = SCENARIOS[scen_name]

    sun_frac_fn = params["sun_frac"]
    avail_fn    = params["avail_fn"]
    curt_fn     = params["curt_fn"]
    use_screen  = params["terrain_screen"]
    use_dir     = params["use_dir"]
    tol_azdeg   = TOL_AZDEG #params["tol_azdeg"]
    
    meta = [
        ["#META","year", YEAR],
        ["#META","scenario", scen_name],
        ["#META","project", PROJECT],
        ["#META","esia", ESIA],
        ["#META","turbine_data_struct", _profile_to_json_str(TURBINE_DATA_STRUCT)],
        ["#META","turbines", _profile_to_json_str(TURBINES)],
        ["#META","receptors", _profile_to_json_str(RPTS)],
        ["#META","sunshine_frac_month", _profile_to_json_str(SUNSHINE_FRAC_MONTH_PROFILES.get(scen_name, {}))],
        ["#META","avail_month_profile", _profile_to_json_str(AVAIL_MONTH_PROFILES.get(scen_name, {}))],
        ["#META","curtail_month_profile", _profile_to_json_str(CURTAIL_MONTH_PROFILES.get(scen_name, {}))],
        ["#META","wind_weibull_kc", _profile_to_json_str(WIND_WEIBULL)],
        ["#META","wind_rose_data", _profile_to_json_str(WIND_DIR_MONTH)],
        ["#META","use_per_turbine_sunpos", str(USE_PER_TURBINE_SUNPOS)],
        ["#META","time_step_min", TIME_STEP_MIN],
        ["#META","grid_step_m", GRID_STEP_M],
        ["#META","h_rec_raster", H_REC_RASTER],
        ["#META","raster_tol_m", RASTER_TOL_M],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for row in meta: w.writerow(row)
        w.writerow(header_cols)
        for r in rows: w.writerow(r)

# ---------- (7) Escenaris ----------
SCENARIOS = {
    "WORST": {
        "sun_frac": lambda m: 1.0,
        "avail_fn": avail_fn_factory(AVAIL_MONTH_PROFILES["WORST"]),
        "curt_fn":  curtail_fn_factory(CURTAIL_MONTH_PROFILES["WORST"]),
        "terrain_screen": True,   # WORST també filtra per DEM!
        "use_dir": False,         # sense reponderació direccional
        "tol_azdeg": 10.0,
    },
    "REALISTIC": {
        "sun_frac": sunshinefrac_fn_factory(SUNSHINE_FRAC_MONTH_PROFILES["REALISTIC"]), #lambda m: SUNSHINE_FRAC_MONTH.get(m, 0.6),
        "avail_fn": avail_fn_factory(AVAIL_MONTH_PROFILES["REALISTIC"]),
        "curt_fn":  curtail_fn_factory(CURTAIL_MONTH_PROFILES["REALISTIC"]),
        "terrain_screen": True,
        "use_dir": USE_DIR_REWEIGHT,
        "tol_azdeg": 8.0,
    },
}

def build_horizon_table(tx, ty, Z, TA, TB, TC, TD, TE, TF,
                        bearings=np.arange(0,360,2),
                        max_dist_m=3000.0, step_m=25.0,
                        rotor_shade_h=200.0):
    """Retorna dict: bearing (deg) -> elevació mínima del sol (deg) per “veure” per sobre del terreny."""
    z_t = _dem_bilinear_scalar(Z, tx, ty, TA, TB, TC, TD, TE, TF) + rotor_shade_h
    out = {}
    for b in bearings:  # 0=N, 90=E
        rad = math.radians(b)
        max_ang = -90.0
        d = step_m
        while d <= max_dist_m:
            x = tx + math.sin(rad)*d
            y = ty + math.cos(rad)*d
            z = _dem_bilinear_scalar(Z, x, y, TA, TB, TC, TD, TE, TF)
            ang = math.degrees(math.atan2(z - z_t, d))
            if ang > max_ang: max_ang = ang
            d += step_m
        out[float(b)] = float(max_ang)  # requeriment mínim d’elevació solar
    return out

def interp_horizon(bearing_deg, Htable):
    # interpola circularment cada 2°
    b = (bearing_deg % 360.0)
    b0 = 2.0*math.floor(b/2.0)
    b1 = (b0 + 2.0) % 360.0
    t = (b - b0) / 2.0
    def get(key):
        return Htable.get(key, Htable.get((key+360)%360, -90.0))
    return (1.0 - t)*get(b0) + t*get(b1)

def quick_terrain_clear(tx, ty, px, py, elev_deg, Htable, margin_deg=0.5):
    """Accepta ràpid si l’alçada solar supera l’horitzó en el rumb turbina→punt."""
    bearing = (math.degrees(math.atan2(px - tx, py - ty)) + 360.0) % 360.0
    h_req = interp_horizon(bearing, Htable)
    return elev_deg >= (h_req - margin_deg)

@njit(cache=True, fastmath=True)
def los_clear_turbine_to_point(tx, ty, rx, ry,
                               rotor_shade_h, hrec,
                               DEM_Z, TA, TB, TC, TD, TE, TF,
                               n_samp_per_km, tol_m):
    # Altura a la turbina: DEM + altura efectiva d’ombra del rotor
    z_t = _dem_bilinear_scalar(DEM_Z, tx, ty, TA, TB, TC, TD, TE, TF) + rotor_shade_h
    # Altura al receptor: DEM + alçada del receptor
    z_r = _dem_bilinear_scalar(DEM_Z, rx, ry, TA, TB, TC, TD, TE, TF) + hrec
    step_m = 1000.0 / max(1, int(n_samp_per_km))
    return _los_clear_tol_affine(tx, ty, z_t,  rx, ry, z_r,
                                 DEM_Z, TA, TB, TC, TD, TE, TF,
                                 step_m, float(tol_m))

def next_step_minutes(elev_deg: float) -> float:
    # Tall LAI/central: <3° → podem saltar ràpid
    step_minutes = TIME_STEP_MIN 
    if elev_deg < 3.0:  step_minutes =  5.0
    if elev_deg < 5.0: step_minutes = 4.0
    if elev_deg < 10.0: step_minutes = 3.0
    if elev_deg < 15.0: step_minutes = 2.0
    #if elev_deg < 20.0: step_minutes = 2.0
    if (step_minutes >= TIME_STEP_MIN):
        return step_minutes
    else:
        return TIME_STEP_MIN


# ---------- (8) Càlcul principal ----------
# ---------- Multi Processor -----------
def compute_shadow_flicker_month(scen_name, month, args,
                                 n_samp_per_km=48, max_checks_per_step=4000):
    
    cfg = load_config(args, globals())
    ensure_dem_loaded()
    
    IS_WORST = (scen_name == "WORST")
    
    params = SCENARIOS[scen_name]

    sun_frac_fn = params["sun_frac"]
    avail_fn    = params["avail_fn"]
    curt_fn     = params["curt_fn"]
    use_screen  = params["terrain_screen"]
    use_dir     = params["use_dir"]
    tol_azdeg   = TOL_AZDEG # params["tol_azdeg"]
    
    
    # prepara XX,YY, acumuladors (idèntics en forma i dtype a l’anual)
    xmin, ymin, xmax, ymax = XMIN, YMIN, XMAX, YMAX
    
    xx = np.arange(xmin, xmax+GRID_STEP_M*0.5, GRID_STEP_M)
    yy = np.arange(ymin, ymax+GRID_STEP_M*0.5, GRID_STEP_M)
    XX, YY = np.meshgrid(xx, yy)
    acc_min_grid  = np.zeros_like(XX, dtype=np.float32)
    acc_days_grid = np.zeros_like(XX, dtype=np.uint16)
    daily_hit_mask = np.zeros_like(XX, dtype=bool)

    # next two line original year run
    keyTR = defaultdict(float)
    keyR  = defaultdict(float)

    acc_min_R   = defaultdict(float)
    acc_days_R  = defaultdict(int)
    hit_today_R = defaultdict(bool)
    acc_min_day_R = defaultdict(float)  # (rid, date) -> minuts

    # next line original year run
    step_w_R = defaultdict(float)  # rid -> w màxim del minut
    
    # Timing
    timing = _new_timing_bucket() if ENABLE_TIMING else None

    # timeline del mes
    m = int(month)
    t0 = dt.datetime(YEAR, m, 1, 0, 0, tzinfo=dt.timezone.utc)
    # final del mes
    if m == 12:
        t1 = dt.datetime(YEAR, 12, 31, 23, 59, tzinfo=dt.timezone.utc)
    else:
        t1 = dt.datetime(YEAR, m+1, 1, 0, 0, tzinfo=dt.timezone.utc) - dt.timedelta(minutes=1)
        
    step = dt.timedelta(minutes=TIME_STEP_MIN)

    current_day = None
        
    global REC_PIX_EXTRA
    REC_PIX_EXTRA = build_receptor_pixels_for_screening_on_grid(
        RPTS, xx, yy, ring_pix=REC_PIX_RING
    )
    
    def flush_day():
        for rid, flag in list(hit_today_R.items()):
            if flag:
                acc_days_R[rid] += 1
                hit_today_R[rid] = False
                
        acc_days_grid[:] += daily_hit_mask.astype(np.uint16)
        daily_hit_mask[:] = False
 
    # Bucle temporal
    t = t0
    while t <= t1:
        day = t.date()
        if current_day is None:
            current_day = day
        elif day != current_day:
            flush_day()
            current_day = day
            
        step_w_R.clear()
        
        m = t.month
        # Sol global (referència)
        elev_g, az_g = solar_pos_utc(t, SITE_LAT, SITE_LON)
        dt_min = next_step_minutes(elev_g)  # minuts del pas actual

        # az0 = az_t  # azimut al començament del pas
        # az1 = solar_pos_utc(t + dt.timedelta(minutes=dt_min), SITE_LAT, SITE_LON)[1]
        # az_drift = abs(((az1 - az0 + 180) % 360) - 180)

        # tol_azdeg_pair = dynamic_tol_azdeg(dist_h, rotor_d, az_drift, k=1.2)
        # if abs(((az_rt - az_t + 180) % 360) - 180.0) > tol_azdeg_pair:
        #    continue        
        
        # 0.5 to low check if this is ok, probably has to be something like 2.5 as intensity is higher in Catalonia (Germany 3.0)
        # We leave it to 3º of elevation to assure standards
        if elev_g < MIN_ELEV: 
            t += dt.timedelta(minutes=dt_min)
            #t += step
            continue

        # --- RECEPTORS ---
        for (nuc, rid, rx, ry, hrec) in RPTS:
            for (tname, tx, ty, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power) in TURBINES:
                rotor_shade_h = hub_h + plat_h + (rotor_d * shade_corr)
                max_sf_dist = 10 * rotor_d
                dx, dy  = tx - rx, ty - ry
                dist_h  = (dx*dx + dy*dy) ** 0.5
                if dist_h > max_sf_dist:
                    continue

                # Sol per turbina? Absurd, overkill
                if USE_PER_TURBINE_SUNPOS:
                    lat_t, lon_t = lat, lon
                    elev_t, az_t = solar_pos_utc(t, lat_t, lon_t)
                    if elev_t <= MIN_ELEV:
                        continue
                else:
                    elev_t, az_t = elev_g, az_g

                w_base = 1.0                    
                if IS_WORST:
                    sun = 1.0
                    avail = 1.0
                    curt = 1.0
                    ilum_w = 1.0
                    wind = 1.0
                    weibull = 1.0
                else:
                    # Pes temporal
                    sun   = sun_frac_fn(m)
                    avail = avail_fn(m)
                    curt  = curt_fn(m)
                    weibull = wind_oper_prob_month(m, cut_in, cut_out)
                    # suficent potència? Ha de ser >= 120 W/m2 en funció de la elevació 
                    #if elev_t > 70:
                    #    print("Elev_t: " + str(elev_t))
                    
                    illum_w = elev_min_deg_for_ghi_watts(elev_t) #sun_effective_LAI(elev_t)
                    if use_dir: # and establish not worst if necessary
                        #sec, sec2 = sector_of_az(az_t)
                        wind = wind_dir_yaw(m, sector_of_az(az_t), 0.3) 
                        #wind = wind_dir_yaw(m, sec, sec2)
                        
                    w_base = sun * avail * curt * weibull * wind * illum_w
                    
                    
                # Clip i sanity check
                if not np.isfinite(w_base): 
                    w_base = 0.0                        
                w_base = max(0.0, min(1.0, float(w_base)))
                
                # Abast vertical
                L = (rotor_shade_h - hrec) / max(math.tan(math.radians(elev_t)), 1e-6)
                if dist_h > L:
                    continue

                # Alineació azimutal
                #az_rt = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                #if abs(((az_rt - az_t + 180.0) % 360.0) - 180.0) > tol_azdeg:
                #    continue
                # vector receptor -> turbina al pla
                #dxv = rx - tx
                #dyv = ry - ty
                #dist_hv = math.hypot(dx, dy)

                # Azimut receptor->turbina amb convenció 0°=N, 90°=E (coherent amb el teu sol)
                # atan2(y, x) dóna 0° a l’eix +X. Per tenir 0° cap al Nord, usem atan2(dx, dy).
                az_rt = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

                # Diferència d’azimut mínima (−180..+180)
                daz = abs(((az_rt - az_t + 180.0) % 360.0) - 180.0)

                # Finestra angular geomètrica del disc
                R = 0.5 * rotor_d
                ratio = min(1.0, R / max(dist_h, 1e-6))
                theta_geom = math.degrees(math.asin(ratio))

                # Marge addicional opcional (0° en WINDPRO-strict)
                tol_extra_deg = tol_azdeg  # o el que configuris
                theta_allow = theta_geom + tol_extra_deg

                # Comprovació angular correcta
                if daz > theta_allow:
                    continue
                
                
                # DEM screening, first unexpensive simple test
                if use_screen:
                    H = get_horizon_for(tname, tx, ty, rotor_shade_h)                    
                    if not quick_terrain_clear(tx, ty, rx, ry, elev_t, H):
                        continue
                
                # Now expensive los test
                if not los_clear_turbine_to_point(tx, ty, rx, ry,
                                                  rotor_shade_h, hrec,
                                                  DEM_Z, TA, TB, TC, TD, TE, TF,
                                                  N_SAMP_PER_KM, RASTER_TOL_M):
                    continue
                    
                # minutes_w = TIME_STEP_MIN if IS_WORST else TIME_STEP_MIN * w_base
                minutes_w = dt_min if IS_WORST else dt_min * w_base
                if minutes_w > 0:
                    step_w_R[rid] = max(step_w_R[rid], (1.0 if IS_WORST else w_base))
                    hit_today_R[rid]     = True

        # --- RASTER ---
        # per a cada turbina
        step_elapsed_sum = 0.0
        max_checks_dyn = max_checks_per_step

        step_w_grid = np.zeros_like(XX, dtype=float)

        for (tname, tx, ty, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power) in TURBINES:
            rotor_shade_h = hub_h + plat_h + (rotor_d * shade_corr)
            max_sf_dist = 10 * rotor_d
            # Sol per turbina? Absurd, overkill
            if USE_PER_TURBINE_SUNPOS:
                lat_t, lon_t = lat, lon
                elev_t, az_t = solar_pos_utc(t, lat_t, lon_t)
                if elev_t <= MIN_ELEV:
                    continue
            else:
                elev_t, az_t = elev_g, az_g

            w_eff_grid = 1.0
            if IS_WORST:
                sun = 1.0
                avail = 1.0
                curt = 1.0
                ilum_w = 1.0
                wind = 1.0
                weibull = 1.0
            else:
                # Pes temporal
                sun   = sun_frac_fn(m)
                avail = avail_fn(m)
                curt  = curt_fn(m)
                weibull = wind_oper_prob_month(m, cut_in, cut_out)
                # suficent potència? Ha de ser >= 120 W/m2 en funció de la elevació 
                illum_w = elev_min_deg_for_ghi_watts(elev_t) #sun_effective_LAI(elev_t)

                if use_dir: # and establish not worst if necessary
                    wind = wind_dir_yaw(m, sector_of_az(az_t), 0.3) 

                w_eff_grid = sun * avail * curt * weibull * wind * illum_w

            # Clip i sanity check
            if not np.isfinite(w_eff_grid): 
                w_eff_grid = 0.0
            w_eff_grid = max(0.0, min(1.0, float(w_eff_grid)))                

            elapsed_ms, max_checks_dyn, updated, rr, cc = raster_step_with_screening_numba(
                XX, YY, acc_min_grid,
                elev=elev_t, az=az_t, w=w_eff_grid, dt_min=dt_min,
                h_rec_raster=H_REC_RASTER, tx=tx, ty=ty, max_sf_dist=max_sf_dist, rotor_shade_h=rotor_shade_h,
                rotor_d=rotor_d, rotor_tol_deg=tol_azdeg, n_samp_per_km=n_samp_per_km, tol_m=RASTER_TOL_M,
                max_checks_per_step=max_checks_dyn,
                timing_bucket=timing, tname=tname, throttle=True,
                daily_hit_mask=daily_hit_mask
            )
            # Un cop tens rr,cc per la turbina corrent:
            #if rr.size:
            #    step_w_grid[rr, cc] = np.maximum(step_w_grid[rr, cc], (1.0 if IS_WORST else w_eff_grid))
            #    daily_hit_mask[rr, cc] = True  # dies afectats segueix essent unió                

            step_elapsed_sum += elapsed_ms

        if ENABLE_TIMING and timing is not None:
            timing["per_step"].append(step_elapsed_sum)

        # update turbine minutes for raster steps 
        #acc_min_grid += TIME_STEP_MIN * step_w_grid
        #for rid, wmax in step_w_R.items():
                
        for rid, wmax in step_w_R.items():
            #minutes_w = TIME_STEP_MIN * wmax
            minutes_w = dt_min * wmax
            keyTR[(tname, rid)] += minutes_w
            keyR[rid]           += minutes_w
            acc_min_R[rid]      += minutes_w
            acc_min_day_R[(rid, day)] += minutes_w
            
                
        t += step

    # Flush final
    if current_day is not None:
        flush_day()

    # Timing resum
    if ENABLE_TIMING and timing is not None:
        total_ms = timing["total_ms"]; total_calls = timing["total_calls"]
        mean_ms_call = (total_ms/total_calls) if total_calls else 0.0
        mean_ms_step = (sum(timing["per_step"])/len(timing["per_step"])) if timing["per_step"] else 0.0
        print(f"[{scen_name}] Raster timing: {total_ms:.0f} ms total, {total_calls} calls, "
              f"{mean_ms_call:.2f} ms/call, {mean_ms_step:.1f} ms/step.")
        items = [(k, v["ms"], v["calls"]) for k, v in timing["per_turb"].items()]
        items.sort(key=lambda x: x[1], reverse=True)
        for (tn, ms, calls) in items[:6]:
            print(f"   · {tn}: {ms:.0f} ms ({calls} calls, {ms/max(calls,1):.1f} ms/call)")


    # … bucle temporal com el teu (amb dt_min adaptatiu),
    # … p_sol * p_oper * p_dir * p_av * p_cur (CENTRAL) o tot 1.0 (WORST),
    # … quick_terrain_clear abans del ray-tracing,
    # … suma a receptors i raster.

    # al canviar de dia, fes flush com fas a l’anual
    # al final, calcula minutes_per_day_grid i minutes_per_day_R del mes (opcional)

    # retorna parcial (mateixa clau que fas servir després):
    return {
        "XX":XX, "YY":YY,
        "acc_min_grid": acc_min_grid,
        "acc_days_grid": acc_days_grid,
        "receptors_min": dict(acc_min_R),
        "receptors_days": dict(acc_days_R),
        #"keyTR": dict(keyTR), 
        #"keyR": dict(keyR),        
        "acc_min_day_R": { (rid, d): acc_min_day_R[(rid,d)] for (rid,d) in acc_min_day_R },
    }


def merge_month_results(partials):
    # Assumim mateixa malla en tots
    XX = partials[0]["XX"]; YY = partials[0]["YY"]
    acc_min_grid  = np.zeros_like(XX, dtype=np.float32)
    acc_days_grid = np.zeros_like(XX, dtype=np.uint16)
    receptors_min = defaultdict(float)
    receptors_days= defaultdict(int)
    acc_min_day_R = defaultdict(float)
    #keyTR = defaultdict(float)
    #keyR  = defaultdict(float)

    for p in partials:
        acc_min_grid  += p["acc_min_grid"].astype(np.float32)
        acc_days_grid += p["acc_days_grid"].astype(np.uint16)
        for rid, v in p["receptors_min"].items():
            receptors_min[rid] += float(v)
        for rid, v in p["receptors_days"].items():
            receptors_days[rid] += int(v)
        for key, v in p["acc_min_day_R"].items():
            acc_min_day_R[key] += float(v)
            
        # do we need those bellow?   
        #keyTR[(tname, rid)] += minutes_w
        #keyR[rid]           += minutes_w
            
            

    return XX, YY, acc_min_grid, acc_days_grid, receptors_min, receptors_days, acc_min_day_R

def compute_shadow_flicker_multiproc(scen_name, args, workers=None, n_samp_per_km=48, max_checks_per_step=4000):
    workers = workers or os.cpu_count()
    
    cfg = load_config(args, globals())    
    ensure_dem_loaded()
    
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compute_shadow_flicker_month, scen_name, m, args, n_samp_per_km=n_samp_per_km, max_checks_per_step=max_checks_per_step): m for m in range(1,13)}
        for fut in as_completed(futs):
            parts.append(fut.result())

    XX, YY, acc_min_grid, acc_days_grid, receptors_min, receptors_days, acc_min_day_R = merge_month_results(parts)

    # Derivats finals
    with np.errstate(divide='ignore', invalid='ignore'):
        minutes_per_day_grid = np.where(acc_days_grid>0, acc_min_grid/acc_days_grid, 0.0)
    minutes_per_day_R = {}
    for rid, mins in receptors_min.items():
        d = receptors_days.get(rid, 0)
        minutes_per_day_R[rid] = (mins/d) if d>0 else 0.0

    # Checks 30 min/dia
    over30_days_R = defaultdict(int)
    max_day_min_R = defaultdict(float)
    max_day_date_R= {}
    for (rid, dte), mins in acc_min_day_R.items():
        if mins > MAX_MIN_PER_DAY:
            over30_days_R[rid] += 1
        if mins > max_day_min_R[rid]:
            max_day_min_R[rid]  = mins
            max_day_date_R[rid] = dte

    return {
        "XX":XX, "YY":YY,
        "acc_min_grid": acc_min_grid,
        "acc_days_grid": acc_days_grid,
        "minutes_per_day_grid": minutes_per_day_grid,
        "receptors_min": dict(receptors_min),
        "receptors_days": dict(receptors_days),
        "receptors_mpd": minutes_per_day_R,
        "over30_days_R": dict(over30_days_R),
        "max_day_min_R": dict(max_day_min_R),
        "max_day_date_R": {k: v.isoformat() for k, v in max_day_date_R.items()},
    }


# ---------- (9) Plot helpers ----------
def colormap_hours():
    # 0.1–10 blau, 10–30 verd, 30–100 vermell, 100–2000 groc
    bounds = [0.1, 8, 30, 100, 2000]
    colors = [
        (102/255, 178/255, 255/255),
        (102/255, 204/255, 102/255),
        (255/255, 80/255, 80/255),
        (255/255, 230/255, 0/255),
        (128/255, 96/255, 0/255,0.0),
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N, extend="max")
    return bounds, cmap, norm

def colormap_days():
    # 0.1–10 blau, 10–30 verd, 30–100 vermell, 100–2000 groc
    bounds = [0.1, 8, 30, 60, 120, 240, 365]
    colors  = [
        (102/255, 178/255, 255/255),
        (102/255, 204/255, 102/255),
        (255/255, 80/255, 80/255),
        (255/255, 110/255, 110/255),
        (220/255, 195/255, 0/255),
        (255/255, 230/255, 0/255),
        (128/255, 96/255, 0/255,0.0),
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N, extend="max")
    return bounds, cmap, norm

def colormap_minuts():
    # 0.1–10 blau, 10–30 verd, 30–100 vermell, 100–2000 groc
    bounds = [0.1, 10, 20, 30, 60, 90, 120, 720]
    colors  = [
        (102/255, 178/255, 255/255),
        (102/255, 204/255, 102/255),
        (255/255, 80/255, 80/255),
        (255/255, 110/255, 110/255),
        (255/255, 140/255, 140/255),
        (225/255, 200/255, 0/255),
        (255/255, 230/255, 0/255),
        (128/255, 96/255, 0/255,0.0),
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N, extend="max")
    return bounds, cmap, norm

# Colors i estils per a les anotacions
CLR_TURB_CIRCLE = (0.10, 0.40, 0.90)  # blau fosc per al cercle
CLR_TURB_CROSS  = (0.10, 0.40, 0.90)  # mateix color per la creu
CLR_TURB_LABEL  = (0.05, 0.10, 0.20)  # text fosc
CLR_RECEPTOR    = (0.15, 0.55, 0.15)  # verd
CLR_RECEPT_LABEL= (0.05, 0.10, 0.20)

TURB_CIRCLE_LW  = 1.2
TURB_CROSS_MS   = 9            # mida del marker '+'
RECEPT_MS       = 28           # mida del marker 's' (points^2)
LABEL_FONTSIZE  = 8

# desplaçament petit per a etiquetes per evitar solapar exactament amb el marker
LABEL_DX = 25.0   # metres en UTM X
LABEL_DY = 25.0   # metres en UTM Y

def format_eu(n: float, decimals: int = 2) -> str:
    # Converteix un float en str amb coma europea com a separador decimal.
    # Exemple: 1234.567 -> '1234,57' si decimals=2
    return f"{n:.{decimals}f}".replace(".", ",")


def draw_turbines_and_receptors(ax, turbines, receptors, res, mode,
                                draw_circles=True, draw_cross=True,
                                show_turb_labels=True, show_rec_labels=True):
    
    # Dibuixa turbines i receptors sobre un Axes existent (després del DEM/hillshade i del contourf).
    # - turbines: [(name, x, y), ...]
    # - receptors: [(Nucli, ReceptorID, x, y, h_rec_m), ...]
    # --- Turbines ---
    for (tname, tx, ty, lat, lon, plat_h, hub_h, rotor_d, shade_corr, cut_in, cut_out, model, power) in turbines:
        # Cercle de diàmetre ROTOR_D
        if draw_circles:
            circ = mpatches.Circle((tx, ty), radius=rotor_d * shade_corr,
                                   edgecolor=CLR_TURB_CIRCLE, facecolor='none',
                                   linewidth=TURB_CIRCLE_LW, zorder=50)
            ax.add_patch(circ)
        # Creu al centre
        if draw_cross:
            ax.plot(tx, ty, marker='+', ms=TURB_CROSS_MS, mew=1.4,
                    color=CLR_TURB_CROSS, zorder=55)
        # Etiqueta de la turbina
        if show_turb_labels:
            ax.text(tx + LABEL_DX, ty + LABEL_DY, tname,
                    fontsize=LABEL_FONTSIZE, color=CLR_TURB_LABEL,
                    ha='left', va='bottom', zorder=60,
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc=(1,1,1,0.65), ec=(0,0,0,0.15), lw=0.5))

    # --- Receptors ---
    mins_dict = res.get("receptors_min", {})
    days_dict = res.get("receptors_days", {})
    mpd_dict  = res.get("receptors_mpd", {})
    mpds_reg  = res.get("minutes_per_day_grid", {})
    over30_dict = res.get("over30_days_R", {})
    #if mode=="hours":
    #    print(mpds_reg)

    
    for (nuc, rid, rx, ry, hrec) in receptors:
        #txth = "n/a"
        txtdata = "n/a"
        #txtm = "n/a"
        if mode == "hours":            
            v_min = float(mins_dict.get(rid, 0.0))
            value = format_eu((v_min/60.0), 1)
            txtdata = f"{value} hores/any"
            print(txtdata)
        if mode == "days":            
            value = int(days_dict.get(rid, 0))
            txtdata = f"{value:d} dies/any"
            print(txtdata)
        if mode == "minuts":
            value = format_eu((float(mpd_dict.get(rid, 0.0))), 1)
            txtdata = f"x̄{value} min/dia"
            print(txtdata)
                              
        value = int(over30_dict.get(rid, 0.0))
        txto = f"{value:d} dies > {int(MAX_MIN_PER_DAY)}min"
        txtutmx = f"UTM X: {rx}"
        txtutmy = f"UTM Y: {ry}"
            
        ax.scatter([rx], [ry], marker='s', s=RECEPT_MS,
                   facecolor=CLR_RECEPTOR, edgecolor='white', linewidths=0.8,
                   zorder=52)
        if show_rec_labels:
            ax.text(rx + LABEL_DX, ry + LABEL_DY, f"{nuc}:\n{txtdata}\n{txto}\n{txtutmx}\n{txtutmy}", # sense :{rid}
                    fontsize=LABEL_FONTSIZE, color=CLR_RECEPT_LABEL,
                    ha='left', va='bottom', zorder=58,
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc=(1,1,1,0.65), ec=(0,0,0,0.15), lw=0.5))

            
def plot_all_maps(res, title_prefix="Shadow Flicker", file_suffix=""):
    XX, YY = res["XX"], res["YY"]
    # assure to assign the correct output data to isometric coloring on DEM
    H = res["acc_min_grid"]/60.0   # hores/any
    D = res["acc_days_grid"].astype(float)
    MPD = res["minutes_per_day_grid"]

    fig, ax = plt.subplots(1,1, figsize=(12,6), constrained_layout=True)
    fig.suptitle(f"{title_prefix} – hores/any")

    # Hores/any
    draw_dem_hillshade(ax, alpha_hs=0.9, alpha_dem=0.35)   # << fons DEM+HS    

    b,cmap,norm = colormap_hours()
    cf0 = ax.contourf(XX, YY, H, levels=b, alpha=0.35, cmap=cmap, norm=norm)
    #cfc0= ax.contour(XX, YY, H, levels=[30], colors="#d62728", linewidths=1.0)
    
    cbar0 = plt.colorbar(cf0, ax=ax, ticks=[4, 19, 65, 1000])
    cbar0.ax.set_yticklabels(["0–8 h","8–30 h","30–100 h","100–2000 h"])
    cbar0.set_label("Hores/any")
    
    draw_turbines_and_receptors(ax, TURBINES, RPTS, res, mode="hours")
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("UTM Y (m)")    

    result_path = os.path.join(OUTPUT_DIR, f"SHADOW_{YEAR}_HOURS_YEAR_{file_suffix}.png")    
    plt.savefig(result_path, dpi=180); plt.close(fig)
    
    fig, ax = plt.subplots(1,1, figsize=(12,6), constrained_layout=True)
    fig.suptitle(f"{title_prefix} – dies/any")

    # Dies/any
    draw_dem_hillshade(ax, alpha_hs=0.9, alpha_dem=0.35)
        
    b,cmap,norm = colormap_days()
    cf1 = ax.contourf(XX, YY, D, levels=b, alpha=0.35, cmap=cmap, norm=norm)
    #cfc1 = ax.contour(XX, YY, H, levels=[8], colors="#d62728", linewidths=1.0)


    cbar1 = plt.colorbar(cf1, ax=ax, ticks=[4, 18, 45, 90, 180, 300])
    cbar1.ax.set_yticklabels(["0–8d","8–30d","30–60d","60–120d", "120-240d","240-365d"])
    cbar1.set_label("Dies/any (≥1 min)")
    
    draw_turbines_and_receptors(ax, TURBINES, RPTS, res, mode="days")
    #label_receptors_values(axes[0], RPTS, res, mode="days", dx=25, dy=25)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("UTM Y (m)")    

    result_path = os.path.join(OUTPUT_DIR, f"SHADOW_{YEAR}_DAYS_YEAR_{file_suffix}.png")        
    plt.savefig(result_path, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(1,1, figsize=(12,6), constrained_layout=True)    
    fig.suptitle(f"{title_prefix} – minuts/dia afectat")
    
    # Minuts/dia afectat (mitjana)
    draw_dem_hillshade(ax, alpha_hs=0.9, alpha_dem=0.35)  
        
    b,cmap,norm = colormap_minuts()
    cf2 = ax.contourf(XX, YY, MPD, levels=b, alpha=0.35, cmap=cmap, norm=norm)
    #cfc2 = ax.contour(XX, YY, H, levels=[30], colors="#d62728", linewidths=1.0)

    cbar2 = plt.colorbar(cf2, ax=ax, ticks=[5, 15, 25, 45, 75, 105 , 415])
    cbar2.ax.set_yticklabels(["0–10min","10–20min","20–30min","30–60min","60-90min","90-120min","120-720min"])
    cbar2.set_label("Minuts/dia afectat (mitjana)")
    
    draw_turbines_and_receptors(ax, TURBINES, RPTS, res, mode="minuts")
    #label_receptors_values(axes[0], RPTS, res, mode="minuts", dx=25, dy=25)
        
    ax.set_aspect('equal', 'box')
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("UTM Y (m)")   
    result_path = os.path.join(OUTPUT_DIR, f"SHADOW_{YEAR}_MINUTS_DAY_{file_suffix}.png")
    plt.savefig(result_path, dpi=180); plt.close(fig)

def prepare_csv(scen_name, res):
    """
    Exporta per receptor:
      - Hores_any
      - Dies_afectats
      - Minuts/dia(mitjana)
      - Dies>30min (diari amb criteri > 30 min — ja ho tens implementat)
      - Max_min_dia i Data_max_dia
      + Compli/Excés anual segons escenari:
        WORST     → ANNUAL_LIMIT_ASTR (p.ex. 30 h)
        REALISTIC → ANNUAL_LIMIT_REAL (p.ex. 8 h)
    """
    rows = []
    scen_upper = str(scen_name).upper().strip()

    # Límits anuals segons escenari
    if scen_upper == "WORST":
        limit_h = float(ANNUAL_LIMIT_ASTR)
    else:
        limit_h = float(ANNUAL_LIMIT_REAL)

    for (nuc, rid, rx, ry, hrec) in RPTS:
        mins_year   = float(res["receptors_min"].get(rid, 0.0))   # minuts acumulats any
        hours_year  = mins_year / 60.0
        days_hit    = int(res["receptors_days"].get(rid, 0))
        mpd         = float(res["receptors_mpd"].get(rid, 0.0))   # minuts/dia mitjana
        days_over30 = int(res["over30_days_R"].get(rid, 0))       # criteri > 30 min (ja implementat)
        max_day_min = float(res["max_day_min_R"].get(rid, 0.0))
        max_day_dt  = res["max_day_date_R"].get(rid, "")

        # Compliment anual
        annual_ok  = hours_year <= limit_h
        annual_exc = max(0.0, hours_year - limit_h)

        rows.append([
            nuc, rid, rx, ry, hrec,
            round(hours_year, 2),
            days_hit,
            round(mpd, 2),
            days_over30,
            round(max_day_min, 1),
            max_day_dt,
            limit_h,
            annual_ok,
            round(annual_exc, 2),
        ])

    result_path = os.path.join(OUTPUT_DIR, f"SHADOW_{YEAR}_{scen_name}_{EXPORT_SUFFIX}_receptors.csv")

    write_csv_with_meta(
        result_path,
        [
            "Nucli","ReceptorID","X","Y","h_rec(m)","Hores_any",
            "Dies_afectats","Minuts/dia(mitjana)",
            f"Dies>{int(MAX_MIN_PER_DAY)}min","Max_min_dia","Data_max_dia",
            "Límit_h_any","Compl_any","Excés_h_any"
        ],
        rows, scen_name
    )
    

# ---------- Check for available processing cores ----------
def get_optimal_workers():
    """
    Retorna un nombre de workers òptim segons la màquina:
    - Apple Silicon (M1/M2): limita a 4 (nuclis performance).
    - Altres: num. nuclis físics, com a mínim 2.
    """
    system = platform.system()
    machine = platform.machine().lower()
    cpu_count = mp.cpu_count()

    if system == "Darwin" and ("arm" in machine or "apple" in machine):
        # Apple Silicon → 4 nuclis potents
        return min(4, cpu_count)
    else:
        # En altres sistemes: fem servir nuclis físics si es pot
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
            if physical:
                return max(2, physical)
        except ImportError:
            pass
        return max(2, cpu_count)


_DEM_CACHE = False
def ensure_dem_loaded():
    global _DEM_CACHE, DEM_DS, DEM_Z, DEM_T, DEM_BOUNDS, DEM_XRES, DEM_YRES
    global TA, TB, TC, TD, TE, TF, DEM_NORM, DEM_SHADE
    if _DEM_CACHE:
        return
    DEM_DS, DEM_Z, DEM_T, DEM_BOUNDS, DEM_XRES, DEM_YRES = load_dem(DEM_PATH)

    DEM_Z = np.ascontiguousarray(DEM_Z.astype(np.float32, copy=False))
    
    TA, TB, TC = float(DEM_T.a), float(DEM_T.b), float(DEM_T.c)
    TD, TE, TF = float(DEM_T.d), float(DEM_T.e), float(DEM_T.f)
    DEM_NORM  = normalize01(DEM_Z)
    DEM_SHADE = make_hillshade(DEM_Z, DEM_XRES, DEM_YRES, az_deg=315.0, alt_deg=45.0,
                               valleys_light=VALLEYS_LIGHT)
    
    
    _DEM_CACHE = True

def run_all(args) -> int:
    try:
        cfg = load_config(args, globals())          # segueix omplint globals
        ensure_dem_loaded()                          # DEM real segons YAML
        if OUTPUT_DIR:
            ensure_output_dir(OUTPUT_DIR)        

        # Determinar quins nuclis són realment per processing
        workers = get_optimal_workers()

        if (args.scene == None or args.scene.upper() == "WORST"):
            # WORST (astronòmic)
            #res_worst = compute_shadow_flicker("WORST", None, True, N_SAMP_PER_KM, MAX_CHECKS_STEP)
            res_worst = compute_shadow_flicker_multiproc("WORST", args, workers, N_SAMP_PER_KM, MAX_CHECKS_STEP)
            plot_all_maps(res_worst, title_prefix="WORST (astronòmic, amb DEM)", file_suffix=f"WORST_{EXPORT_SUFFIX}")
            prepare_csv("WORST", res_worst)

        # REALISTIC (realistic)
        if (args.scene == None or args.scene.upper() == "REAL"):    
            #res_real = compute_shadow_flicker("REALISTIC", None, True, N_SAMP_PER_KM, MAX_CHECKS_STEP)
            res_real = compute_shadow_flicker_multiproc("REALISTIC", args, workers, N_SAMP_PER_KM, MAX_CHECKS_STEP)
            plot_all_maps(res_real, title_prefix="REALISTIC (probable, amb DEM)", file_suffix=f"REALISTIC_{EXPORT_SUFFIX}")
            prepare_csv("REALISTIC", res_real)
        return 0
    except FileNotFoundError as e:
        print(f"[ERROR] Fitxer no trobat: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"[ERROR] Execució: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        import traceback
        print("[ERROR] No previst:", file=sys.stderr)
        print(" └─", e, file=sys.stderr)
        traceback.print_exc()
        return 10    
    #cfg = load_config(args, globals())
    #ensure_dem_loaded()
    

# Descomenta per executar en el teu entorn:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shadow Flicker Assessment"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Ruta al fitxer YAML de configuració"
    )
    parser.add_argument(
        "-s", "--scene",
        required=False,
        help="Real o Worst. Per executar solament un escenari."
    )    
    parser.add_argument(
        "-f", "--fast",
        required=False,
        help="yes o y. Tests ràpids"
    )
    
    exit_code = run_all(parser.parse_args())
    sys.exit(exit_code)

