# -*- coding: utf-8 -*-
"""
Shadow Flicker – WORST + REALISTIC
Amb DEM screening (Numba), raster i receptors, dies afectats i minuts/dia.
Inclou: CSV #META, timing, curtailment mensual, Weibull vent, reponderació direccional (mitjana=1),
SOL per turbina (opcional), tall 10×D, tolerància azimutal.
"""

import math, time, yaml, argparse, sys, json, csv, datetime as dt
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LightSource
import rasterio
from rasterio.transform import Affine
import rasterio.transform
from shapely.geometry import Point, box, Polygon, MultiPolygon, shape
from typing import Dict, Any, List, Tuple
from functools import lru_cache

cfg = None

def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default
    
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
        sys.exit(f"[ERROR] Fitxer de configuració no trobat: {path}")
    except yaml.YAMLError as e:
        sys.exit(f"[ERROR] Error al parsejar el YAML {path}: {e}")

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
            g["GRID_STEP_M"]  = 50.0
            g["H_REC_RASTER"] = 2.0
            g["TIME_STEP_MIN"] = 10.0
            g["RASTER_TOL_M"] = 0.5            
        else:
            g["GRID_STEP_M"]  = _num(sf.get("grid_step_m"), g.get("GRID_STEP_M", 25.0))
            g["H_REC_RASTER"] = _num(sf.get("h_rec_raster_m"), g.get("H_REC_RASTER", 2.0))
            g["TIME_STEP_MIN"] = _num(sf.get("time_step_min"), g.get("TIME_STEP_MIN", 10.0))
            g["RASTER_TOL_M"] = _num(sf.get("raster_tol_m"), g.get("RASTER_TOL_M", 0.5))

    # ---- TURBINES (llista i índex per id) ----
    legacy = []
    turbine_by_id = {}
    for t in cfg.get("turbines", []):
        tid   = str(t["id"])
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

TURBINES = [
    ("YA3", 347327, 4598444, 41.52286141, 1.170478, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
    ("Y09", 348363, 4598715, 41.525559, 1.182837, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
    ("Y05", 348718, 4598000, 41.519005, 1.186957, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
    ("Y06", 349377, 4597537, 41.515277, 1.194854, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
    ("Y07", 350011, 4597190, 41.512064, 1.202922, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
    ("Y8B", 350526, 4596948, 41.510136, 1.209102, 2.0, 112.0, 172.0, 0.485, 3.0, 25.0, "N175/6.0x", 6.23),
]

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
MAX_SF_DIST = "Per turbine config rotor_d * 10" #10.0 * ROTOR_D   # 10 × D → ~1.72 km per D=172

# Resolució temporal i espacial
TIME_STEP_MIN = 10.0            # puja a 5–10 min si vols calcular més ràpid per proves, però cal fixar-ho en 1 (1 minut) per càlculs exactes
GRID_STEP_M   = 50.0           # pas del raster XY, 25 millor, 50 per proves
H_REC_RASTER  = 2.0            # alçada del receptor “raster” (mapa d’envolvent)
RASTER_TOL_M       = 0.5     # tolerància d'excedència (impuresa) DEM per línia SOL→píxel (m)

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


# Il·luminància (lx) per alçada solar (deg) – Annex LAI (p. 8) (Germany data)
#LUX_BY_HEIGHT = {
#     3:  389,  5:  664, 10: 1402, 15: 2207, 20: 3071,
#    25: 3986, 30: 4942, 35: 5929, 40: 6935, 45: 7949,
#    50: 8959, 55: 9951, 60: 10912
#}

# To assure results we lower Catalonia Passanant values with x1.449838 factor aplied to Mid-Germany example values in German guidelines
# Helper script available to compare and transpose data from one location to another
# python compute_lux_uplift_pvgis.py
# or with explicit coords/names:
# python compute_lux_uplift_pvgis.py --baseline "Frankfurt" 50.1109 8.6821 --target "Passanant" 41.5336 1.1981
# (optional) also save the raw PVGIS JSON responses:
# python compute_lux_uplift_pvgis.py --save-json
#LUX_BY_HEIGHT = {
#     3:  564,  5:  963, 10: 2033, 15: 3200, 20: 4452,
#    25: 5779, 30: 7165, 35: 8596, 40: 10055, 45: 11525,
#    50: 12989, 55: 14427, 60: 15821
#}

# "Strahlungsäquivalent" (lx per W/m²) per alçada – punts ancorats de la LAI
# De l'existència de equivalents per energia i per tenir per tots els valors també de la LAI a Alemanya valors per sobre de 120 W/m2 
# podem deduir que s'han de considerar les intensitats perquè l'òptica estableix que el contrast i per tant l'efecte "shadow flicker" és 
# més intens
#K_EQ_BY_HEIGHT = { 3: 62.0, 60: 105.0 }  # interpolarem linealment entre 3° i 60°

# WEIGHT one forth of radiation intensity as shadow is more intense, 
# more preceptable when contrast and photons (energy) is higher
#LUX_WEIGHT = 0.25

# No cal mirar per elevació per sota de 3º
MIN_ELEV = 3.0

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

import math
import datetime as dt

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

DEM_DS, DEM_Z, DEM_T, DEM_BOUNDS, DEM_XRES, DEM_YRES = load_dem(DEM_PATH)
# Constants de transformada (affine) per a Numba
TA = float(DEM_T.a); TB = float(DEM_T.b); TC = float(DEM_T.c)
TD = float(DEM_T.d); TE = float(DEM_T.e); TF = float(DEM_T.f)

DEM_NORM  = normalize01(DEM_Z)
DEM_SHADE = make_hillshade(DEM_Z, DEM_XRES, DEM_YRES, az_deg=315.0, alt_deg=45.0, valleys_light=VALLEYS_LIGHT)

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



# ---------- Intensitat lúminica ----------
"""
def _interp_table(x, xp, fp):
    # interp lineal amb clamp
    xs = sorted(xp); 
    if x <= xs[0]: return fp[xs[0]]
    if x >= xs[-1]: return fp[xs[-1]]
    for i in range(len(xs)-1):
        x0, x1 = xs[i], xs[i+1]
        if x0 <= x <= x1:
            y0, y1 = fp[x0], fp[x1]
            t = (x - x0) / (x1 - x0)
            return y0 + t*(y1 - y0)

def lux_from_height(hdeg):
    return _interp_table(hdeg, LUX_BY_HEIGHT, LUX_BY_HEIGHT)

def k_equiv_from_height(hdeg):
    # 62 lx/Wm² @ 3°, 105 lx/Wm² @ 60° (LAI, p. 8)
    return _interp_table(hdeg, K_EQ_BY_HEIGHT, K_EQ_BY_HEIGHT)

# 0 or 1
def sun_effective_LAI(hdeg):
    if hdeg < 3.0: 
        return 0  # LAI: per sota ~3° es pot negligir
    E_lux = lux_from_height(hdeg)          # horitzontal (LAI)
    # print(E_lux)
    k_eq  = k_equiv_from_height(hdeg)      # lx per W/m² “equivalent” (LAI)
    # print(k_eq)
    #E_prop = k_eq / 120                   # llindar lx "equivalent" a DNI=120
    # print(E_min)
    # Convert ponderated lux to watts
    E_watts = lux_to_dni(E_lux, hdeg)
    E_watts_prop = E_watts / 120                   # llindar lx "equivalent" a DNI=120
    if (E_watts_prop >= 1):
        return 1
    else:
        return 0

def lux_to_dni(lux: float, solar_elevation_deg: float) -> float:
    # Convert illuminance in lux to Direct Normal Irradiance (W/m²).
    # Assumes luminous efficacy of 105 lm/W for daylight.
    
    # lux: measured horizontal illuminance
    # solar_elevation_deg: solar elevation angle in degrees (0° = horizon, 90° = zenith)
    LUMINOUS_EFFICACY_SUN = 105.0  # lm/W
    
    h_rad = math.radians(solar_elevation_deg)
    if h_rad <= 0:
        return 0.0  # sun below horizon
    return lux / (LUMINOUS_EFFICACY_SUN * math.sin(h_rad))

# intensitat ponderada
def illum_weight_LAI(hdeg):
    if hdeg < MIN_ELEV: return 0.0
    E = lux_from_height(hdeg); k = k_equiv_from_height(hdeg); Emin = 120.0*k
    return 1 + (max(0.0, min(1.0, E / Emin)) * LUX_WEIGHT)  
    # 1 més fracció lux_weight de valor entre 0..1; 1 quan E >= Emin
    lux_weighted = max(0.0, min(1.0, E / Emin))
    if lux_weighted > 0:
        lux_weighted = 1 + (lux_weighted * LUX_WEIGHT)
    return lux_weighted
"""

# ---------- (4) Numba screening ----------
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

@njit(cache=True, fastmath=True)
def _dem_bilinear_scalar(Z, x, y, TA, TB, TC, TD, TE, TF):
    col = (x - TC) / TA
    row = (y - TF) / TE
    r0 = int(math.floor(row)); c0 = int(math.floor(col))
    r1 = r0 + 1; c1 = c0 + 1
    if r0 < 0: r0 = 0
    if c0 < 0: c0 = 0
    if r1 >= Z.shape[0]: r1 = Z.shape[0]-1
    if c1 >= Z.shape[1]: c1 = Z.shape[1]-1
    fr = row - r0; fc = col - c0
    z00 = Z[r0, c0]; z10 = Z[r1, c0]; z01 = Z[r0, c1]; z11 = Z[r1, c1]
    z0 = z00*(1.0-fr) + z10*fr
    z1 = z01*(1.0-fr) + z11*fr
    return z0*(1.0-fc) + z1*fc

@njit(cache=True, fastmath=True)
def _terrain_screen_ok_path_old(tx, ty, px, py, elev_deg, rec_h_m,
                            Z, TA, TB, TC, TD, TE, TF,
                            n_samp_per_km, tol_m):
    dist = math.hypot(px - tx, py - ty)
    if dist < 1.0: return True
    n_samp = int((dist/1000.0) * n_samp_per_km)
    if n_samp < 16: n_samp = 16
    z_p = _dem_bilinear_scalar(Z, px, py, TA, TB, TC, TD, TE, TF)
    tan_e = math.tan(math.radians(elev_deg))
    max_ex = -1e9
    for i in range(n_samp):
        t = 0.0 if n_samp == 1 else (i/(n_samp - 1.0))
        x = px + (tx - px)*t
        y = py + (ty - py)*t
        z = _dem_bilinear_scalar(Z, x, y, TA, TB, TC, TD, TE, TF)
        line = z_p + rec_h_m + tan_e*(t*dist)
        ex = z - line
        if ex > max_ex:
            max_ex = ex
            if max_ex > tol_m:   # tol per receptors (ajusta si cal)
                return False
    return (max_ex <= tol_m)

@njit(cache=True, fastmath=True, parallel=False)
def terrain_screen_ok_batch(tx, ty, elev_deg, rec_h_m,
                            rows, cols, XX, YY,
                            Z, TA, TB, TC, TD, TE, TF,
                            n_samp_per_km, tol_m):
    out = np.zeros(rows.shape[0], dtype=np.uint8)
    for k in range(rows.shape[0]):
        ri = int(rows[k]); ci = int(cols[k])
        px = float(XX[ri, ci]); py = float(YY[ri, ci])
        ok = _terrain_screen_ok_path(tx, ty, px, py, elev_deg, rec_h_m,
                                     Z, TA, TB, TC, TD, TE, TF,
                                     n_samp_per_km, tol_m)
        out[k] = 1 if ok else 0
    return out


import numpy as np
from functools import lru_cache
from numba import njit

# ---------- LOS robust amb tolerància ----------
@njit(cache=True, fastmath=True)
def _los_clear_tol(x0,y0,z0, x1,y1,z1, dem_x,dem_y, dem_z, step_m, tol):
    dx, dy = x1 - x0, y1 - y0
    dist = (dx*dx + dy*dy) ** 0.5
    if dist <= step_m:
        return True
    n = max(1, int(dist / step_m))
    for i in range(1, n):  # excloem extrems
        t = i / n
        xi = x0 + t * dx
        yi = y0 + t * dy
        zi = z0 + t * (z1 - z0)

        # bilinear DEM (assumim north-up, eixos monòtons)
        ix = np.searchsorted(dem_x, xi) - 1
        iy = np.searchsorted(dem_y, yi) - 1
        if ix < 0 or iy < 0 or ix >= dem_x.size - 1 or iy >= dem_y.size - 1:
            # fora del DEM → no podem afirmar obstacle
            continue

        xL, xR = dem_x[ix], dem_x[ix+1]
        yB, yT = dem_y[iy], dem_y[iy+1]
        tx = (xi - xL) / (xR - xL)
        ty = (yi - yB) / (yT - yB)

        zLB = dem_z[iy, ix];   zRB = dem_z[iy, ix+1]
        zLT = dem_z[iy+1, ix]; zRT = dem_z[iy+1, ix+1]
        zdem = (1.0 - ty)*((1.0 - tx)*zLB + tx*zRB) + ty*((1.0 - tx)*zLT + tx*zRT)

        # Tolerància: permet un marge per soroll de DEM/interpolació
        if zdem > zi + tol:
            return False
    return True

# ---------- Conversió GeoTransform → eixos ----------
def _assert_north_up(TA, TB, TC, TD, TE, TF):
    if abs(TC) > 1e-12 or abs(TE) > 1e-12:
        raise NotImplementedError("GeoTransform amb rotació no suportada: TC/TE ≠ 0 (cal reprojecció/warp).")

@lru_cache(maxsize=8)
def _dem_axes_cached(rows, cols, TA, TB, TC, TD, TE, TF):
    _assert_north_up(TA, TB, TC, TD, TE, TF)
    # TA,TD = cantonada superior-esquerra de la cel·la [0,0]
    # Centres de píxel:
    xs = TA + (np.arange(cols) + 0.5) * TB
    ys = TD + (np.arange(rows) + 0.5) * TF  # sovint TF<0
    dem_flip_x = False
    dem_flip_y = False
    if cols >= 2 and xs[1] < xs[0]:
        xs = xs[::-1]; dem_flip_x = True
    if rows >= 2 and ys[1] < ys[0]:
        ys = ys[::-1]; dem_flip_y = True
    return xs.astype(np.float64), ys.astype(np.float64), dem_flip_x, dem_flip_y

def _maybe_flip_dem_z(dem_z, flip_x, flip_y):
    z = dem_z
    if flip_y: z = z[::-1, :]
    if flip_x: z = z[:, ::-1]
    return z

# ---------- ADAPTER compatible amb la teva signatura ----------
def _terrain_screen_ok_path(tx, ty, rx, ry, elev_t, hrec,
                            DEM_Z, TA, TB, TC, TD, TE, TF,
                            n_samp_per_km, RASTER_TOL_M):
    """
    Retorna True si la línia (tx,ty,elev_t)→(rx,ry,hrec) NO és obstruïda.
    - step_m = 1000 / n_samp_per_km
    - Tolerància vertical = RASTER_TOL_M
    - Suporta north-up (sense rotació). Si hi ha rotació al GeoTransform → NotImplementedError.
    """
    rows, cols = DEM_Z.shape
    dem_x, dem_y, fx, fy = _dem_axes_cached(rows, cols, TA, TB, TC, TD, TE, TF)
    dem_z = _maybe_flip_dem_z(DEM_Z, fx, fy).astype(np.float64, copy=False)

    # Pas de mostreig segons n_samp_per_km (evita div/0)
    samples = max(1, int(n_samp_per_km))
    step_m = 1000.0 / float(samples)
    tol = float(max(0.0, RASTER_TOL_M))

    return _los_clear_tol(tx, ty, elev_t, rx, ry, hrec,
                          dem_x, dem_y, dem_z,
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
TARGET_MS_PER_TURB = 12.0
def _new_timing_bucket():
    return {"total_calls":0, "total_ms":0.0, "per_turb":{}, "per_step":[]}

def raster_step_with_screening_numba(XX, YY, acc_min, elev, az, w, h_rec_raster,
                                     tx, ty, max_sf_dist, rotor_shade_h, rotor_tol_deg=8.0,
                                     n_samp_per_km=48, tol_m=0.5,
                                     max_checks_per_step=4000,
                                     timing_bucket=None, tname="TURB",
                                     throttle=True,
                                     daily_hit_mask=None):
    t0 = time.perf_counter()
    DX = XX - tx; DY = YY - ty
    dist_h = np.hypot(DX, DY)
    az_rt  = (np.degrees(np.arctan2(DX, DY)) % 360.0 + 360.0) % 360.0
    L = (rotor_shade_h - h_rec_raster) / max(np.tan(np.radians(max(elev, 0.1))), 1e-6)

    mask_cand = (
        (dist_h <= min(L, max_sf_dist)) &
        (np.abs(((az_rt - az + 180.0) % 360.0) - 180.0) <= rotor_tol_deg)
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

    rows_sel = idx_sel[:,0].astype(np.int64)
    cols_sel = idx_sel[:,1].astype(np.int64)

    ok_mask = terrain_screen_ok_batch(
        float(tx), float(ty), float(elev), float(h_rec_raster),
        rows_sel, cols_sel, XX, YY,
        DEM_Z, TA, TB, TC, TD, TE, TF,
        int(n_samp_per_km), float(tol_m)
    )

    updated = 0
    if np.any(ok_mask):
        rr = rows_sel[ok_mask.astype(bool)]
        cc = cols_sel[ok_mask.astype(bool)]
        #acc_min[rr, cc] += TIME_STEP_MIN * w
        #if daily_hit_mask is not None:
        #    daily_hit_mask[rr, cc] = True
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
    tol_azdeg   = params["tol_azdeg"]
    
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
        "tol_azdeg": 10.0,
    },
}

# ---------- (8) Càlcul principal ----------
def compute_shadow_flicker(scen_name, bbox=None, grid=True, n_samp_per_km=48,
                           max_checks_per_step=4000):
    
    IS_WORST = (scen_name == "WORST")
    
    params = SCENARIOS[scen_name]

    sun_frac_fn = params["sun_frac"]
    avail_fn    = params["avail_fn"]
    curt_fn     = params["curt_fn"]
    use_screen  = params["terrain_screen"]
    use_dir     = params["use_dir"]
    tol_azdeg   = params["tol_azdeg"]

    # Malla (raster)

    xx = np.arange(XMIN, XMAX+GRID_STEP_M, GRID_STEP_M)
    yy = np.arange(YMIN, YMAX+GRID_STEP_M, GRID_STEP_M)
    XX, YY = np.meshgrid(xx, yy)
    acc_min_grid  = np.zeros_like(XX, dtype=float)
    acc_days_grid = np.zeros_like(XX, dtype=np.uint16)
    daily_hit_mask = np.zeros_like(XX, dtype=bool)

    # Receptors acumuladors
    keyTR = defaultdict(float)
    keyR  = defaultdict(float)
    acc_min_R   = defaultdict(float)
    acc_days_R  = defaultdict(int)
    hit_today_R = defaultdict(bool)
    acc_min_day_R = defaultdict(float)   # clau: (rid, date) → minuts en aquell dia
    # Afegim per-minut:
    step_w_R = defaultdict(float)  # rid -> w màxim del minut

    # Timing
    timing = _new_timing_bucket() if ENABLE_TIMING else None

    # Línia temporal (cada TIME_STEP_MIN)
    t0 = dt.datetime(YEAR,1,1,0,0,0, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(YEAR,12,31,23,59,0, tzinfo=dt.timezone.utc)
    step = dt.timedelta(minutes=TIME_STEP_MIN)

    current_day = None

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
        # 0.5 to low check if this is ok, probably has to be something like 2.5 as intensity is higher in Catalonia (Germany 3.0)
        # We leave it to 3º of elevation to assure standards
        if elev_g < MIN_ELEV: 
            t += step
            continue

            
        # alternativa bool de càlcul, n'hi ha o no suficent força de llum solar    
        # illum_w = 1.0 if not IS_WORST else sun_effective_LAI(elev_t)
        # w_base *= illum_w
        
        # weighted no s'aplica amb la guia d'Alemanya, a Holanda i al Regne Unit sí...
        # és perfectament defensable aplicar un augment per intensitat energètica de la llum
        # però no ho fem per evitar problemes amb el protocol alemany que és del standard "de facto" a Catalunya i Espanya
        # illum_w = 1.0 if IS_WORST else illum_weight_LAI(elev_t)  # o usa sun_effective_LAI(...) per gating
        # w_base *= illum_w
        # if w_base <= 0:
        #    continue  # no hi ha “sol efectiu” segons LAI, no s’acumula                    

            
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
                #if not np.isfinite(w_base): 
                #    w_base = 0.0
                        
                #w_base = max(0.0, min(1.0, float(w_base)))
                
                # Abast vertical
                L = (rotor_shade_h - hrec) / max(math.tan(math.radians(elev_t)), 1e-6)
                if dist_h > L:
                    continue

                # Alineació azimutal
                az_rt = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                if abs(((az_rt - az_t + 180.0) % 360.0) - 180.0) > tol_azdeg:
                    continue

                # DEM screening
                if use_screen:
                    if not _terrain_screen_ok_path(tx, ty, rx, ry, elev_t, hrec,
                                                   DEM_Z, TA, TB, TC, TD, TE, TF,
                                                   n_samp_per_km, RASTER_TOL_M):
                        continue
                """
                if scen_name == "REALISTIC":
                    # debug puntual a migdia
                    dbg = {
                        "m": m,
                        "sun": round(sun,3),
                        "avail": round(avail,3),
                        "curt": round(curt,3),
                        "weibull": weibull,
                        "wind_dir": round(wind,3),
                        "illum_w": illum_w,
                        "w_base": round(w_base if 'w_base' in locals() else w,3)
                    }
                    print("[DBG]", dbg)
                """     
                minutes_w = TIME_STEP_MIN if IS_WORST else TIME_STEP_MIN * w_base
                if minutes_w > 0:
                    step_w_R[rid] = max(step_w_R[rid], (1.0 if IS_WORST else w_base))
                    hit_today_R[rid]     = True

        # --- RASTER ---
        # per a cada turbina
        if True:
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
                    elev=elev_t, az=az_t, w=w_eff_grid,
                    h_rec_raster=H_REC_RASTER, tx=tx, ty=ty, max_sf_dist=max_sf_dist, rotor_shade_h=rotor_shade_h,
                    rotor_tol_deg=tol_azdeg, n_samp_per_km=n_samp_per_km, tol_m=RASTER_TOL_M,
                    max_checks_per_step=max_checks_dyn,
                    timing_bucket=timing, tname=tname, throttle=True,
                    daily_hit_mask=daily_hit_mask
                )
                # Un cop tens rr,cc per la turbina corrent:
                if rr.size:
                    step_w_grid[rr, cc] = np.maximum(step_w_grid[rr, cc], (1.0 if IS_WORST else w_eff_grid))
                    daily_hit_mask[rr, cc] = True  # dies afectats segueix essent unió                
                
                step_elapsed_sum += elapsed_ms
            
            # update turbine minutes for raster steps 
            acc_min_grid += TIME_STEP_MIN * step_w_grid
            if ENABLE_TIMING and timing is not None:
                timing["per_step"].append(step_elapsed_sum)

        for rid, wmax in step_w_R.items():
            minutes_w = TIME_STEP_MIN * wmax
            keyTR[(tname, rid)] += minutes_w
            keyR[rid]           += minutes_w
            acc_min_R[rid]      += minutes_w
            acc_min_day_R[(rid, day)] += minutes_w
            
                
        t += step

    # Flush final
    if current_day is not None:
        flush_day()

    # Derivats per receptor: dies amb superació i màxim diari
    over30_days_R   = defaultdict(int)     # #dies amb mins >= MAX_MIN_PER_DAY (it's normally 30min + something)
    max_day_min_R   = defaultdict(float)   # màxim de minuts en 1 dia
    max_day_date_R  = {}                   # data del màxim        
    
    # Recorre totes les claus (rid, date)
    for (rid, dte), mins in acc_min_day_R.items():
        # acccumulate using >= as we can suppose that most of the values even when exactly 30min
        # have a seconds remainder pushing the value over the limit 3min 0sec will be statistically neglectable
        if mins >= MAX_MIN_PER_DAY:
            over30_days_R[rid] += 1
            if (rid == "GLO_1"):
                print(scen_name + "|GLO_1|" + format_eu(mins) + "|" + str(over30_days_R[rid]))
        else:
            if (rid == "GLO_1" and scen_name == "REALISTIC"):
                print(scen_name + "|GLO_1|" + format_eu(mins) + "|" + str(over30_days_R[rid]))
            
        if mins > max_day_min_R[rid]:
            max_day_min_R[rid]  = mins
            max_day_date_R[rid] = dte
            
    # Derivats raster: minuts/dia afectat
    with np.errstate(divide='ignore', invalid='ignore'):
        minutes_per_day_grid = np.where(acc_days_grid > 0,
                                        acc_min_grid / acc_days_grid,
                                        0.0)

    # Derivats receptors: minuts/dia
    minutes_per_day_R = {}
    for rid, mins in acc_min_R.items():
        d = acc_days_R.get(rid, 0)
        minutes_per_day_R[rid] = (mins / d) if d > 0 else 0.0

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

    # Retorna resultats
    return {
        "XX":XX, "YY":YY,
        "acc_min_grid": acc_min_grid,               # minuts/any
        "acc_days_grid": acc_days_grid,             # dies/any amb ≥1 min
        "minutes_per_day_grid": minutes_per_day_grid,
        "receptors_min": acc_min_R,                 # minuts/any per receptor
        "receptors_days": acc_days_R,               # dies/any per receptor
        "receptors_mpd": minutes_per_day_R,         # minuts/dia (mitjana) per receptor
        "keyTR": keyTR, "keyR": keyR,
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
        txth = "n/a"
        txtd = "n/a"
        txtm = "n/a"
        v_min = float(mins_dict.get(rid, 0.0))
        value = v_min/60.0
        txth = f"{value:.1f} hores/any"
        value = int(days_dict.get(rid, 0))
        txtd = f"{value:d} dies/any"
        value = float(mpd_dict.get(rid, 0.0))
        txtm = f"x̄{value:.1f} min/dia"
        value = int(over30_dict.get(rid, 0.0))
        txto = f"{value} dies > {int(MAX_MIN_PER_DAY)}min"
            
        ax.scatter([rx], [ry], marker='s', s=RECEPT_MS,
                   facecolor=CLR_RECEPTOR, edgecolor='white', linewidths=0.8,
                   zorder=52)
        if show_rec_labels:
            ax.text(rx + LABEL_DX, ry + LABEL_DY, f"{nuc}:\n{txth}\n{txtd}\n{txtm}\n{txto}", # sense :{rid}
                    fontsize=LABEL_FONTSIZE, color=CLR_RECEPT_LABEL,
                    ha='left', va='bottom', zorder=58,
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc=(1,1,1,0.65), ec=(0,0,0,0.15), lw=0.5))

            
def plot_all_maps(res, title_prefix="Shadow Flicker", file_suffix=""):
    XX, YY = res["XX"], res["YY"]
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

    png = f"SHADOW_{YEAR}_HOURS_YEAR_{file_suffix}.png"
    plt.savefig(png, dpi=180); plt.close(fig)
    
    fig, ax = plt.subplots(1,1, figsize=(12,6), constrained_layout=True)
    fig.suptitle(f"{title_prefix} – dies/any")

    # Dies/any
    draw_dem_hillshade(ax, alpha_hs=0.9, alpha_dem=0.35)
        
    b,cmap,norm = colormap_days()
    cf1 = ax.contourf(XX, YY, H, levels=b, alpha=0.35, cmap=cmap, norm=norm)
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
    png = f"SHADOW_{YEAR}_DAYS_YEAR_{file_suffix}.png"
    plt.savefig(png, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(1,1, figsize=(12,6), constrained_layout=True)    
    fig.suptitle(f"{title_prefix} – minuts/dia afectat")
    
    # Minuts/dia afectat (mitjana)
    draw_dem_hillshade(ax, alpha_hs=0.9, alpha_dem=0.35)  
        
    b,cmap,norm = colormap_minuts()
    cf2 = ax.contourf(XX, YY, H, levels=b, alpha=0.35, cmap=cmap, norm=norm)
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
    png = f"SHADOW_{YEAR}_MINUTS_DAY_{file_suffix}.png"
    plt.savefig(png, dpi=180); plt.close(fig)

    
def prepare_csv(scen_name, res):
    # CSV receptors SCENE (hores/dies/minuts/dia)
    rows = []
    for (nuc, rid, rx, ry, hrec) in RPTS:
        mins = res["receptors_min"].get(rid, 0.0)
        dys  = res["receptors_days"].get(rid, 0)
        mpd  = res["receptors_mpd"].get(rid, 0.0)
        over = res["over30_days_R"].get(rid, 0)
        mxd  = res["max_day_min_R"].get(rid, 0.0)
        mxd_date = res["max_day_date_R"].get(rid, "")
        rows.append([nuc, rid, rx, ry, hrec, round(mins/60.0,2), dys, round(mpd,2), over, round(mxd,1), mxd_date])
        
    write_csv_with_meta(f"SHADOW_{YEAR}_{scen_name}_{EXPORT_SUFFIX}_receptors.csv",
                        ["Nucli","ReceptorID","X","Y","h_rec(m)","Hores_any","Dies_afectats","Minuts/dia(mitjana)",
                        f"Dies>{int(MAX_MIN_PER_DAY)}min","Max_min_dia","Data_max_dia"],
                        rows, scen_name)
    
# ---------- (10) Exemple d’ús ----------
def run_all():
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
    args = parser.parse_args()

    cfg = load_config(args, globals())

    if (args.scene == None or args.scene.upper() == "WORST"):
        # WORST (astronòmic)
        res_worst = compute_shadow_flicker("WORST")
        plot_all_maps(res_worst, title_prefix="WORST (astronòmic, amb DEM)", file_suffix=f"WORST_{EXPORT_SUFFIX}")
        prepare_csv("WORST", res_worst)

    # REALISTIC (realistic)
    if (args.scene == None or args.scene.upper() == "REAL"):    
        res_cent = compute_shadow_flicker("REALISTIC")
        plot_all_maps(res_cent, title_prefix="REALISTIC (probable, amb DEM)", file_suffix=f"REALISTIC_{EXPORT_SUFFIX}")
        prepare_csv("REALISTIC", res_cent)

# Descomenta per executar en el teu entorn:
if __name__ == "__main__":
    run_all()

#run_all()
