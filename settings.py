# settings.py
import os, yaml, pandas as pd, numpy as np
from pathlib import Path

class Config(dict):
    """Simple dict-like config with attribute access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"{path} is empty or invalid YAML.")
    
    cfg = Config(data)

    # Derive caps
    pv = cfg["pv"]
    if pv.get("area_m2") is not None and pv.get("kw_per_m2") is not None:
        cfg["pv"]["p_max_kw"] = float(pv["area_m2"]) * float(pv["kw_per_m2"])
    else:
        cfg["pv"]["p_max_kw"] = None

    # Battery power caps from load if "auto"
    # (We compute later after load is known; keep flags here)
    return cfg

def resolve_battery_power_caps(cfg, load_series: pd.Series):
    b = cfg["battery"]
    q95 = float(load_series.quantile(0.95))
    p_ch = q95 if str(b.get("p_ch_max_kw", "")).lower()=="auto" else b.get("p_ch_max_kw")
    p_dis = q95 if str(b.get("p_dis_max_kw","")).lower()=="auto" else b.get("p_dis_max_kw")
    return float(p_ch) if p_ch is not None else None, float(p_dis) if p_dis is not None else None

def get_ninja_token() -> str:
    token = os.getenv("NINJA_TOKEN")
    if not token:
        raise RuntimeError("NINJA_TOKEN not found in environment.")
    return token
