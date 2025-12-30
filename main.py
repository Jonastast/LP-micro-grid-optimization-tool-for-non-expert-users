# main.py (minimal orchestration)
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from settings import load_config, resolve_battery_power_caps, get_ninja_token
from data_io import (
    fetch_pvgis_pv_1kw,
    fetch_ninja_wind,
    scale_load_profile_annual,
    load_hourly_grid_prices,
    load_scaled_heat_profile,
    build_grid_price_profile_from_components,
)
from model import optimize_hourly
import logging
log = logging.getLogger("tool")
logging.basicConfig(level=logging.INFO, format="%(message)s") 

def annualized_unit_cost(c0, r, N, life, om_pct, salvage=True):
    def crf(r, n):
        return (r * (1 + r) ** n) / ((1 + r) ** n - 1) if r > 0 else 1.0 / n

    # NPC (Initial cost and replacements)
    npc = float(c0)

    if life and life < N:
        for t in range(life, N, life):  # IMPORTANT: strictly < N
            npc += float(c0) / ((1 + r) ** t)

        if salvage:
            # last installation occurs at t_last = floor((N-1)/life)*life
            t_last = ((N - 1) // life) * life
            remaining_years = (t_last + life) - N  # calculate remaining years, for example: 20+10-25=5
            if remaining_years > 0:
                salvage_frac = remaining_years / life
                salvage_value_N = float(c0) * salvage_frac
                npc -= salvage_value_N / ((1 + r) ** N)

    annual = npc * crf(r, N) + float(om_pct) * float(c0)
    return annual


def load_and_scale_csv(path, column, target_annual_mwh):
    df = pd.read_csv(path)
    series = pd.Series(df[column].astype(float), name=column)
    series, sf, orig_mwh = scale_load_profile_annual(series, target_annual_mwh)
    series = series.clip(lower=0)
    assert len(series) in (8760, 8784)
    return series, sf, orig_mwh


def maybe_add_ev_load(load_series, ev_cfg): #Add EV load
    if not ev_cfg or not ev_cfg.get("enabled"):
        return load_series, pd.Series(np.zeros(len(load_series)), name="ev_load")

    vehicles = int(ev_cfg.get("vehicles", 0) or 0)
    if vehicles <= 0:
        return load_series, pd.Series(np.zeros(len(load_series)), name="ev_load")

    path_ev = ev_cfg.get("csv_path")
    if not path_ev:
        raise ValueError("EV section enabled but csv_path is missing.")

    df = pd.read_csv(path_ev)
    column = ev_cfg.get("column")
    if column:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {path_ev}.")
    else:
        non_ts_cols = [c for c in df.columns if c.lower() != "timestamp"]
        if not non_ts_cols:
            raise ValueError(f"No non-timestamp columns found in {path_ev}.")
        column = non_ts_cols[0]

    ev_series = pd.Series(df[column].astype(float), name="ev_load").reset_index(drop=True)

    load_series = load_series.reset_index(drop=True)
    if len(ev_series) < len(load_series):
        raise ValueError(f"EV profile length ({len(ev_series)}) is shorter than load ({len(load_series)}).")
    if len(ev_series) > len(load_series):
        ev_series = ev_series.iloc[: len(load_series)]

    ev_added = ev_series * vehicles
    combined = load_series + ev_added

    print(
        f"EV load enabled: {vehicles} vehicles | adds {ev_added.sum()/1000:.1f} MWh/yr "
        f"(total load {combined.sum()/1000:.1f} MWh/yr)"
    )
    return combined, ev_added

def ensure_fig_dir(out_dir: str, run_name: str, scenario_name: str = "") -> str: # creates folder for saving figures
    run_id = _safe_id(run_name)
    scen_id = _safe_id(scenario_name) if scenario_name else ""
    fig_dir = os.path.join(out_dir, "figs", run_id, scen_id)
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

def save_fig(fig, savepath: str | None): # Save figure as pdf
    if not savepath:
        return
    fig.savefig(savepath, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {savepath}")

def _safe_id(s: str) -> str: #Make scenario id safe for filenames
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def export_scenario_csv( # Summ scenarios up and make csv files
    *,
    out_dir: str,
    scenario_id: str,
    cfg: dict,
    idx: pd.DatetimeIndex,
    res: dict,
    annual_scale: float,
    load_house: np.ndarray,
    load_ev: np.ndarray,
    load_heat_el: np.ndarray,
    # economics (annualized)
    ann_capex_elec: float,
    ann_grid: float,
    heat_total_annual: float,
    total_annual: float,
    # optional, for reporting
    total_heat_kwh: float = 0.0,
    heat_peak_kw: float = 0.0,
    gzip_timeseries: bool = True,
    append_summary: bool = True,
    ):
    """
    Writes as:
      1) results/summary.csv        (one row per scenario, append)
      2) results/timeseries_<id>.csv(.gz)  (8760 rows)
    All power/energy are in kWh per hour in timeseries, MWh/year in summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    sid = _safe_id(scenario_id)
 
    # --- Time series arrays from solver ---
    pv   = np.asarray(res.get("pv_gen_profile", []), dtype=float)
    wind = np.asarray(res.get("wind_gen_profile", []), dtype=float)
    ch   = np.asarray(res.get("ch_profile", []), dtype=float)
    dis  = np.asarray(res.get("dis_profile", []), dtype=float)
    soc  = np.asarray(res.get("soc_profile", []), dtype=float)
    spill= np.asarray(res.get("spill_profile", []), dtype=float)
    gimp = np.asarray(res.get("grid_import_profile", np.zeros(len(pv))), dtype=float)
    gexp = np.asarray(res.get("grid_export_profile", np.zeros(len(pv))), dtype=float)

    # align lengths safely
    T = min(len(idx), len(pv), len(wind), len(ch), len(dis), len(soc), len(spill), len(gimp), len(gexp),
            len(load_house), len(load_ev), len(load_heat_el))
    idx = idx[:T]
    pv, wind, ch, dis, soc, spill, gimp, gexp = pv[:T], wind[:T], ch[:T], dis[:T], soc[:T], spill[:T], gimp[:T], gexp[:T]
    load_house, load_ev, load_heat_el = load_house[:T], load_ev[:T], load_heat_el[:T]

    load_total = load_house + load_ev + load_heat_el

    # --- Write timeseries ---
    df_ts = pd.DataFrame({
        "timestamp": idx,
        "load_house_kwh": load_house,
        "load_ev_kwh": load_ev,
        "load_heat_el_kwh": load_heat_el,
        "load_total_kwh": load_total,
        "pv_gen_kwh": pv,
        "wind_gen_kwh": wind,
        "grid_import_kwh": gimp,
        "grid_export_kwh": gexp,
        "batt_charge_kwh": ch,
        "batt_discharge_kwh": dis,
        "soc_kwh": soc,
        "spill_kwh": spill,
    })

    ts_path = os.path.join(out_dir, f"timeseries_{sid}.csv" + (".gz" if gzip_timeseries else ""))
    if gzip_timeseries:
        df_ts.to_csv(ts_path, index=False, compression="gzip")
    else:
        df_ts.to_csv(ts_path, index=False)

    # --- Summary row ---
    E = res.get("energy", {})
    pv_gen   = float(E.get("pv_gen", pv.sum()))
    wind_gen = float(E.get("wind_gen", wind.sum()))
    spill_e  = float(E.get("spill", spill.sum()))
    imp_kwh  = float(E.get("imp_kwh", gimp.sum()))
    exp_kwh  = float(E.get("exp_kwh", gexp.sum()))  # may not exist; ok if 0

    load_kwh_period = float(load_total.sum())
    load_mwh_year   = annual_scale * load_kwh_period / 1000.0

    gen_kwh = pv_gen + wind_gen
    curtailment_ratio = (spill_e / gen_kwh) if gen_kwh > 0 else np.nan
    self_suff = 1.0 - (imp_kwh / load_kwh_period) if load_kwh_period > 0 else np.nan

    batt_kwh = float(res.get("Battery", 0.0))
    discharge_kwh = float(E.get("discharge", dis.sum()))
    cycles_per_year = (annual_scale * discharge_kwh / batt_kwh) if batt_kwh > 1e-9 else np.nan

    # “delivered electricity cost”
    lco_served = (total_annual / (annual_scale * load_kwh_period)) if load_kwh_period > 0 else np.nan  # €/kWh

    # CAPEX (initial) from config unit costs
    pv_kw   = float(res.get("P_pv", 0.0))
    wind_kw = float(res.get("P_w", 0.0))
    capex_total = (
        float(cfg["costs"]["pv_kw"]) * pv_kw +
        float(cfg["costs"]["wind_kw"]) * wind_kw +
        float(cfg["costs"]["battery_kwh"]) * batt_kwh
    )
        # --- Annualized split of electrical assets (PV / wind / battery) ---
    econ = cfg["economics"]
    om   = cfg["o_and_m"]
    N = int(econ["project_life_years"])
    r = float(econ["real_discount_rate"])

    ac_pv = annualized_unit_cost(cfg["costs"]["pv_kw"],       r, N, life=25, om_pct=om["pv_pct"])
    ac_w  = annualized_unit_cost(cfg["costs"]["wind_kw"],     r, N, life=25, om_pct=om["wind_pct"])
    ac_b  = annualized_unit_cost(cfg["costs"]["battery_kwh"], r, N, life=10, om_pct=om["battery_pct"])

    ann_pv_eur_per_year   = ac_pv * pv_kw
    ann_wind_eur_per_year = ac_w  * wind_kw
    ann_batt_eur_per_year = ac_b  * batt_kwh

    df_sum_row = pd.DataFrame([{
        "scenario_id": sid,
        "pv_kw": pv_kw,
        "wind_kw": wind_kw,
        "battery_kwh": batt_kwh,
        "elec_load_mwh_per_year": load_mwh_year,
        "heat_mwh_per_year": (total_heat_kwh / 1000.0) if total_heat_kwh else 0.0,
        "heat_peak_kw": heat_peak_kw,
        "pv_gen_mwh": annual_scale * pv_gen / 1000.0,
        "wind_gen_mwh": annual_scale * wind_gen / 1000.0,
        "spill_mwh": annual_scale * spill_e / 1000.0,
        "import_mwh": annual_scale * imp_kwh / 1000.0,
        "export_mwh": annual_scale * exp_kwh / 1000.0,
        "curtailment_ratio": curtailment_ratio,
        "self_sufficiency": self_suff,
        "battery_cycles_per_year": cycles_per_year,
        "capex_total_eur": capex_total,
        "ann_capex_elec_eur_per_year": ann_capex_elec,
        "ann_pv_eur_per_year": ann_pv_eur_per_year,
        "ann_wind_eur_per_year": ann_wind_eur_per_year,
        "ann_batt_eur_per_year": ann_batt_eur_per_year,
        "ann_grid_eur_per_year": ann_grid,
        "heat_total_annual_eur_per_year": heat_total_annual,
        "total_annual_eur_per_year": total_annual,
        "lco_served_eur_per_kwh": lco_served,
        "timeseries_file": os.path.basename(ts_path),
    }])

    sum_path = os.path.join(out_dir, "summary.csv")
    if append_summary and os.path.exists(sum_path):
        df_sum_row.to_csv(sum_path, mode="a", header=False, index=False)
    else:
        df_sum_row.to_csv(sum_path, index=False)

    print(f"Saved: {sum_path}")
    print(f"Saved: {ts_path}")


def main(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)

    VERBOSE = bool(cfg.get("verbose", False))
    def debug(msg):
        if VERBOSE:
            print(msg)

    run_cfg = cfg.get("run", {})
    base_name = run_cfg.get("name", "run")
    out_dir = run_cfg.get("out_dir", "results")
    fig_dir = ensure_fig_dir(out_dir=out_dir, run_name=base_name)
    gzip_timeseries = bool(run_cfg.get("gzip_timeseries", True))
    append_summary = bool(run_cfg.get("append_summary", True))
    add_tech_suffix = bool(run_cfg.get("add_tech_suffix", True))


    # Annualized unit costs (€/year per unit)
    econ = cfg["economics"]; om = cfg["o_and_m"]; grid = cfg["grid"]
    N = int(econ["project_life_years"]); r = float(econ["real_discount_rate"])

    ac_pv  = annualized_unit_cost(cfg["costs"]["pv_kw"],       r, N, life=25,                     om_pct=om["pv_pct"])
    ac_w   = annualized_unit_cost(cfg["costs"]["wind_kw"],     r, N, life=25,                     om_pct=om["wind_pct"])
    ac_b   = annualized_unit_cost(cfg["costs"]["battery_kwh"], r, N, life=10,                     om_pct=om["battery_pct"])

    # Load + scale load
    load_series, sf, orig_mwh = load_and_scale_csv(
        cfg["load"]["csv_path"],
        cfg["load"]["column"],
        cfg["load"]["target_annual_mwh"]
    )
    print(f"Load scaled {sf:.3f}x | {orig_mwh:.1f} → {load_series.sum()/1000:.1f} MWh/yr")
    
    # Print base load sum before EV/heating additions
    base_load_sum = load_series.sum() / 1000.0
    debug(f"Base load (before EV/heating): {base_load_sum:.1f} MWh/yr")

    #load_series = maybe_add_ev_load(load_series, cfg.get("ev"))
    # --- keep household load BEFORE adding EV ---
    load_house_series = load_series.copy()

    # --- add EV and also get the EV-only profile ---
    load_series, ev_added_series = maybe_add_ev_load(load_series, cfg.get("ev"))

    # Fetch PV & Wind (1 kW profiles)
    site = cfg["site"]
    pv_series = fetch_pvgis_pv_1kw(
        site["latitude"], site["longitude"],
        tilt=cfg["pv"]["tilt_deg"],
        azim=cfg["pv"]["azimuth_deg"],
        loss_pct=cfg["pv"]["system_losses_pct"],
        year=site["year"],
        timezone=site["timezone"]
    )
    token = get_ninja_token()
    wind_cfg = cfg["wind"]
    wind_series = fetch_ninja_wind(
        site["latitude"], site["longitude"],
        years=wind_cfg.get("years", site["year"]),
        turbine=cfg["wind"]["turbine"],
        token=token,
        height=cfg["wind"]["hub_height_m"]
    )

    wind_losses_pct = float(cfg["wind"].get("losses_pct", 0.0) or 0.0)
    if wind_losses_pct > 0:
        wind_series = wind_series * (1.0 - wind_losses_pct/100.0)

    # Save PV and Wind series to CSV
    pv_series.name = "pv_1kw"
    wind_series.name = "wind_1kw"
    pv_series.to_csv("pv_1kw_profile.csv", index=True)
    wind_series.to_csv("wind_1kw_profile.csv", index=True)
    debug("Saved: pv_1kw_profile.csv, wind_1kw_profile.csv")

    # Align lengths
    T = min(len(pv_series), len(wind_series), len(load_series))
    g_pv = pv_series.values[:T]
    g_w  = wind_series.values[:T]
    load = load_series.values[:T]
    # Annualization scale for grid terms if simulating sub-year or leap year
    print(f"Profiles: PV CF~{g_pv.sum()/max(T,1):.3f}, Wind CF~{g_w.sum()/max(T,1):.3f}") #CF for techs
    annual_scale = 8760.0 / float(T)
    
    tz = cfg["site"].get("timezone", "Europe/Copenhagen")
    year_for_idx = int(cfg["site"].get("year", 2019))
    idx = pd.date_range(f"{year_for_idx}-01-01 00:00", periods=T, freq="h", tz=tz)

    # Build hourly grid import price profile (if CSV provided)
    grid_cfg = cfg["grid"]
    grid_year = grid_cfg.get("year", None)

    # --- Import / export enable flags ---
    max_grid = float(grid_cfg.get("max_grid_percent", 0.0) or 0.0)
    import_enabled = max_grid > 0.0

    export_enabled = bool(grid_cfg.get("export_enabled", True))
    export_price_cfg = float(grid_cfg.get("export_price", 0.0) or 0.0)

    # Grid fully disabled only if NO import and NO export allowed
    grid_fully_disabled = (not import_enabled) and (not export_enabled)

    if grid_fully_disabled:
        # Fully islanded: no import, no export, no price profiles needed
        print("Grid fully disabled (no import and no export). "
              "Using zero import and export prices.")
        c_imp_profile = 0.0   # treated as flat 0 EUR/kWh
        c_exp_profile = 0.0   # treated as flat 0 EUR/kWh

    else: #grid enabled
        # ---- Import price profile (incl. KONSTANT + tariffs + VAT) ----
        if import_enabled:
            if grid_cfg.get("spot_price_csv") and grid_cfg.get("transport_price_csv"):
                price_series = build_grid_price_profile_from_components(
                    grid_cfg,
                    year=grid_year,
                    timezone=cfg["site"].get("timezone", "Europe/Copenhagen"),
                )

                price_series = price_series.sort_index()
                if len(price_series) < T:
                    raise ValueError(
                        f"Price profile length ({len(price_series)}) is shorter than simulation horizon ({T})."
                    )
                if len(price_series) > T:
                    price_series = price_series.iloc[:T]

                c_imp_profile = price_series.values
                print(
                    f"Loaded hourly grid prices (incl. KONSTANT + tariffs + VAT): "
                    f"{len(c_imp_profile)} hours, mean={c_imp_profile.mean():.4f} EUR/kWh"
                )

            elif grid_cfg.get("import_price_csv"):
                price_series = load_hourly_grid_prices(
                    path=grid_cfg["import_price_csv"],
                    col=grid_cfg.get("import_price_column", "RealPrice_EUR_kWh"),
                    time_col=grid_cfg.get("import_price_time_col", "HourUTC"),
                    year=cfg["site"].get("year", 2024),
                    timezone=cfg["site"].get("timezone", "Europe/Copenhagen"),
                )
                price_series = price_series.sort_index()
                if len(price_series) < T:
                    raise ValueError(
                        f"Price profile length ({len(price_series)}) is shorter than simulation horizon ({T})."
                    )
                if len(price_series) > T:
                    price_series = price_series.iloc[:T]

                c_imp_profile = price_series.values
                print(
                    f"Loaded hourly grid prices: {len(c_imp_profile)} hours, "
                    f"mean={c_imp_profile.mean():.3f} EUR/kWh"
                )
            else:
                # Fully flat price as last resort
                c_imp_profile = float(grid_cfg["import_cost"])
                print(f"Using flat import price: {c_imp_profile:.3f} EUR/kWh")
        else:
            # No imports allowed, but export may still be allowed
            c_imp_profile = 0.0
            print("Grid import disabled (max_grid_percent <= 0). Using zero import price.")    

        # ---- Export price profile (spot only) ----
        if export_enabled:
            if grid_cfg.get("spot_price_csv"):
                exp_series = load_hourly_grid_prices(
                    path=grid_cfg["spot_price_csv"],
                    col=grid_cfg.get("spot_price_column", "SpotPrice_EUR_kWh"),
                    time_col=grid_cfg.get("spot_price_time_col", "HourUTC"),
                    year=grid_year,  # use the same year as for grid prices
                    timezone=cfg["site"].get("timezone", "Europe/Copenhagen"),
                )
                exp_series = exp_series.sort_index()
                if len(exp_series) < T:
                    # Incomplete spot series → just disable export
                    print(
                        f"Warning: export price profile length ({len(exp_series)}) "
                        f"is shorter than simulation horizon ({T}). "
                        f"Disabling export revenue."
                    )
                    c_exp_profile = None
                    export_enabled = False
                else:
                    if len(exp_series) > T:
                        exp_series = exp_series.iloc[:T]

                    c_exp_profile = exp_series.values
                    p_exp_true = exp_series.values.copy()          # <-- keep for reporting
                    c_exp_profile_lp = np.zeros_like(p_exp_true)  # <-- use in optimization
                    print(
                        f"Loaded hourly export prices (spot only): "
                        f"{len(c_exp_profile)} hours, mean={c_exp_profile.mean():.4f} EUR/kWh"
                    )
            else:
                # export enabled in config, but no spot CSV to base it on
                print(
                    "Warning: export_enabled=True but no spot_price_csv provided. "
                    "Disabling export revenue."
                )
                c_exp_profile = None
                export_enabled = False
        else:
            c_exp_profile = None
            print("Export disabled in config.")

    # ---- Heating demand (space + water) ----
    heating_cfg = cfg.get("heating", {})
    heat_enabled = bool(heating_cfg.get("enabled", False))

    heat_series = None
    if heat_enabled:
        space_cfg = heating_cfg["space"]
        water_cfg = heating_cfg["water"]

        heat_space = load_scaled_heat_profile(
            space_cfg["csv_path"],
            space_cfg.get("column", "load_kWh"),
            space_cfg.get("capacity_mwh", 1.0),
        )
        heat_water = load_scaled_heat_profile(
            water_cfg["csv_path"],
            water_cfg.get("column", "load_kWh"),
            water_cfg.get("capacity_mwh", 1.0),
        )

        # DEBUG: Check raw CSV data before alignment
        debug(f"heat_space: {len(heat_space)} hours, sum={heat_space.sum():.1f} kWh, mean={heat_space.mean():.3f} kW")
        debug(f"heat_water: {len(heat_water)} hours, sum={heat_water.sum():.1f} kWh, mean={heat_water.mean():.3f} kW")


        # Align to T (like you do for load/pv/wind)
        heat_space = heat_space.iloc[:T].reset_index(drop=True)
        heat_water = heat_water.iloc[:T].reset_index(drop=True)
        heat_series = (heat_space + heat_water).rename("heat_kWh")

        print(
            f"Heating demand: space {heat_space.sum()/1000:.1f} MWh/yr, "
            f"water {heat_water.sum()/1000:.1f} MWh/yr, "
            f"total {heat_series.sum()/1000:.1f} MWh/yr"
        )
    else:
        print("Heating disabled in config.")


    # Optional guard: if no PV area cap and export price has positive mean,
    # disable export credit to avoid unbounded oversizing when export is very profitable.
    try:
        area = cfg["pv"].get("area_m2")
    except Exception:
        area = None

    if area is None and export_enabled and c_exp_profile is not None:
        if np.ndim(c_exp_profile) == 0:
            positive_export = float(c_exp_profile) > 0
        else:
            positive_export = float(np.mean(c_exp_profile)) > 0
        if positive_export:
            print(
                "Note: No PV area cap set and export price > 0. "
                "Setting export credit to 0 to avoid unbounded sizing."
            )
            c_exp_profile = None
            export_enabled = False


    # Battery power caps (kW as kWh-in-hour)
    eta_eff = float(cfg["battery"]["eta_roundtrip"])

    # Make sure we always have an array of prices for cost calculations
    if np.isscalar(c_imp_profile):
        price_arr = np.full(T, float(c_imp_profile))
    else:
        price_arr = np.asarray(c_imp_profile, dtype=float)[:T]

    # ---- Heating technology scenarios (full system optimization per tech) ----
    best_scenario = None
    all_scenarios = []

    if heat_enabled:
        heating_cfg = cfg["heating"]
        tech_cfgs = heating_cfg["techs"]
        heat_kwh = heat_series.values  # total space+water heat, aligned to T
        total_heat_kwh = float(heat_kwh.sum())
        heat_peak_kw = float(np.max(heat_kwh))  # peak hourly heat demand

        def ann_th_cost(per_kwth, life, om_pct):
            return annualized_unit_cost(
                c0=per_kwth,
                r=r,
                N=N,
                life=life,
                om_pct=om_pct,
            )

        tech_defs = [
            ("gshp",     tech_cfgs.get("gshp", {})),
            ("ashp",     tech_cfgs.get("ashp", {})),
            ("electric", tech_cfgs.get("electric", {})),
            ("district", tech_cfgs.get("district", {})),
        ]

        for tech_name, tcfg in tech_defs:
            if not tcfg.get("enabled", False):
                continue

            print(f"\n=== Running scenario: {tech_name} ===")

            # 1) How much extra electric load does heating add?
            if tech_name in ("gshp", "ashp", "electric"):
                cop = float(tcfg.get("cop", 3.0 if tech_name == "ashp" else 4.0))
                if cop <= 0:
                    raise ValueError(f"COP must be > 0 for {tech_name}")
                extra_elec = heat_kwh / cop  # kWh electricity for heating
                # Heating variable cost counted via grid electricity in the LP
                var_heat_annual = 0.0
            else:  # district heating
                extra_elec = np.zeros_like(heat_kwh)
                dh_price = float(tcfg.get("variable_tariff_eur_per_kwh", 0.0))
                var_heat_period = dh_price * total_heat_kwh
                var_heat_annual = annual_scale * var_heat_period

            scenario_load = load + extra_elec

            # 2) Battery power caps based on scenario load
            P_ch_max, P_dis_max = resolve_battery_power_caps(cfg, pd.Series(scenario_load))

            # 3) Annualized heating CAPEX (EUR/year)
            if tech_name == "district":
                ac_th = ann_th_cost(
                    tcfg.get("capex_per_kwth", 0.0),
                    tcfg.get("lifetime_years", 25),
                    tcfg.get("om_pct", 0.0),
                )
                dh_conn_fixed_annual = float(tcfg.get("connection_fixed_annual", 0.0))
                heat_capex_annual = ac_th * heat_peak_kw + dh_conn_fixed_annual
            else:
                ac_th = ann_th_cost(
                    tcfg.get("capex_per_kwth", 0.0),
                    tcfg.get("lifetime_years", 20),
                    tcfg.get("om_pct", 0.0),
                )
                heat_capex_annual = ac_th * heat_peak_kw

            # 4) Run your existing LP with this scenario load
            res_s = optimize_hourly(
                scenario_load, g_pv, g_w,
                ac_pv=ac_pv,
                ac_w=ac_w,
                ac_b=ac_b,
                c_imp=c_imp_profile,
                c_exp=c_exp_profile_lp,
                max_grid_import_frac=cfg["grid"].get("max_grid_percent", 0.0),
                annual_scale=annual_scale,
                eta=eta_eff,
                A_roof=cfg["pv"]["area_m2"],
                eta_pv=cfg["pv"]["kw_per_m2"],
                P_w_max=cfg["wind"]["p_max_kw"],
                soc_cycle=True,
                min_autonomy_hours=cfg["battery"]["min_autonomy_hours"],
                spill_penalty=cfg["battery"].get("spill_penalty", 0.0),
                cycle_penalty=cfg["battery"].get("cycle_penalty", 0.0),
                P_ch_max=P_ch_max,
                P_dis_max=P_dis_max,
                soc_min_frac=cfg["battery"]["soc_min_frac"],
                soc_max_frac=cfg["battery"]["soc_max_frac"],
                allow_export=export_enabled,
            )
            # 5) Compute annualized electrical CAPEX (PV, wind, batt)
            Ppv_v  = float(res_s["P_pv"])
            Pw_v   = float(res_s["P_w"])
            B_v    = float(res_s["Battery"])

            ann_capex_elec = ac_pv*Ppv_v + ac_w*Pw_v + ac_b*B_v
            

            # 6) Grid costs with time-varying import/export prices
            gimp_full = np.asarray(res_s.get("grid_import_profile", np.zeros(T)), dtype=float)
            gexp_full = np.asarray(res_s.get("grid_export_profile", np.zeros(T)), dtype=float)

            imp_kwh_period = float(gimp_full.sum())
            exp_kwh_period = float(gexp_full.sum())

            # Import cost
            if c_imp_profile is None:
                imp_cost_period = 0.0
            elif np.isscalar(c_imp_profile):
                imp_cost_period = float(c_imp_profile) * imp_kwh_period
            else:
                c_imp_arr = np.asarray(c_imp_profile, dtype=float)
                if c_imp_arr.shape[0] != gimp_full.shape[0]:
                    raise ValueError(
                        f"Length mismatch: c_imp_profile={c_imp_arr.shape[0]}, "
                        f"grid_import_profile={gimp_full.shape[0]}"
                    )
                imp_cost_period = float(np.dot(c_imp_arr, gimp_full))

            # Export revenue
            if c_exp_profile is None:
                exp_rev_period = 0.0
            elif np.isscalar(c_exp_profile):
                exp_rev_period = float(c_exp_profile) * exp_kwh_period
            else:
                c_exp_arr = np.asarray(c_exp_profile, dtype=float)
                if c_exp_arr.shape[0] != gexp_full.shape[0]:
                    raise ValueError(
                        f"Length mismatch: c_exp_profile={c_exp_arr.shape[0]}, "
                        f"grid_export_profile={gexp_full.shape[0]}"
                    )
                exp_rev_period = float(np.dot(c_exp_arr, gexp_full))

            ann_imp_cost = annual_scale * imp_cost_period
            ann_exp_rev  = annual_scale * exp_rev_period
            ann_grid     = ann_imp_cost - ann_exp_rev

            # 7) Total annual heating cost (CAPEX + DH variable)
            heat_total_annual = heat_capex_annual + var_heat_annual

            # 8) Total annual system cost for this scenario
            total_annual = ann_capex_elec + ann_grid + heat_total_annual

            scenario_result = {
                "name": tech_name,
                "res": res_s,
                "load": scenario_load,
                "pv_kw": Ppv_v,
                "wind_kw": Pw_v,
                "batt_kwh": B_v,
                "extra_elec_profile": extra_elec,
                "extra_elec_heat_kwh": float(extra_elec.sum()),
                "heat_total_annual": heat_total_annual,
                "ann_capex_elec": ann_capex_elec,
                "ann_grid": ann_grid,
                "total_annual": total_annual,
                "imp_kwh_period": imp_kwh_period,
                "exp_kwh_period": exp_kwh_period,
                "imp_cost_period": imp_cost_period,
                "exp_rev_period": exp_rev_period,
                "heat_peak_kw": heat_peak_kw,
                "total_heat_kwh": total_heat_kwh,
            }

            # --- AFTER scenario_result dict is created ---
            scenario_id = f"{base_name}_{tech_name}" if add_tech_suffix else base_name   # or include grid mode/year/etc.

            export_scenario_csv(
                out_dir=out_dir,
                scenario_id=scenario_id,
                cfg=cfg,
                idx=idx,
                res=res_s,
                annual_scale=annual_scale,
                load_house=load_house_series.values[:T],
                load_ev=ev_added_series.values[:T],
                load_heat_el=extra_elec[:T],                 # <-- heating electricity profile for this tech
                ann_capex_elec=ann_capex_elec,
                ann_grid=ann_grid,
                heat_total_annual=heat_total_annual,
                total_annual=total_annual,
                total_heat_kwh=total_heat_kwh,
                heat_peak_kw=heat_peak_kw,
                gzip_timeseries=gzip_timeseries,
                append_summary=append_summary,
            )

            all_scenarios.append(scenario_result)

            print(
                f"Scenario {tech_name}: total annual cost ≈ {total_annual:,.0f} EUR/yr "
                f"(elec capex {ann_capex_elec:,.0f}, grid {ann_grid:,.0f}, heating {heat_total_annual:,.0f})"
            )

        # 9) Choose best scenario and set res/load for the rest of the script
        if all_scenarios:
            best_scenario = min(all_scenarios, key=lambda s: s["total_annual"])
            print("\n=== Heating scenario summary (EUR/year) ===")
            for s in sorted(all_scenarios, key=lambda x: x["total_annual"]):
                print(
                    f"  {s['name']:8s}: total {s['total_annual']:,.0f}  "
                    f"(elec capex {s['ann_capex_elec']:,.0f}, grid {s['ann_grid']:,.0f}, "
                    f"heating {s['heat_total_annual']:,.0f})"
                )
                print(
                f"             PV {s['pv_kw']:.1f} kW, "
                f"Wind {s['wind_kw']:.1f} kW, "
                f"Battery {s['batt_kwh']:.1f} kWh, "
                f"Grid import {s['imp_kwh_period']/1000:.1f} MWh/yr"
                )
            print(
                f"\n=> Cheapest heating tech: {best_scenario['name']} "
                f"with total ≈ {best_scenario['total_annual']:,.0f} EUR/year "
                f"for {best_scenario['total_heat_kwh']/1000:.1f} MWh/yr heat."
            )

            # Print installed heating capacity (peak demand) for the chosen scenario
            try:
                heat_peak_kw = best_scenario.get("heat_peak_kw", None)
                if heat_peak_kw is not None:
                    print(f"Installed heating peak (kW): {heat_peak_kw:.1f} kW")
            except Exception:
                pass

            res = best_scenario["res"]
            load = best_scenario["load"]
            heat_elec = best_scenario.get("heat_elec_profile", np.zeros(T))
            tech_name = best_scenario.get("tech_name", "base")
        else:
            print("No heating technologies enabled; falling back to base scenario.")
            # just run once with original load
            P_ch_max, P_dis_max = resolve_battery_power_caps(cfg, pd.Series(load))
            res = optimize_hourly(
                load, g_pv, g_w,
                ac_pv=ac_pv,
                ac_w=ac_w,
                ac_b=ac_b,
                c_imp=c_imp_profile,
                c_exp=c_exp_profile_lp,
                max_grid_import_frac=cfg["grid"].get("max_grid_percent", 0.0),
                annual_scale=annual_scale,
                eta=eta_eff,
                A_roof=cfg["pv"]["area_m2"],
                eta_pv=cfg["pv"]["kw_per_m2"],
                P_w_max=cfg["wind"]["p_max_kw"],
                soc_cycle=True,
                min_autonomy_hours=cfg["battery"]["min_autonomy_hours"],
                spill_penalty=cfg["battery"].get("spill_penalty", 0.0),
                cycle_penalty=cfg["battery"].get("cycle_penalty", 0.0),
                P_ch_max=P_ch_max,
                P_dis_max=P_dis_max,
                soc_min_frac=cfg["battery"]["soc_min_frac"],
                soc_max_frac=cfg["battery"]["soc_max_frac"],
                allow_export=export_enabled,
            )
    else:
        # Heating disabled: just run once with base load
        P_ch_max, P_dis_max = resolve_battery_power_caps(cfg, pd.Series(load))
        res = optimize_hourly(
            load, g_pv, g_w,
            ac_pv=ac_pv,
            ac_w=ac_w,
            ac_b=ac_b,
            c_imp=c_imp_profile,
            c_exp=c_exp_profile_lp,
            max_grid_import_frac=cfg["grid"].get("max_grid_percent", 0.0),
            annual_scale=annual_scale,
            eta=eta_eff,
            A_roof=cfg["pv"]["area_m2"],
            eta_pv=cfg["pv"]["kw_per_m2"],
            P_w_max=cfg["wind"]["p_max_kw"],
            soc_cycle=True,
            min_autonomy_hours=cfg["battery"]["min_autonomy_hours"],
            spill_penalty=cfg["battery"].get("spill_penalty", 0.0),
            cycle_penalty=cfg["battery"].get("cycle_penalty", 0.0),
            P_ch_max=P_ch_max,
            P_dis_max=P_dis_max,
            soc_min_frac=cfg["battery"]["soc_min_frac"],
            soc_max_frac=cfg["battery"]["soc_max_frac"],
            allow_export=export_enabled,
        )
                    
    gexp_full = np.asarray(res.get("grid_export_profile", np.zeros(T)), dtype=float)

    # spot-price revenue over the simulated period (EUR)
    if p_exp_true.shape[0] != gexp_full.shape[0]:
        raise ValueError(f"Length mismatch: p_exp_true={p_exp_true.shape[0]}, gexp={gexp_full.shape[0]}")

    exp_rev_period_eur = float(np.dot(p_exp_true, gexp_full))

    # annualized (EUR/year), consistent with the rest of main.py
    exp_rev_annual_eur = annual_scale * exp_rev_period_eur

    print(f"Export revenue (ex-post, spot): {exp_rev_period_eur:,.0f} EUR over period, {exp_rev_annual_eur:,.0f} EUR/yr")

    # ===== BUILD LOAD PARTS (for plotting) =====
    load_house = load_house_series.values[:T]          # household kWh each hour
    load_ev    = ev_added_series.values[:T]            # EV kWh each hour

    if heat_enabled and best_scenario is not None:
        load_heat = np.asarray(best_scenario["extra_elec_profile"][:T], dtype=float)  # heating electricity
        heat_tech = best_scenario["name"]
    else:
        load_heat = np.zeros(T)
        heat_tech = "none"

    # ---- EXPORT CSV for scenario ----

    if heat_enabled and best_scenario is not None:
        scenario_id = f"{base_name}_final_{best_scenario['name']}"
        ann_capex_elec = float(best_scenario["ann_capex_elec"])
        ann_grid = float(best_scenario["ann_grid"])
        heat_total_annual = float(best_scenario["heat_total_annual"])
        total_annual = float(best_scenario["total_annual"])
        total_heat_kwh = float(best_scenario.get("total_heat_kwh", 0.0))
        heat_peak_kw = float(best_scenario.get("heat_peak_kw", 0.0))
    else:
        scenario_id = "final_no_heating"
        # If you haven't computed these yet here, you can set them to NaN or compute later
        ann_capex_elec = np.nan
        ann_grid = np.nan
        heat_total_annual = 0.0
        total_annual = np.nan
        total_heat_kwh = 0.0
        heat_peak_kw = 0.0

    export_scenario_csv(
        out_dir=out_dir,
        scenario_id=scenario_id,
        cfg=cfg,
        idx=idx,
        res=res,
        annual_scale=annual_scale,
        load_house=load_house,
        load_ev=load_ev,
        load_heat_el=load_heat,
        ann_capex_elec=ann_capex_elec,
        ann_grid=ann_grid,
        heat_total_annual=heat_total_annual,
        total_annual=total_annual,
        total_heat_kwh=total_heat_kwh,
        heat_peak_kw=heat_peak_kw,
        gzip_timeseries=gzip_timeseries,
        append_summary=append_summary,
    )

    E = res["energy"]
    print(f"Status: {res['status']}")
    period_hours = T
    period_days = period_hours / 24.0
    print(f"Energy over period ({period_hours} h ≈ {period_days:.1f} days), kWh:")
    
    # Print installed capacities
    heating_capacity_str = ""
    if heat_enabled and best_scenario is not None:
        heat_peak_kw = best_scenario.get("heat_peak_kw", 0.0)
        total_heat_kwh = best_scenario.get("total_heat_kwh", 0.0)
        heating_capacity_str = f" | Heating {heat_peak_kw:.1f} kW ({total_heat_kwh/1000:.1f} MWh/yr)"
    
    print("Installed: PV {0:.1f} kW | Wind {1:.1f} kW | Battery {2:.1f} kWh".format(res["P_pv"], res["P_w"], res["Battery"], heating_capacity_str))
    print(f"  PV gen:      {E['pv_gen']:.1f} kWh")
    print(f"  Wind gen:    {E['wind_gen']:.1f} kWh")
    print(f"  Charge:      {E['charge']:.1f} kWh | Discharge: {E['discharge']:.1f} kWh")
    print(f"  Load:        {E['load']:.1f} kWh | Spill: {E['spill']:.1f} kWh")
    print(f"  Grid import: {E['imp_kwh']:.1f} kWh ")
    
    # Grid import fraction vs limit
    try:
        gimp = res.get("grid_import_profile"); load_kwh = float(np.sum(load))
        if gimp is not None and load_kwh > 0:
            imp_frac = float(np.sum(gimp)) / load_kwh
            grid_limit = grid.get("max_grid_percent", 0.0)
            print(f"  Grid import fraction (period): {imp_frac:.2%} (limit {grid_limit:.0%})")
    except Exception:
        pass
    # SOC percent summary
    try:
        B_cap = max(res["Battery"], 1e-9)
        soc_pct = (res["soc_profile"] / B_cap) * 100.0
        print(f"  SOC min/max: {np.min(soc_pct):.1f}% / {np.max(soc_pct):.1f}%")
    except Exception:
        pass

    # Additional cost breakdown (annualized) + CAPEX
    try:
        Ppv_v  = res['P_pv']
        Pw_v   = res['P_w']
        B_v    = res['Battery']

        # Annualized CAPEX (EUR/year)
        ann_capex = ac_pv * Ppv_v + ac_w * Pw_v + ac_b * B_v

        # --- Detailed grid import/export breakdown ---
        gimp_full = res.get('grid_import_profile')
        gexp_full = res.get('grid_export_profile')

        if gimp_full is not None and gexp_full is not None:
            gimp_full = np.asarray(gimp_full, dtype=float)
            gexp_full = np.asarray(gexp_full, dtype=float)

            # kWh over the simulated period
            imp_kwh_period = float(gimp_full.sum())
            exp_kwh_period = float(gexp_full.sum())

            # EUR over the simulated period: import
            if c_imp_profile is None:
                imp_cost_period = 0.0
            elif np.isscalar(c_imp_profile):
                imp_cost_period = float(c_imp_profile) * imp_kwh_period
            else:
                c_imp_arr = np.asarray(c_imp_profile, dtype=float)
                if c_imp_arr.shape[0] != gimp_full.shape[0]:
                    raise ValueError(
                        f"Length mismatch: c_imp_profile={c_imp_arr.shape[0]}, "
                        f"grid_import_profile={gimp_full.shape[0]}"
                    )
                imp_cost_period = float(np.dot(c_imp_arr, gimp_full))

            # EUR over the simulated period: export
            if c_exp_profile is None:
                exp_rev_period = 0.0
            elif np.isscalar(c_exp_profile):
                exp_rev_period = float(c_exp_profile) * exp_kwh_period
            else:
                c_exp_arr = np.asarray(c_exp_profile, dtype=float)
                if c_exp_arr.shape[0] != gexp_full.shape[0]:
                    raise ValueError(
                        f"Length mismatch: c_exp_profile={c_exp_arr.shape[0]}, "
                        f"grid_export_profile={gexp_full.shape[0]}"
                    )
                exp_rev_period = float(np.dot(c_exp_arr, gexp_full))

            # Annualized EUR/year
            ann_imp_cost = annual_scale * imp_cost_period
            ann_exp_rev  = annual_scale * exp_rev_period
            ann_grid     = ann_imp_cost - ann_exp_rev

            # Average import price over the period
            if imp_kwh_period > 0:
                imp_price_avg = imp_cost_period / imp_kwh_period
            else:
                imp_price_avg = float('nan')
        else:
            ann_grid = float('nan')
            imp_kwh_period = exp_kwh_period = 0.0
            imp_cost_period = exp_rev_period = 0.0
            imp_price_avg = float('nan')

        ann_total = ann_capex + (0.0 if np.isnan(ann_grid) else ann_grid)

        print("Annualized cost (EUR/year):")
        print(
            f"  PV: {ac_pv*Ppv_v:,.0f} | Wind: {ac_w*Pw_v:,.0f} | "
            f"Battery: {ac_b*B_v:,.0f}"
        )

        if not np.isnan(ann_grid):
            if abs(annual_scale - 1.0) < 1e-6:
                scale_str = ""
            else:
                scale_str = f", scale {annual_scale:.3f}x"

            print(
                f"  Grid net: {ann_grid:,.0f} EUR/yr | avg {imp_price_avg:.3f} €/kWh over period)"
                f"-{exp_rev_period:,.0f} EUR export{scale_str})"
            )

            if exp_kwh_period > 0:
                print(
                    f"  Export revenue (savings): "
                    f"Grid export: {exp_kwh_period:,.0f} kWh "
                    f"{exp_rev_period:,.0f} EUR over period, "
                    f"{ann_exp_rev:,.0f} EUR/yr"
                )


        print(f"  Total annualized: {ann_total:,.0f}")

        # Initial CAPEX (EUR)
        cap_pv  = cfg["costs"]["pv_kw"] * Ppv_v
        cap_w   = cfg["costs"]["wind_kw"] * Pw_v
        cap_b   = cfg["costs"]["battery_kwh"] * B_v
        cap_total = cap_pv + cap_w + cap_b

        print("Initial CAPEX (EUR):")
        print(f"  PV: {cap_pv:,.0f} | Wind: {cap_w:,.0f} | Battery: {cap_b:,.0f}")
        print(f"  Total CAPEX: {cap_total:,.0f}")

    except Exception as e:
        print(f"Cost breakdown unavailable: {e}")

    try:
        gimp = np.asarray(res.get("grid_import_profile", np.zeros(T)), dtype=float)
        imp_kwh = float(gimp.sum())
        load_kwh = float(np.sum(load))
        imp_frac = imp_kwh / load_kwh if load_kwh > 0 else 0.0
    except Exception:
        imp_kwh, imp_frac = 0.0, 0.0

    tech = best_scenario["name"] if (heat_enabled and best_scenario is not None) else "base"
    print(
    f"RESULT | tech={tech} | PV={res['P_pv']:.1f} kW | Wind={res['P_w']:.1f} kW | "
    f"Batt={res['Battery']:.1f} kWh | Import={imp_kwh/1000:.1f} MWh/yr ({imp_frac*100:.2f}%) | "
    f"Total={ann_total:,.0f} EUR/yr"
    )
    #idx = pd.date_range("2019-01-01 00:00", periods=T, freq="H")


    # Save the final import price profile as CSV
    if np.isscalar(c_imp_profile):
        # Create constant series for T hours
        import_series = pd.Series([float(c_imp_profile)] * T, name="import_price_eur_per_kwh")
    else:
        # It's already an array
        import_series = pd.Series(c_imp_profile[:T], name="import_price_eur_per_kwh")
    

    import_series.to_csv("final_import_price_profile.csv", index=True)
    debug("Saved: final_import_price_profile.csv")


    # --- Build plotting arrays from result ---
    pv  = np.asarray(res["pv_gen_profile"], dtype=float)
    wind= np.asarray(res["wind_gen_profile"], dtype=float)
    ch  = np.asarray(res["ch_profile"], dtype=float)
    dis = np.asarray(res["dis_profile"], dtype=float)
    soc = np.asarray(res["soc_profile"], dtype=float)
    spill = np.asarray(res["spill_profile"], dtype=float)
    gimp  = np.asarray(res.get("grid_import_profile", np.zeros(T)), dtype=float)
    

    # Ex-post export (if you want): export curtailed only when spot > 0
    if export_enabled and "p_exp_true" in locals():
        gexp_expost = np.where(p_exp_true[:T] > 0.0, spill, 0.0)
    else:
        gexp_expost = None
    charge_from_surplus = np.minimum(ch, spill + 0.0)  # or just ch if spill already means "excess"
    net_spill = np.maximum(spill - ch, 0.0)

    # choose a nice week to show (highest load week)
    H = 24 * 7
    start = int(np.argmax(np.convolve(load, np.ones(H), mode="valid"))) if T > H else 0

    tz = cfg["site"].get("timezone", "Europe/Copenhagen")
    year_for_idx = int(cfg["site"].get("year", 2019))
    idx = pd.date_range(f"{year_for_idx}-01-01 00:00", periods=T, freq="H", tz=tz)

    # --- ONE idx, timezone-aware ---
    tz = cfg["site"].get("timezone", "Europe/Copenhagen")
    year_for_idx = int(cfg["site"].get("year", 2019))
    idx = pd.date_range(f"{year_for_idx}-01-01 00:00", periods=T, freq="H", tz=tz)

    # Total demand that your plot uses (house+ev+heat+charge)
    H = 24 * 7
    load_total_for_weekpick = (load_house + load_ev + load_heat + ch)

    start_winter = pick_peak_week_in_months(load_total_for_weekpick, idx, months=(1, 2), H=H)
    start_summer = pick_peak_week_in_months(load_total_for_weekpick, idx, months=(6, 7), H=H)

    print(f"Winter week starts: {idx[start_winter]}  (hours {start_winter}–{start_winter+H})")
    print(f"Summer week starts: {idx[start_summer]}  (hours {start_summer}–{start_summer+H})")

    def make_avgday(start):
        return {
            "load_house": average_window_to_24h(load_house, idx, start, H),
            "load_ev":    average_window_to_24h(load_ev,    idx, start, H),
            "load_heat":  average_window_to_24h(load_heat,  idx, start, H),
            "ch":         average_window_to_24h(ch,         idx, start, H),
            "pv":         average_window_to_24h(pv,         idx, start, H),
            "wind":       average_window_to_24h(wind,       idx, start, H),
            "gimp":       average_window_to_24h(gimp,       idx, start, H),
            "dis":        average_window_to_24h(dis,        idx, start, H),
            "soc":        average_window_to_24h(soc,        idx, start, H) if soc is not None else None,
        }

    winter = make_avgday(start_winter)
    summer = make_avgday(start_summer)

    plot_dispatch_like_example(
        winter["load_house"], winter["load_ev"], winter["load_heat"], winter["ch"],
        winter["pv"], winter["wind"], winter["gimp"], winter["dis"],
        soc=winter["soc"], batt_cap_kwh=res["Battery"],
        start=0, H=24,
        title="Average Winter Day (24h)",
        savepath=os.path.join(fig_dir, "dispatch_winter_avgday.pdf"),
        show=False
    )

    plot_dispatch_like_example(
        summer["load_house"], summer["load_ev"], summer["load_heat"], summer["ch"],
        summer["pv"], summer["wind"], summer["gimp"], summer["dis"],
        soc=summer["soc"], batt_cap_kwh=res["Battery"],
        start=0, H=24,
        title="Average Summer Day (24h)",
        savepath=os.path.join(fig_dir, "dispatch_summer_avgday.pdf"),
        show=False
    )

    plot_two_stacked_bars(
        res, load_house, load_ev, load_heat, heat_tech=heat_tech,
        savepath=os.path.join(fig_dir, "energy_balance_stacked.pdf"),
        show=False
    )

    plot_two_grouped_bars(
        res, load_house, load_ev, load_heat, heat_tech=heat_tech,
        savepath=os.path.join(fig_dir, "energy_balance_grouped.pdf"),
        show=False
    )

    plot_heatmap_month_hour(spill, "Mean spill (curtailment) by month/hour", year=grid_year or 2019)

    if export_enabled and "p_exp_true" in locals():
        plot_spill_vs_export_price(spill, p_exp_true[:T])

def average_window_to_24h(x, idx, start, H=24*7):
    sl = slice(start, min(start+H, len(x)))
    s = pd.Series(np.asarray(x, float)[sl], index=idx[sl])
    return s.groupby(s.index.hour).mean().reindex(range(24)).values


def plot_two_stacked_bars(res, load_house, load_ev, load_heat, heat_tech="none",
                          savepath=None, show=True):
    import matplotlib.pyplot as plt
    import numpy as np

    E = res["energy"]

    # Values in MWh (better scale for report)
    src_labels = ["PV", "Wind", "Grid import", "Battery discharge"]
    src_vals = np.array([
        float(E.get("pv_gen", 0.0)),
        float(E.get("wind_gen", 0.0)),
        float(E.get("imp_kwh", 0.0)),
        float(E.get("discharge", 0.0)),
    ]) / 1000.0

    sink_labels = ["Household", "EV", "Battery charge", "Spill", f"Heating ({heat_tech})"]
    sink_vals = np.array([
        float(np.sum(load_house)),
        float(np.sum(load_ev)),
        float(E.get("charge", 0.0)),
        float(E.get("spill", 0.0)),
        float(np.sum(load_heat))
    ]) / 1000.0

    color_map = {
        "PV": "orange",
        "Wind": "tab:blue",
        "Grid import": "tab:gray",
        "Battery discharge": "tab:green",
        "Household": "tab:purple",
        "EV": "tab:pink",
        "Heating": "tab:brown",
        "Battery charge": "tab:olive",
        "Spill": "tab:red",
    }

    # Bigger + higher DPI
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=250)

    x = np.array([0.0, 0.25])   # spacing between Produced/Used
    width = 0.2
    ax.margins(y=0.08)

    def draw_stack(x0, labels, vals):
        bottom = 0.0
        for lab, v in zip(labels, vals):
            if v <= 0:
                continue
            base = lab.split(" (")[0]  # Heating (ashp) -> Heating
            ax.bar(
                x0, v, width=width, bottom=bottom,
                color=color_map.get(base, "tab:gray"),
                edgecolor="white", linewidth=0.6
            )

            # labels: only show if segment is big enough
            if v >= 0.08 * np.sum(vals):   # 8% of bar
                ax.text(x0, bottom + v/2, f"{v:.0f}",
                        ha="center", va="center", fontsize=9, color="black")

            bottom += v
        return bottom

    total_src = draw_stack(x[0], src_labels, src_vals)
    total_sink = draw_stack(x[1], sink_labels, sink_vals)

    # optional totals on top
    ax.text(x[0], total_src + 0.02*max(total_src, total_sink), f"{total_src:.0f}",
            ha="center", va="bottom", fontsize=10)
    ax.text(x[1], total_sink + 0.02*max(total_src, total_sink), f"{total_sink:.0f}",
            ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(["Produced", "Used"])
    ax.set_ylabel("Energy over period [MWh]")
    ax.set_title("Energy balance (stacked)")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # smaller legend, below plot
    legend_items = ["PV", "Wind", "EV", "Grid import", "Household", "Heating", "Battery discharge", "Battery charge", "Spill"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[l]) for l in legend_items]
    ax.legend(handles, legend_items, ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.47, -0.08), fontsize=10)

    plt.tight_layout()

    if savepath:
        # Use PDF for LaTeX = infinitely sharp
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

def plot_two_grouped_bars(res, load_house, load_ev, load_heat, heat_tech="none",
                          savepath=None, show=True):
    import numpy as np
    import matplotlib.pyplot as plt

    E = res["energy"]

    # --- Totals over the period (kWh) -> convert to MWh ---
    produced = {
        "PV": float(E.get("pv_gen", 0.0)) / 1000.0,
        "Wind": float(E.get("wind_gen", 0.0)) / 1000.0,
    #    "Grid import": float(E.get("imp_kwh", 0.0)) / 1000.0,
        "Battery discharge": float(E.get("discharge", 0.0)) / 1000.0,
    }

    used = {
        "Household": float(np.sum(load_house)) / 1000.0,
        "EV": float(np.sum(load_ev)) / 1000.0,
        "Battery charge": float(E.get("charge", 0.0)) / 1000.0,
        "Spill": float(E.get("spill", 0.0)) / 1000.0,
        f"Heating ({heat_tech})": float(np.sum(load_heat)) / 1000.0,
    }

    # Keep a consistent order (important for readability)
    prod_keys = ["PV", "Wind", "Battery discharge"] #"Grid import",
    use_keys  = ["Household", "EV", "Battery charge", "Spill", f"Heating ({heat_tech})"]

    # Color map (same idea as yours)
    color_map = {
        "PV": "orange",
        "Wind": "tab:blue",
    #    "Grid import": "tab:gray",
        "Battery discharge": "tab:green",
        "Household": "tab:purple",
        "EV": "tab:pink",
        "Battery charge": "tab:olive",
        "Spill": "tab:red",
        "Heating": "tab:brown",
    }

    def pick_color(label):
        base = label.split(" (")[0]  # "Heating (ashp)" -> "Heating"
        return color_map.get(base, "tab:gray")

    # --- Build plotting arrays ---
    groups = ["Produced", "Used"]
    xg = np.arange(len(groups))  # [0, 1]

    # bar geometry: thin bars, lots of breathing room
    max_bars_in_group = max(len(prod_keys), len(use_keys))
    bar_w = 0.15
    group_w = max_bars_in_group * bar_w
    offsets_prod = (np.arange(len(prod_keys)) - (len(prod_keys) - 1) / 2) * bar_w
    offsets_use  = (np.arange(len(use_keys))  - (len(use_keys)  - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=(5.5, 6.0))

    # --- Draw Produced group ---
    for i, k in enumerate(prod_keys):
        v = produced[k]
        if v <= 0:
            continue
        xi = xg[0] + offsets_prod[i]
        ax.bar(xi, v, width=bar_w, color=pick_color(k), edgecolor="white", linewidth=0.7, label=k)
        ax.text(xi, v + 0.01*max(produced.values()), f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    # --- Draw Used group ---
    for i, k in enumerate(use_keys):
        v = used[k]
        if v <= 0:
            continue
        xi = xg[1] + offsets_use[i]
        ax.bar(xi, v, width=bar_w, color=pick_color(k), edgecolor="white", linewidth=0.7, label=k)
        ax.text(xi, v + 0.01*max(used.values()), f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    # --- Formatting ---
    ax.set_xticks(xg)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Energy over period [MWh]")
    ax.set_title("Energy balance (grouped)")

    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h); l2.append(l); seen.add(l)

    ax.legend(h2, l2, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.21))

    plt.tight_layout()
    save_fig(fig, savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)



def rotate_to_midnight(x, start_hour):
    """
    Rotate a 24h array so index 0 corresponds to 00:00.
    """
    return np.roll(x, -start_hour)

def average_week_to_24h(x, start, H=24*7):
    """
    Take a 7-day slice [start : start+H] and average to a 24h profile:
    hour 0 = mean of all 7 occurrences of hour 0 in that week, etc.
    """
    x = np.asarray(x, float)[start:start+H]
    H = min(H, len(x))
    days = H // 24
    if days < 1:
        # not enough data, just return first 24 hours padded/truncated
        out = np.zeros(24)
        out[:min(24, len(x))] = x[:min(24, len(x))]
        return out
    x = x[:days*24].reshape(days, 24)
    return x.mean(axis=0)

def pick_peak_week_in_months(load_total, idx, months=(1, 2), H=24*7):
    """
    Pick the start hour of the week (H hours) with the largest sum(load_total)
    within the given months.
    """
    load_total = np.asarray(load_total, float)
    mask = idx.month.isin(months)
    candidate = np.where(mask)[0]
    if len(candidate) < H:
        return 0

    # valid week starts where the whole week stays inside the allowed months
    valid_starts = []
    for s in candidate:
        e = s + H
        if e <= len(idx) and np.all(mask[s:e]):
            valid_starts.append(s)
    if not valid_starts:
        return int(candidate[0])

    # choose start with max weekly sum
    best_s = max(valid_starts, key=lambda s: float(load_total[s:s+H].sum()))
    return int(best_s)

def plot_dispatch_like_example(
    load_house, load_ev, load_heat, ch,      # demand parts (kWh/h)
    pv, wind, gimp, dis,                     # supply parts (kWh/h)
    soc=None, batt_cap_kwh=None,             # <-- NEW: SOC profile + battery capacity
    start=0, H=24,
    title="Electricity Dispatch",
    savepath=None,
    show=True,
):
    """
    Bars (left axis): supply stack (PV/Wind/Grid/Batt discharge)
      - hatched part = curtailment (PV/Wind only)
    Lines (left axis): cumulative demand stack (House -> +EV -> +Heat -> +Charge)

    SOC (right axis): % of battery capacity
    """

    # --- slice ---
    end = min(len(pv), start + H)
    t = np.arange(start, end)

    # Demand parts
    Hh = np.asarray(load_house[start:end], float)
    Ev = np.asarray(load_ev[start:end], float)
    He = np.asarray(load_heat[start:end], float)
    Ch = np.asarray(ch[start:end], float)          # battery charge (positive)

    # Supply parts
    PV = np.asarray(pv[start:end], float)
    W  = np.asarray(wind[start:end], float)
    GI = np.asarray(gimp[start:end], float)
    D  = np.asarray(dis[start:end], float)

    # Total demand (the thing supply should meet)
    demand = Hh + Ev + He + Ch

    # --- colors ---
    C = {
        "PV": "orange",
        "Wind": "cornflowerblue",
        "Grid": "lightgray",
        "Dis": "mediumseagreen",
        "HouseLine": "black",
        "EVLine": "darkgreen",
        "HeatLine": "tab:red",
        "ChargeLine": "dimgray",
        "SOC": "magenta",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    # Thin/clean bars
    bar_w = 0.72
    bottom_supply = np.zeros_like(demand)

    bottom_served = np.zeros_like(demand)  # used to compute remaining demand
    bottom_plot   = np.zeros_like(demand)  # used to stack visuals (served+curtail)

    def stack_supply_with_curtail(y, label, color, allow_curtail=True):
        nonlocal bottom_served, bottom_plot

        remaining = np.maximum(0.0, demand - bottom_served)
        served = np.minimum(y, remaining)
        surplus = y - served

        # Served (solid) stacked by bottom_plot (so it sits above whatever is already drawn)
        ax.bar(
            t, served, bottom=bottom_plot, width=bar_w,
            color=color, edgecolor="none", label=label
        )

        # Curtailment (hatched) stacked above served (still using bottom_plot)
        if allow_curtail:
            ax.bar(
                t, surplus, bottom=bottom_plot + served, width=bar_w,
                color=color, edgecolor="none", hatch="///", alpha=0.30,
                label=f"{label} Curtailment"
            )

        # Accounting bottom: only served meets demand
        bottom_served[:] += served

        # Visual bottom: include the full bar height so later things don't overlap
        bottom_plot[:] += (served + (surplus if allow_curtail else 0.0))

    # Supply order (who gets curtailed first)
    stack_supply_with_curtail(PV, "PV", C["PV"], allow_curtail=True)
    stack_supply_with_curtail(W,  "Wind", C["Wind"], allow_curtail=True)
    stack_supply_with_curtail(GI, "Grid import", C["Grid"], allow_curtail=False)
    stack_supply_with_curtail(D,  "Battery discharge", C["Dis"], allow_curtail=False)

    # --- Demand lines (cumulative) ---
    line_house = Hh
    line_ev    = Hh + Ev
    line_heat  = Hh + Ev + He
    line_ch    = Hh + Ev + He + Ch   # total demand

    # household stays clean
    ax.plot(t, line_house, color=C["HouseLine"], lw=2.6, label="Household load", zorder=5)

    # make the 3 cumulative ones thicker + with dots
    ax.plot(t, line_ev,   color=C["EVLine"],   lw=3.2, ls="--",
            marker="o", markersize=6, markerfacecolor=C["EVLine"], markeredgewidth=0,
            label="EV", zorder=6)

    ax.plot(t, line_heat, color=C["HeatLine"], lw=3.2, ls="--",
            marker="o", markersize=6, markerfacecolor=C["HeatLine"], markeredgewidth=0,
            label="Heating", zorder=7)

    ax.plot(t, line_ch,   color=C["ChargeLine"], lw=3.6, ls=":",
            marker="o", markersize=6, markerfacecolor=C["ChargeLine"], markeredgewidth=0,
            label="Battery charge", zorder=8)


    # --- SOC on right axis (NEW) ---
    if soc is not None:
        soc_slice = np.asarray(soc[start:end], float)

        # batt_cap_kwh: prefer what you pass in, else fall back to max SOC seen (safe-ish default)
        if batt_cap_kwh is None:
            batt_cap_kwh = float(np.nanmax(soc_slice)) if np.isfinite(np.nanmax(soc_slice)) else 0.0

        batt_cap_kwh = max(float(batt_cap_kwh), 1e-9)
        soc_pct = 100.0 * soc_slice / batt_cap_kwh

        ax2 = ax.twinx()
        ax2.plot(t, soc_pct, color=C["SOC"], lw=2.2, label="SOC [%]")
        ax2.set_ylabel("SOC [%]", fontsize=14)
        ax2.tick_params(axis="y", labelsize=13)   # only if ax2 exists
        ax2.set_ylim(0, 100)

        # Merge legends (ax + ax2)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles = h1 + h2
        labels  = l1 + l2
    else:
        handles, labels = ax.get_legend_handles_labels()

    # --- formatting ---
    ax.set_title(title + f" (hours {start}–{end})(stacked for supply and demand)")
    ax.set_ylabel("kWh per hour", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, axis="y", alpha=0.22)

    ax.set_xticks(t)
    ax.set_xticklabels([f"H{(i % 24):02d}" for i in t], fontsize=13)

    ax.legend(
        handles, labels,
        fontsize=13,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.20),
        columnspacing=1.2,
        handlelength=2.0
    )

    plt.tight_layout()
    save_fig(fig, savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)




def plot_heatmap_month_hour(x, title, year=2019):
    x = np.asarray(x, float)
    idx = pd.date_range(f"{year}-01-01", periods=len(x), freq="H")
    df = pd.DataFrame({"x": x}, index=idx)
    pivot = df.pivot_table(index=df.index.month, columns=df.index.hour, values="x", aggfunc="mean")

    plt.figure(figsize=(12,5))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="Mean kWh per hour")
    plt.yticks(np.arange(12), [f"{m:02d}" for m in range(1,13)])
    plt.xticks(np.arange(0,24,2), [f"{h:02d}" for h in range(0,24,2)])
    plt.xlabel("Hour of day")
    plt.ylabel("Month")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spill_vs_export_price(spill, c_exp):
    spill = np.asarray(spill, float)
    c_exp = np.asarray(c_exp, float)

    plt.figure(figsize=(6,5))
    plt.scatter(c_exp, spill, s=5)
    plt.axvline(0.0)
    plt.xlabel("Export price (EUR/kWh)")
    plt.ylabel("Spill (kWh)")
    plt.title("Spill vs export price")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(c_exp, label="Export price")
    plt.axhline(0.0)
    plt.ylabel("EUR/kWh")
    plt.title("Export price time series")
    plt.tight_layout()
    plt.show()

def plot_cost_breakdown(ac_pv, ac_w, ac_b, res, ann_grid=0.0):
    Ppv = res["P_pv"]; Pw = res["P_w"]; B = res["Battery"]
    cap_pv = ac_pv*Ppv
    cap_w  = ac_w*Pw
    cap_b  = ac_b*B

    labels = ["PV ann.", "Wind ann.", "Battery ann.", "Grid net ann."]
    vals = [cap_pv, cap_w, cap_b, ann_grid]

    plt.figure(figsize=(8,4))
    plt.bar(labels, vals)
    plt.ylabel("EUR/year")
    plt.title("Annualized cost breakdown")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
