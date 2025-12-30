import pandas as pd
import requests
from datetime import datetime

#### Fetch PV Data #### 
def fetch_pvgis_pv_1kw(lat, lon, tilt=30, azim=0, loss_pct=14, year=2019, timezone="Europe/Copenhagen"):
    """
    Returns:
        pd.Series of PV power output (kW per 1 kWp installed),
        indexed by hourly timestamps (timezone-corrected).
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "raddatabase": "PVGIS-ERA5",  # recent years
        "pvtechchoice": "crystSi",
        "pvcalculation": 1,           # <-- KEY: compute PV output
        "peakpower": 1.0,
        "loss": loss_pct,             # %
        "mountingplace": "free",
        "angle": tilt,
        "aspect": azim % 360,         # PVGIS: 0=south, 90=west, 180=north, 270=east
        "startyear": year,
        "endyear": year,
        "outputformat": "json",
        "usehorizon": 1,
        "jsontimes": 1,
        "components": 1               # include component irradiances as well
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data["outputs"]:
        raise RuntimeError(f"No hourly data. Outputs keys: {list(data.get('outputs', {}).keys())}")

    hourly = pd.DataFrame(data["outputs"]["hourly"])
    time_col = "time" if "time" in hourly.columns else "time(UTC)"
    hourly[time_col] = pd.to_datetime(hourly[time_col], format="%Y%m%d:%H%M", utc=True).dt.tz_convert(timezone)
    hourly = hourly.set_index(time_col).sort_index()

    # Prefer PV power column if present
    if "P" in hourly.columns:
        return hourly["P"] / 1000.0  # W → kW
    elif "P_AC" in hourly.columns:
        return hourly["P_AC"] / 1000.0

    # Fallback: compute an approximate PV power from irradiance if P is still missing
    if all(col in hourly.columns for col in ["Gb(i)", "Gd(i)", "Gr(i)"]):
        G_poa = hourly["Gb(i)"] + hourly["Gd(i)"] + hourly["Gr(i)"]  # W/m² on plane of array
        # Simple approx: 1 kWp produces G_poa/1000 kW at STC, apply system losses
        derate = 1.0 - (loss_pct / 100.0)
        pv_kw_approx = (G_poa / 1000.0) * derate
        return pv_kw_approx

    raise RuntimeError(f"PV power column not found and cannot approximate. Columns: {hourly.columns.tolist()}")


def fetch_ninja_wind(lat, lon, years=2019, turbine="Vestas V110 2000", 
                     token=None, height=100, timezone="Europe/Copenhagen", timeout=60):
    """
    Fetch wind capacity factor(s) from Renewables.ninja API.
    If `years` is a single int -> fetch one year.
    If `years` is a list/tuple -> fetch multiple years and return average profile.

    Returns:
        pd.Series (hourly capacity factor for 1 kW installed)
    """
    if token is None:
        raise ValueError("Provide Renewables.ninja API token via config or environment variable.")

    if isinstance(years, int):
        years = [years]  # single-year fallback

    all_series = []

    headers = {"Authorization": f"Token {token}"}
    url = "https://www.renewables.ninja/api/data/wind"

    for yr in years:
        params = {
            "lat": lat,
            "lon": lon,
            "date_from": f"{yr}-01-01",
            "date_to": f"{yr}-12-31",
            "capacity": 1.0,
            "turbine": turbine,
            "height": height,
            "format": "json",
            "raw": True
        }
        print(f"Fetching Renewables.ninja wind data for {yr}...")
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame.from_dict(data["data"], orient="index")

        # --- Safe timestamp parsing ---
        raw_idx = pd.Index(df.index.astype(str))
        try:
            # Detect if timestamps look like UNIX milliseconds
            if raw_idx.str.match(r"^\d{11,13}$").all():
                dt_idx = pd.to_datetime(raw_idx.astype("int64"), unit="ms", utc=True)
            else:
                dt_idx = pd.to_datetime(raw_idx, utc=True, errors="coerce")

            # Drop invalid or NaT rows
            dt_idx = dt_idx.dropna()
            df.index = dt_idx.tz_convert(timezone)
        except Exception as e:
            print(f"Warning: Fallback datetime parsing due to {e}")
            df.index = pd.to_datetime(raw_idx, errors="coerce", utc=True).tz_convert(timezone)


        if "capacity_factor" in df.columns:
            s = df["capacity_factor"]
        elif "electricity" in df.columns:
            s = df["electricity"]
        else:
            raise RuntimeError(f"Unexpected Ninja columns for {yr}: {df.columns.tolist()}")

        all_series.append(s)

    # Align and average
    df_all = pd.concat(all_series, axis=1)
    df_all.columns = [str(y) for y in years]
    s_mean = df_all.mean(axis=1)

    # If multiple years, reindex to a "typical year" (8760 hours Jan–Dec)
    if len(years) > 1:
        typical_year = years[-1]  # use last year for datetime reference
        start = pd.Timestamp(f"{typical_year}-01-01 00:00", tz=timezone)
        s_mean.index = pd.date_range(start=start, periods=len(s_mean), freq="H", tz=timezone)

    return s_mean


####### Load profile scaling ######
def scale_load_profile_annual(load_series, target_annual_mwh):
    """
    Scale an hourly load profile (kWh per hour) to hit a target annual energy (MWh).
    Returns (scaled_series, scale_factor, original_annual_mwh).
    """
    scaled = load_series.astype(float).clip(lower=0).copy()
    current_annual_mwh = scaled.sum() / 1000.0  # kWh -> MWh
    if current_annual_mwh <= 0:
        raise ValueError("Current annual energy is non-positive; cannot scale.")
    sf = target_annual_mwh / current_annual_mwh
    return scaled * sf, sf, current_annual_mwh

def load_hourly_grid_prices(
    path,
    col="RealPrice_EUR_kWh",
    time_col="HourUTC",
    year=2024,
    timezone="Europe/Copenhagen",
):
    """
    Load hourly grid prices from a CSV.

    Returns:
        pd.Series (float) indexed by localized timestamps, filtered to a given year.
    """
    df = pd.read_csv(path)

    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in {path}.")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}.")

    # We support both the old format "30-09-2025 21:00"
    # and ISO-like formats "2024-01-01 00:00" / "2024-01-01 00:00:00"
    time_str = df[time_col].astype(str)

    try:
        # Try the old explicit format first (keeps existing files working)
        df[time_col] = pd.to_datetime(time_str, format="%d-%m-%Y %H:%M", utc=True)
    except Exception:
        # Fallback: let pandas infer (handles ISO etc.)
        df[time_col] = pd.to_datetime(time_str, utc=True, errors="coerce")
        if df[time_col].isna().any():
            bad = time_str[df[time_col].isna()].iloc[0]
            raise ValueError(
                f"Could not parse datetime '{bad}' in column '{time_col}' from {path}."
            )

    # Filter to a specific year if requested
    if year is not None:
        df = df[df[time_col].dt.year == year]

    df = df.set_index(time_col).sort_index()

    if timezone:
        df.index = df.index.tz_convert(timezone)

    price_series = df[col].astype(float)
    price_series.name = "grid_import_price_eur_per_kwh"
    return price_series


def build_grid_price_profile_from_components(
    grid_cfg: dict,
    year: int | None = None,
    timezone: str = "Europe/Copenhagen",
):
    """
    Build final grid import price (EUR/kWh) from:
      - spot price CSV (EUR/kWh)
      - KONSTANT hourly transport tariff CSV (EUR/kWh)
      - national per-kWh tariffs in DKK/kWh (config)
      - VAT ("moms") percentage

    Returns:
        pd.Series indexed by timestamps (tz-aware) with name
        'grid_import_price_eur_per_kwh'.
    """

    # 1) Spot price (EUR/kWh)
    spot = load_hourly_grid_prices(
        path=grid_cfg["spot_price_csv"],
        col=grid_cfg.get("spot_price_column", "SpotPrice_EUR_kWh"),
        time_col=grid_cfg.get("spot_price_time_col", "HourUTC"),
        year=year,
        timezone=timezone,
    )

    # 2) KONSTANT hourly transport tariff (EUR/kWh)
    transport = load_hourly_grid_prices(
        path=grid_cfg["transport_price_csv"],
        col=grid_cfg.get("transport_price_column", "Tariff_EUR_kWh"),
        time_col=grid_cfg.get("transport_price_time_col", "Datetime"),
        year=year,
        timezone=timezone,
    )

    # Align indices
    df = pd.concat({"spot": spot, "transport": transport}, axis=1).dropna()

    base_eur_per_kwh = df["spot"] + df["transport"]

    # 3) Extra tariffs + tax in DKK/kWh → EUR/kWh
    dkk_per_eur = float(grid_cfg.get("dkk_per_eur", 7.45))
    extra_dkk_per_kwh = (
        float(grid_cfg.get("transmission_nettariff_dkk_per_kwh", 0.0))
        + float(grid_cfg.get("transmission_systemtariff_dkk_per_kwh", 0.0))
        + float(grid_cfg.get("electricity_tax_dkk_per_kwh", 0.0))
    )
    extra_eur_per_kwh = extra_dkk_per_kwh / dkk_per_eur

    total_eur_per_kwh = base_eur_per_kwh + extra_eur_per_kwh

    # 4) Add VAT ("moms")
    vat_pct = float(grid_cfg.get("vat_pct", 0.25))
    total_eur_per_kwh *= (1.0 + vat_pct)

    total_eur_per_kwh.name = "grid_import_price_eur_per_kwh"
    return total_eur_per_kwh


def load_scaled_heat_profile(path, column, capacity_mwh):
    """
    Load an hourly heat profile normalized to 1 MW and scale by capacity_mw.
    Assumes the CSV column is in kWh per hour for 1 MW.
    """
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {path}.")
    s = pd.Series(df[column].astype(float), name=column)
    assert len(s) in (8760, 8784), "Heat profile must be 8760/8784 hours"
    return s * float(capacity_mwh)