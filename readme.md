# LP Microgrid Optimization Tool (PV + Wind + Battery + Optional EV & Heating)

This repository contains a Python tool that **optimizes the sizing and hourly operation** of:
- **Solar PV (kW)**
- **Wind (kW)**
- **Battery energy capacity (kWh)**

…to meet an hourly demand profile at minimum **annualized total cost**, optionally including:
- **EV charging load** (added to the base load)
- **Heating demand** with multiple technology scenarios (GSHP / ASHP / Electric / District Heating)

The core is a **linear program (LP)** solved with PuLP (preferably using **HiGHS** if installed). :contentReference[oaicite:0]{index=0}

---

## What the model does

For each hour, the optimizer enforces an energy balance:

> PV + Wind + Battery discharge + Grid import = Load + Battery charge + Spill + Grid export

It chooses capacities and dispatch to minimize annualized cost:

- Annualized CAPEX for PV, wind, and battery (using CRF + replacements/salvage logic)
- Annualized grid import cost (time-varying supported)
- Optional penalties for curtailment (“spill”) and battery cycling

See the main orchestration and export logic in `main.py`, and the LP formulation in `model.py`. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

## Data sources (built-in fetch)

The tool automatically fetches:
- **PV hourly profile** via PVGIS API (`fetch_pvgis_pv_1kw`) :contentReference[oaicite:3]{index=3}  
- **Wind hourly capacity factors** via Renewables.ninja API (`fetch_ninja_wind`) :contentReference[oaicite:4]{index=4}  

> Renewables.ninja requires an API token (see below).

---

## Repository layout

- `main.py` – end-to-end run: load + EV + heating scenarios + prices + optimize + export :contentReference[oaicite:5]{index=5}  
- `model.py` – LP optimization model (PuLP) :contentReference[oaicite:6]{index=6}  
- `data_io.py` – PVGIS + Renewables.ninja fetch + CSV loaders + grid price builder :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}  
- `settings.py` – config loader + battery power cap helper + token getter :contentReference[oaicite:9]{index=9}  
- `config.yaml` – all run parameters (site, load, EV, heating, costs, grid, economics) :contentReference[oaicite:10]{index=10}  

---

## Installation

### Option A — Conda
```bash
conda create -n microgrid-opt python=3.11 -y
conda activate microgrid-opt

pip install -U pip
pip install numpy pandas matplotlib pyyaml requests pulp

```
### Option B — Pip / venv
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -U pip
pip install numpy pandas matplotlib pyyaml requests pulp

```
Solver (recommended)
PuLP will try HiGHS first, then CBC. Installing HiGHS is strongly recommende

# macOS
brew install highs

# conda
```bash
conda install -c conda-forge highs
```
Required environment variable (Renewables.ninja)

Set your Renewables.ninja token as: export NINJA_TOKEN="your_token_here"

Quickstart

Edit config.yaml to point to your input CSVs and settings

Run:
```python
python main.py
```
main.py reads config.yaml by default.


## Configuration (config.yaml)

The main sections are:

### `site`
Latitude/longitude + timezone + year used to fetch PV/wind profiles and build the time index.

### `load`
- `csv_path`: hourly load CSV (**8760** or **8784** rows)
- `column`: column name (kWh per hour)
- `target_annual_mwh`: the load is scaled to match this annual energy

### `ev` (optional)
If enabled, the EV profile is loaded and multiplied by `vehicles`, then added to the load.

### `heating` (optional)
If enabled, the script evaluates each enabled heating technology:

- `gshp`, `ashp`, `electric`: converts heat demand to extra electric load using COP  
- `district`: heat is not added to electric load; instead a variable district heat tariff is applied  

The tool then selects the **lowest total annual cost** scenario.

### `grid`
Supports:
- Fully islanded mode (no import/export)
- Import-only, export-only, or both
- Time-varying import price built from **spot + transport tariff + national tariffs + VAT**

### `costs`, `economics`, `o_and_m`
Used to compute annualized costs for PV/wind/battery and heating CAPEX.

---

## Input CSV formats

### Load CSV
- Must contain the configured `load.column` (e.g., `load_kWh`)
- Must be **8760** (normal year) or **8784** (leap year) rows

### EV CSV (if enabled)
- Must contain the configured EV column, or it will use the first non-`timestamp` column
- Must be at least as long as the load; extra rows are truncated

### Heating CSVs (if enabled)
- Space heating and water heating are loaded separately, each expected as **8760/8784** rows
- Scaled by `capacity_mwh` *(note: this is multiplied directly with the hourly series)*

### Grid price CSVs
The tool can load hourly prices from CSV where timestamps are parseable either as:
- `"30-09-2025 21:00"` format, or
- ISO-like timestamps (pandas inference)

It filters by `grid.year` if provided.

---

## Outputs

By default outputs go to `run.out_dir` (e.g. `results/`).

### Key files
- `results/summary.csv`  
  One row per scenario run (append or overwrite controlled by config).

- `results/timeseries_<scenario_id>.csv.gz` (or `.csv`)  
  Hourly timeseries including load components, generation, battery flows, SOC, spill, grid import/export.

### Also written in the project root
- `pv_1kw_profile.csv`, `wind_1kw_profile.csv` (fetched profiles saved for inspection)
- `final_import_price_profile.csv` (the final import price signal used)

---

## Notes / common gotchas

- No `NINJA_TOKEN` → wind fetch will fai_
