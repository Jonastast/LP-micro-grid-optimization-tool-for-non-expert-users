# Hybrid Energy System Optimization & Dispatch Visualization

This repository contains a Python-based techno-economic model for optimizing and visualizing a hybrid electricity system with **PV, wind, battery storage, grid interaction, EV charging, and heating technologies**.

The model performs an **hourly optimization** over a full year (or sub-year) and produces clear dispatch plots showing how electricity supply meets demand, including battery charging, curtailment, and state of charge (SOC).

---

## Features

- Hourly optimization of:
  - Solar PV
  - Onshore wind
  - Battery storage (energy and power constraints)
  - Grid import and export
- Optional demand components:
  - Electric vehicles (EVs)
  - Electric heating (ASHP, GSHP, resistive)
  - District heating (non-electric)
- Cost optimization in real terms:
  - CAPEX (annualized with replacements)
  - Fixed O&M
  - Time-varying grid import prices
  - Optional export revenue (spot prices)
- Visualization tools:
  - Hourly dispatch (stacked supply vs. demand)
  - Battery SOC overlay
  - Average winter and summer day profiles
  - Energy balance stacked bars
  - Curtailment heatmaps

---

## Repository Structure
.
├── main.py # Main orchestration script
├── model.py # Linear optimization model
├── data_io.py # Data loading and preprocessing
├── settings.py # Configuration handling
├── config.yaml # User-defined system setup
├── pv_1kw_profile.csv # Cached PV profile
├── wind_1kw_profile.csv # Cached wind profile
└── README.md



---

## How the Model Works

1. **Load profiles**
   - Household electricity demand
   - Optional EV charging demand
   - Optional heating demand
   - PV and wind generation (1 kW reference profiles)

2. **Scaling and alignment**
   - All time series are aligned to hourly resolution
   - Load profiles are scaled to match annual energy targets

3. **System optimization**
   - Optimal PV, wind, and battery capacities are selected
   - Hourly dispatch of storage and grid is optimized
   - Objective: minimize total annualized system cost

4. **Post-processing**
   - Cost breakdown (CAPEX, O&M, grid)
   - Energy balance and curtailment
   - Battery SOC trajectories

5. **Visualization**
   - Dispatch plots with stacked supply and demand
   - SOC shown as percentage on a secondary axis

---

## Dispatch Plot Interpretation

### Bars (left axis)
- PV, wind, grid import, battery discharge  
- **Solid**: energy used to meet demand  
- **Hatched**: curtailed energy (PV and wind only)

### Lines (left axis)
- Household load
- Household + EV load
- Household + EV + heating load
- Total demand including battery charging

### Line (right axis)
- Battery state of charge (SOC) as % of capacity

---

## Heating Technologies

Supported heating options:
- Air-source heat pump (ASHP)
- Ground-source heat pump (GSHP)
- Electric resistance heating
- District heating (non-electric)

District heating does **not** add electrical demand and can be hidden from dispatch plots when selected.

---

## Configuration

All system parameters are defined in `config.yaml`, including:
- Load targets and profiles
- PV and wind limits
- Battery parameters
- Grid import/export constraints
- Heating technologies
- Economic assumptions (discount rate, lifetimes, O&M)

---

## Running the Model

```bash
python main.py
