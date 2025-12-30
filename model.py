import numpy as np
import pulp
import shutil

def optimize_hourly(
    load, g_pv, g_w,
    ac_pv, ac_w, ac_b,
    c_imp=None, c_exp=None,
    annual_scale=1.0,
    max_grid_import_frac=1.0,
    eta=0.95,
    A_roof=None, eta_pv=None, P_w_max=None,
    soc_cycle=True, min_autonomy_hours=0.0,
    spill_penalty=0.01, cycle_penalty=0.0,
    P_ch_max=None, P_dis_max=None,
    soc_min_frac=0.10, soc_max_frac=0.95,
    allow_export=False,
):
    """
    Minimize annualized system cost (EUR/year):
      ac_pv*Ppv + ac_w*Pw + ac_b*B + annual_scale*sum(c_imp*G_imp - c_exp*G_exp)
    Subject to hourly power balance, SOC dynamics/limits, optional power caps, and sizing caps.
    """

    # --- inputs as arrays/floats ---
    load = np.asarray(load, dtype=float)
    g_pv = np.asarray(g_pv, dtype=float)
    g_w  = np.asarray(g_w,  dtype=float)

    ac_pv = float(ac_pv); ac_w = float(ac_w); ac_b = float(ac_b)

    # c_imp can be scalar or array-like of length T
    if c_imp is None:
        c_imp_arr = None
        c_imp_is_array = False
    else:
        c_imp_arr = np.asarray(c_imp, dtype=float)
        if c_imp_arr.shape == ():  # scalar
            c_imp_arr = float(c_imp_arr)
            c_imp_is_array = False
        else:
            if c_imp_arr.shape[0] != len(load):
                raise ValueError(f"c_imp length {c_imp_arr.shape[0]} does not match T={len(load)}")
            c_imp_is_array = True

    # c_exp can also be scalar or array-like of length T (like c_imp)
    if c_exp is None:
        c_exp_arr = None
        c_exp_is_array = False
    else:
        c_exp_arr = np.asarray(c_exp, dtype=float)
        if c_exp_arr.shape == ():  # scalar
            c_exp_arr = float(c_exp_arr)
            c_exp_is_array = False
        else:
            if c_exp_arr.shape[0] != len(load):
                raise ValueError(f"c_exp length {c_exp_arr.shape[0]} does not match T={len(load)}")
            c_exp_is_array = True

    allow_export = bool(allow_export)

    annual_scale = float(annual_scale)
    max_grid_import_frac = 1.0 if max_grid_import_frac is None else float(max_grid_import_frac)
    eta  = float(eta)
    spill_penalty  = float(spill_penalty)
    cycle_penalty  = float(cycle_penalty)


    A_roof   = None if A_roof   is None else float(A_roof)
    eta_pv   = None if eta_pv   is None else float(eta_pv)
    P_w_max  = None if P_w_max  is None else float(P_w_max)
    P_ch_max = None if P_ch_max is None else float(P_ch_max)
    P_dis_max= None if P_dis_max is None else float(P_dis_max)
    soc_min_frac = float(soc_min_frac)
    soc_max_frac = float(soc_max_frac)
    min_autonomy_hours = float(min_autonomy_hours)

    T = len(load)
    assert T == len(g_pv) == len(g_w), "Profiles must be same length"

    # Split round-trip efficiency symmetrically
    eta_ch = eta_dis = np.sqrt(eta)

    # --- Model ---
    m = pulp.LpProblem("Hourly_Microgrid_LP", pulp.LpMinimize)

    # Sizes
    Ppv = pulp.LpVariable("P_pv_kw", lowBound=0)
    Pw  = pulp.LpVariable("P_w_kw",  lowBound=0)
    B   = pulp.LpVariable("Battery_kWh", lowBound=0)

    # Hourlies
    ch   = pulp.LpVariable.dicts("ch",   range(T), lowBound=0)  # kWh charged this hour
    dis  = pulp.LpVariable.dicts("dis",  range(T), lowBound=0)  # kWh discharged this hour
    soc  = pulp.LpVariable.dicts("soc",  range(T), lowBound=0)  # kWh
    spill= pulp.LpVariable.dicts("spill",range(T), lowBound=0)  # kWh curtailed
    G_imp = pulp.LpVariable.dicts("grid_import", range(T), lowBound=0)
    G_exp = pulp.LpVariable.dicts("grid_export", range(T), lowBound=0)

    # Objective: annualized system cost (EUR/year)
    grid_term = 0
    if c_imp_arr is not None:
        if c_imp_is_array:
            if c_exp_arr is None:
                # import cost only
                grid_term = annual_scale * pulp.lpSum(
                    c_imp_arr[t] * G_imp[t] for t in range(T)
                )
            elif c_exp_is_array:
                grid_term = annual_scale * pulp.lpSum(
                    c_imp_arr[t] * G_imp[t] - c_exp_arr[t] * G_exp[t] for t in range(T)
                )
            else:  # export scalar, import array
                grid_term = annual_scale * pulp.lpSum(
                    c_imp_arr[t] * G_imp[t] - c_exp_arr * G_exp[t] for t in range(T)
                )
        else:
            if c_exp_arr is None:
                grid_term = annual_scale * pulp.lpSum(
                    c_imp_arr * G_imp[t] for t in range(T)
                )
            elif c_exp_is_array:
                grid_term = annual_scale * pulp.lpSum(
                    c_imp_arr * G_imp[t] - c_exp_arr[t] * G_exp[t] for t in range(T)
                )
            else:
                grid_term = annual_scale * pulp.lpSum(
                    c_imp_arr * G_imp[t] - c_exp_arr * G_exp[t] for t in range(T)
                )


    m += (
        ac_pv * Ppv
        + ac_w  * Pw
        + ac_b  * B
        + grid_term
        + annual_scale * spill_penalty  * pulp.lpSum(spill[t] for t in range(T))
        + annual_scale * cycle_penalty  * pulp.lpSum(ch[t] + dis[t] for t in range(T))
    )

    # Sizing caps (optional)
    if A_roof is not None and eta_pv is not None:
        m += Ppv <= A_roof * eta_pv
    if P_w_max is not None:
        m += Pw <= P_w_max

    # Autonomy (optional)
    if min_autonomy_hours > 0:
        L_ref = float(load.mean())
        m += B >= min_autonomy_hours * L_ref

    # Hourly constraints
    for t in range(T):
        # Power balance in the hour
        m += (Ppv * g_pv[t] + Pw * g_w[t] + dis[t] + G_imp[t] == load[t] + ch[t] + spill[t] + G_exp[t])
        
        # If export is disabled, force G_exp[t] = 0
        if not allow_export:
            m += G_exp[t] == 0

       # if soc[t] <= soc_max_frac * B:
        #     m += spill[t] >= 0

        # SOC upper and lower limits
        m += soc[t] <= soc_max_frac * B
        m += soc[t] >= soc_min_frac * B

        # SOC dynamics
        if t < T - 1:
            m += soc[t+1] == soc[t] + eta_ch * ch[t] - (1.0/eta_dis) * dis[t]
        else:
            if soc_cycle:
                m += soc[0] == soc[t] + eta_ch * ch[t] - (1.0/eta_dis) * dis[t]

        # Power caps
        if P_ch_max is not None:
            m += ch[t]  <= P_ch_max
        if P_dis_max is not None:
            m += dis[t] <= P_dis_max

        # Physical headroom
        m += ch[t]  <= soc_max_frac * B - soc[t]
        m += dis[t] <= soc[t] - soc_min_frac * B

    # Optional: Annual grid import limit over simulated window
    if max_grid_import_frac < 1.0 and c_imp is not None:
        total_load = float(load.sum())
        m += pulp.lpSum(G_imp[t] for t in range(T)) <= max_grid_import_frac * total_load

    # Solve (prefer HiGHS)
    if shutil.which("highs"):
        solver = pulp.HiGHS_CMD(msg=False, timeLimit=600)
    elif shutil.which("cbc"):
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=600)
    else:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=600)
    m.solve(solver)
    status = pulp.LpStatus[m.status]

    # Extract solution
    Ppv_v = float(pulp.value(Ppv))
    Pw_v  = float(pulp.value(Pw))
    B_v   = float(pulp.value(B))

    soc_arr  = np.array([pulp.value(soc[t])   for t in range(T)], dtype=float)
    ch_arr   = np.array([pulp.value(ch[t])    for t in range(T)], dtype=float)
    dis_arr  = np.array([pulp.value(dis[t])   for t in range(T)], dtype=float)
    spill_arr= np.array([pulp.value(spill[t]) for t in range(T)], dtype=float)
    gimp_arr = np.array([pulp.value(G_imp[t]) for t in range(T)], dtype=float)
    gexp_arr = np.array([pulp.value(G_exp[t]) for t in range(T)], dtype=float)

    pv_gen   = Ppv_v * g_pv
    wind_gen = Pw_v  * g_w

    return {
        "status": status,
        "P_pv": Ppv_v,
        "P_w": Pw_v,
        "Battery": B_v,
        "Cost": float(pulp.value(m.objective)),
        "soc_profile": soc_arr,
        "ch_profile": ch_arr,
        "dis_profile": dis_arr,
        "spill_profile": spill_arr,
        "grid_import_profile": gimp_arr,
        "grid_export_profile": gexp_arr,
        "pv_gen_profile": pv_gen,
        "wind_gen_profile": wind_gen,
        "energy": {
            "pv_gen": float(pv_gen.sum()),
            "wind_gen": float(wind_gen.sum()),
            "spill": float(spill_arr.sum()),
            "charge": float(ch_arr.sum()),
            "discharge": float(dis_arr.sum()),
            "load": float(load.sum()),
            "imp_kwh": float(gimp_arr.sum()),
            "exp_kwh": float(gexp_arr.sum()),   
        },
    }

