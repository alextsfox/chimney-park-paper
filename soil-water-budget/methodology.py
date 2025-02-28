"""
Author: Alex Fox, 2024
Physical methods used
"""
from pathlib import Path
import itertools
import warnings
from typing import Literal

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import statsmodels.api as sm
from scipy import stats, odr
from statsmodels.formula.api import ols

########################################################
# Deriving "raw" data
########################################################

def extrapolate_vwc(df, depths=[5, 10, 15, 30, 50], new_depths=[75, 100], pits=["A", "B", "C"], method:Literal["linear","exponential","quadratic", "power"]="linear", interpolate=True):
    """extrapolate soil moisture profile theta(z) to deeper depths using an exponential model
    
    df: dataframe with columns VWC_pit-z where pit is a letter and z is an integer depth
    depths: the depths contained in the dataframe (z's)
    new_depths: depths to extrapolate VWC to
    pits: pit names
    method: the type of regression to use. One of linear, exponential, quadratic, or power.
    interpolate: whether to interpolate missing values in already existing depths

    returns a new dataframe with new columns that contain VWC extrapolated to the new depths given
    """

    vwc_df = df.copy()
    depths = np.asarray(sorted(depths))
    new_depths = np.asarray(sorted(new_depths))
    new_pits = {f"VWC_{pit}-{z}":[] for pit, z in itertools.product(pits, new_depths)}
    interp_pits = {f"VWC_{pit}-{z}":[] for pit, z in itertools.product(pits, depths)}

    # compute theta_sat for each pit
    theta_s_dict = {pit:{depth:None for depth in depths} for pit in pits}
    for pit, depth in itertools.product(pits, depths):
        theta_s_dict[pit][depth] = df[f"VWC_{pit}-{depth}"].max()
    for pit in theta_s_dict:
        theta_s_dict[pit] = max(theta_s_dict[pit].values())
    
    for t, row in tqdm(vwc_df.iterrows(), total=vwc_df.shape[0], mininterval=5):
        for pit in pits:
            vwc = row[[f"VWC_{pit}-{z}" for z in depths]]
            
            func = lambda z: np.nan
            
            match method:
                case "exponential":
                    if len(vwc.dropna()) >= 2:
                        reg = stats.linregress(depths[~vwc.isna()], np.log(vwc.dropna()))
                        slope, intercept, slope2 = reg.slope, np.exp(reg.intercept), None
                        func = lambda z: intercept*np.exp(slope*z)
                    
                case "linear":
                    if len(vwc.dropna()) >= 2:
                        reg = stats.linregress(depths[~vwc.isna()], vwc.dropna())
                        slope, intercept, slope2 = reg.slope, reg.intercept, None
                        func = lambda z: intercept + slope*z
                        
                case "power":
                    if len(vwc.dropna()) >= 2:    
                        reg = stats.linregress(np.log(depths[~vwc.isna()]), np.log(vwc.dropna()))
                        slope, intercept, slope2 = reg.slope, reg.intercept, None
                        func = lambda z: np.exp(intercept + slope*np.log(z))
                        
                case "quadratic":
                    if len(vwc.dropna()) >= 3:
                        reg = (
                            odr.ODR(odr.Data(depths[~vwc.isna()], vwc.dropna()), odr.quadratic)
                            .run()
                        )
                        slope2, slope, intercept = reg.beta
                        func = lambda z: intercept + slope*z + slope2*z**2
                    
                case _:
                    raise ValueError("Method must be one of 'linear', 'loglinear', 'quadratic', 'power'")
                
            for z in new_depths:
                new_vwc = func(z)
                
                if new_vwc < vwc_df[f"VWC_{pit}-{depths[-1]}"].iloc[0]*0.75: new_vwc = np.nan
                elif new_vwc > theta_s_dict[pit]: new_vwc = theta_s_dict[pit]
                new_pits[f"VWC_{pit}-{z}"].append(new_vwc)

            if interpolate:
                for vwc, z in zip(vwc, depths):
                    if np.isnan(vwc) and ~np.isnan(func(z)): 
                        new_vwc = func(z)
                    else: new_vwc = vwc
                    
                    if new_vwc > theta_s_dict[pit]: new_vwc = theta_s_dict[pit]
                    interp_pits[f"VWC_{pit}-{z}"].append(new_vwc)
    
    for k in new_pits: 
        vwc_df[k] = new_pits[k]
    if interpolate:
        for k in interp_pits: 
            vwc_df[k] = interp_pits[k]
        
    return vwc_df

def compute_s(soil_df, depths, pits, field_capacity):
    """Computes the soil moisture storage in each soil layer, and down to each soil layer, in mm
    soil_df: the same format of dataframe used in extrapolate_vwc
    depths: the depths contained in the dataframe (z's)
    pits: the names of each pit
    field_capacity: field capacity of the soil, in m3/m3

    returns: a dataframe that contains new columns S_pit-z of soil moisture storage in mm down to depth z in each pit, and columns S_pit-z1-z2 of soil moisture storage between z1 and z2 in each pit.
    """
    depths = [0] + sorted(depths)
    for pit in pits:
        print(pit)
        for i in range(len(depths) - 1):
            z0 = depths[i]
            z1 = depths[i + 1]
            
            # assume surface soil moisture is the same as 5cm soil moisture
            theta_0 = soil_df[f"VWC_{pit}-{z1}"]
            if i > 0:
                theta_0 = soil_df[f"VWC_{pit}-{z0}"]

            theta_1 = soil_df[f"VWC_{pit}-{z1}"]
            # interpolate 1 missing value
            if i + 1 < len(depths) - 1:
                z2 = depths[i + 2]
                theta_2 = soil_df[f"VWC_{pit}-{z2}"]
                theta_1 = np.where(np.isnan(theta_1), theta_0 + (z1 - z0)*(theta_2 - theta_0)/(z2 - z0), theta_1)
                
            soil_df[f"h_{pit}-{z0}-{z1}"] = 1/2*(theta_0 + theta_1)*(z1 - z0)*0.01  # convert from cm of h2o to m of h2o
            # soil_df[f"h_{pit}-{z0}-{z1}"] = np.where((theta_0 <= field_capacity) & (theta_1 <= field_capacity), soil_df[f"h_{pit}-{z0}-{z1}"], np.nan)
            
            soil_df[f"h_{pit}-{z1}"] = soil_df.filter(regex=f"h_{pit}-[0-9]+-[0-9]+").sum(1, skipna=False)
    
    return soil_df


# First, filter out rain events and saturated soil events with a 1 day buffer after the event.
def filter_rain_events(s_daily, ffill_lim=1, bfill_lim=1, quantile=0.25):
    """
    Finds any rain/saturation events in a soil water storage timeseries and remove them.

    s_daily: a dataframe where each column represents a daily timeseries of soil moisture storage
    ffil_lim, bfill_lim: the number of timesteps to "erase" before and after each found precip/saturation event (days)
    quantile: rain events are identified by jumps in the timeseries. In the distribution D of positive jumps in soil water storage, all soil water storage increases that are more extreme than that defined by the given quantile of D are erased. E.g. the timeseries [5, 4, 3, 2, 10, 9, 8, 15, 10, 5, 6, 7, 8, 12] has 6 jumps: 2->10, 8->15, 5->6, 6->7, 7->8, 8->12, of sizes [8, 7, 1, 1, 1, 1, 4]. If quantile=0.75, all values over the 75th quantile will be removed from the dataset. The 75th quantile for the given jump sizes is 5.5, so the jumps of size 8 and 7 (2->10 and 8->15) will be removed, giving a final timeseries of [5, 4, 3, NA, NA, 9, NA, NA, 10, 5, 6, 7, 8, 12]

    returns a dataset with the jumps converted to nans.
    """
    s_daily = s_daily.copy()
    # filter out rain and saturation events by casting "bad" value to nan
    drainage_cutoff = s_daily.where(s_daily.diff() > 0).diff().quantile(quantile)
    filter = ~(s_daily.diff() < drainage_cutoff)
    # filter out 1 day after each rain and saturation event:
    filter = filter.where(filter)
    filter = filter.ffill(limit=int(ffill_lim)).bfill(limit=int(bfill_lim))
    filter = filter.where(~filter.isna(), False).astype(bool)
    
    s_daily = s_daily.where(~filter)
    return s_daily

# compute the loss of soil moisture/ET for each dry-down event. Dry-down events are identified as runs of non-NA values.
def compute_deltas(s_daily, et_daily):
    """
    Identifies the loss of soil moisture and total evapotranspiration for each contiguous (unbroken) block of soil moisture data in the dataset. If the data has been run through filter_rain_events, then each contiguous block is said to represent one "dry-down event."

    s_daily, et_daily: dataframes containing soil water storage and ET timeseries data, respectively. Must have identical indexes.

    Returns: a containing the start date, end date, total soil moisture loss, and total ET for each dry-down event.
    """
    # compute start/end indices
    runs = {}
    # max_len = (0, None, None)
    # max_c = None
    for c in s_daily:
        x = s_daily[c].values
        i = 0
        starts = []
        ends = []
        for ti in range(1, x.shape[0] - 1):
            if np.isnan(x[ti - 1]) and ~np.isnan(x[ti]):
                starts.append(ti)
            if ~np.isnan(x[ti]) and np.isnan(x[ti + 1]):
                ends.append(ti)
        if len(starts) > len(ends):
            ends.append(ti)
        runs[c] = (starts, ends)
        
    #     for s, e in zip(starts, ends):
    #         if e-s > max_len[0]: 
    #             max_len = (e-s, s, e)
    #             max_c = c
    # print(max_len, max_c)

    # compute delta-h for each dry-down event
    delta_s = {}
    t = {}
    te = {}
    for k, v in runs.items():
        dh = []
        ts = []
        tes = []
        for s, e in zip(*v):
            h_end = s_daily[k].iloc[e]
            h_start = s_daily[k].iloc[s]
            ts.append(s_daily[k].index[s])
            tes.append(s_daily[k].index[e])
            dh.append(h_end - h_start)
        delta_s[k] = dh
        t[k] = ts
        te[k] = tes
    
    # compute total ET for each dry-down event
    total_et = {}
    for k, v in runs.items():
        total_et[k] = [et_daily.iloc[s:e].sum(skipna=False).iloc[0] for s, e in zip(*v)]

    # create dataframe of dry-down fluxes
    depth_lst = []
    pit_lst = []
    total_et_lst = []
    delta_s_lst = []
    t_lst = []
    te_lst = []
    for k in delta_s:
        n = len(total_et[k])
        pit_lst.extend([k[2:].split("-")[0]]*n)
        depth_lst.extend([int(k.split("-")[-1])]*n)
        total_et_lst.extend(total_et[k])
        delta_s_lst.extend(delta_s[k])
        t_lst.extend(t[k])
        te_lst.extend(te[k])
    drydown_fluxes = pd.DataFrame(dict(z=depth_lst, pit=pit_lst, et=total_et_lst, delta_s=delta_s_lst, t_start=t_lst, t_end=te_lst))
    
    return drydown_fluxes


######################################################
# Statistics
######################################################
def compute_drydown_slope(drydown):
    cov = (
        drydown[["z", "et", "delta_s"]]
        .groupby("z")
        .cov().iloc[::2, [1]]
        .reset_index().set_index("z")
        [["delta_s"]]
    )
    var = (
        drydown[["z", "et", "delta_s"]]
        .groupby("z")
        .var()#.iloc[::2, [1]]
        .reset_index().set_index("z")
        [["delta_s"]]
    )

    return (-cov/var).rename(columns=dict(delta_s="Slope"))
    
# def compute_slopes(drydown_fluxes):
#     """
#     Compute the "drydown slope" for a dataframe of drydown fluxes: the closure of the soil moisture-et water budget

#     drydown_fluxes: a long-format dataframe:
#         * a column Site indicating site (key NF, UF, SF)
#         * a column Height indicating height (key Ecosystem, Understory)
#         * a column z indicating soil depth (key 5, 10, 15, 30, 50, 75, 100)
#         * a column delta_s of soil water loss per drydown event
#         * a column et of total ET per drydown event

#     returns: a dataframe giving the slope of the dS ~ SumET relationship for each site, height, and depth.
#     """
#     sites = ["NF", "UF", "SF"]
#     heights = ["Ecosystem", "Understory"]
#     depths = [5, 10, 15, 30, 50, 75, 100]
    
#     slope_dict = {
#         ("Ecosystem", "NF"):[],
#         ("Ecosystem", "UF"):[],
#         ("Ecosystem", "SF"):[],
#         ("Understory", "NF"):[],
#         ("Understory", "UF"):[],
#     }

#     stderr_dict = {
#         ("Ecosystem", "NF"):[],
#         ("Ecosystem", "UF"):[],
#         ("Ecosystem", "SF"):[],
#         ("Understory", "NF"):[],
#         ("Understory", "UF"):[],
#     }
        
#     for (h, s), z in itertools.product(slope_dict, depths):
#         h_old = h
#         if h == "Ecosystem" and s == "SF": h = "Understory"

#         # # avoid "pooling" in the springtime: limit SF analysis to June+
#         # if s == "SF": months = [7, 8, 9, 10]
#         # else: months = [5, 6, 7, 8, 9, 10]
            
#         df = (
#             drydown_fluxes
#             .query("Height == @h and z == @z and Site == @s")
#             .copy()
#             .dropna()
#         )
#         h = h_old
#         # m = ols("et ~ delta_s", data=df).fit()
#         # summary_table = m.summary().tables[1]
#         # slope = float(summary_table.data[2][1])
#         # stderr = float(summary_table.data[2][2])*1.96
#         try:
#             m = stats.linregress(df["delta_s"], df["et"])
#             slope, stderr = m.slope, m.stderr
#             # print(slope, stderr)
#         except ValueError:
#             slope, stderr = np.nan, np.nan
#         if stderr <= 0: stderr = np.nan
#         slope_dict[(h, s)].append(slope)
#         stderr_dict[(h, s)].append(stderr)
    
#     slopes = pd.DataFrame(slope_dict)
#     slopes["Depth"] = depths
#     slopes = slopes.set_index("Depth")

#     stderrs = pd.DataFrame(stderr_dict)
#     stderrs["Depth"] = depths
#     stderrs = stderrs.set_index("Depth")
#     return slopes, stderrs

def compute_slopes(drydown_fluxes, months=[4, 5, 6, 7, 8, 9, 10]):
    """
    Compute the "drydown slope" for a dataframe of drydown fluxes: the closure of the soil moisture-et water budget

    drydown_fluxes: a long-format dataframe:
        * a column Site indicating site (key NF, UF, SF)
        * a column Height indicating height (key Ecosystem, Understory)
        * a column z indicating soil depth (key 5, 10, 15, 30, 50, 75, 100)
        * a column delta_s of soil water loss per drydown event
        * a column et of total ET per drydown event

    returns: a dataframe giving the slope of the dS ~ SumET relationship for each site, height, and depth.
    """
    sites = ["NF", "UF", "SF"]
    heights = ["Ecosystem", "Understory"]
    depths = [5, 10, 15, 30, 50, 75, 100]
    
    slope_dict = {
        ("Ecosystem", "NF"):[],
        ("Ecosystem", "UF"):[],
        ("Ecosystem", "SF"):[],
        ("Understory", "NF"):[],
        ("Understory", "UF"):[],
    }

    stderr_dict = {
        ("Ecosystem", "NF"):[],
        ("Ecosystem", "UF"):[],
        ("Ecosystem", "SF"):[],
        ("Understory", "NF"):[],
        ("Understory", "UF"):[],
    }
        
    for (h, s), z in itertools.product(slope_dict, depths):
        h_old = h
        if h == "Ecosystem" and s == "SF": h = "Understory"

        # # avoid "pooling" in the springtime: limit SF analysis to June+
        # if s == "SF": months = [7, 8, 9, 10]
        # else: months = [5, 6, 7, 8, 9, 10]
            
        df = (
            drydown_fluxes
            .query("Height == @h and z == @z and Site == @s")
            .copy()
            .dropna()
        )
        h = h_old
        # m = ols("et ~ delta_s", data=df).fit()
        # summary_table = m.summary().tables[1]
        # slope = float(summary_table.data[2][1])
        # stderr = float(summary_table.data[2][2])*1.96
        try:
            m = stats.linregress(df["delta_s"], df["et"])
            slope, stderr = m.slope, m.stderr
            # print(slope, stderr)
        except ValueError:
            slope, stderr = np.nan, np.nan
        if stderr <= 0: stderr = np.nan
        slope_dict[(h, s)].append(slope)
        stderr_dict[(h, s)].append(stderr)
    
    slopes = pd.DataFrame(slope_dict)
    slopes["Depth"] = depths
    slopes = slopes.set_index("Depth")

    stderrs = pd.DataFrame(stderr_dict)
    stderrs["Depth"] = depths
    stderrs = stderrs.set_index("Depth")
    return slopes, stderrs

######################################################
# Biometeorological equations
######################################################
def delta_tetens(Tc):
    """slope of the vapor pressure curve at temperature Tc in °C"""
    return np.where(
        Tc >= 0, 
        (2503.08*np.exp((17.27*Tc)/(Tc + 237.3)))/(Tc + 237.3)**2,
        (3548.11*np.exp((21.88*Tc)/(Tc + 265.5)))/(Tc + 265.5)**2
    )

def penman_monteith(T_c, Rn_W, G_W, vpd_kPa, LAI, u, zm, zc, Pa_kPa):
    """Penman monteith equation for PET. Uses M-O theory to derive atmospheric conductance, and uses a rough approximation for canopy conductance based on LAI and maximum stomatal conductance

    T_c: temp in C
    Rn_W: net radiation in W/m2
    G_W: ground heat flux in W/m2
    vpd_kPA: VPD in ... well...kPa
    LAI: leaf area index
    u: windspeed
    zm: measurement height
    zc: canopy height
    Pa_kPa: air pressure in kPa

    returns: PET in mm/s
    """
    d = delta_tetens(T_c)*1000  # Pa C-1
    y = 1.005*Pa_kPa / (2.45*0.622)  # Pa C-1
    rho_a = Pa_kPa*1000 / (287.05*(T_c+273.15))  # kg m-3

    zd, z0 = 0.7*zc, 0.1*zc
    Ca = u / (6.25*np.log((zm-zd)/z0)**2)  # m s-1
    Cc = 0.5*LAI*0.005  # m s-1
    Lv = 2453e3 # kJ m-3

    # print(Ca, Cc)
    
    rad_term = d*(Rn_W - G_W)
    atm_term = rho_a*1005*vpd_kPa*1000*Ca
    denom = (d + y*(1+Ca/Cc))#*Lv

    return (rad_term + atm_term)/denom * 1000/(2257e3*997)  # mm/s

def FAO56_PM(T_c, Rn_MJ_d, G_MJ_d, vpd_kPa, Pa_kPa, u):
    """FAO 1956 Penman monteith equation for PET. 

    T_c: temp in C
    Rn_MJ_d: net radiation in MJ/m2/d
    G_MJ_d: ground heat flux in MJ/m2/d
    vpd_kPA: VPD in ... well...kPa
    Pa_kPa: air pressure in kPa
    u: windspeed at 2m height

    returns: PET in mm/s
    """
    y = 1.005*Pa_kPa / (2.45*0.622) * 1e-3  # kPa C-1
    d = delta_tetens(T_c)  # kPa C-1
    rad_term = 0.408*d*(Rn_MJ_d - G_MJ_d)
    atm_term = y*900/(T_c+273) * u*vpd_kPa
    denom = d + y*(1+0.34*u)
    return (rad_term + atm_term)/denom / 86400 # mm/s

####################################################
# Moire soil moisture budget stuff
####################################################
def fill_best_depths(best_depths_rounded, ref_data, c):
    """
    helper function to populate a timeseries dataframe with uptake depth values
    """
    return (
        best_depths_rounded[[c, "TIMESTAMP"]]
        .merge(ref_data.reset_index()[["TIMESTAMP"]], on="TIMESTAMP", how="outer")
        .set_index("TIMESTAMP")
        .ffill().bfill()
    )

def compute_usable_s(data, depths, pits):
    """
    computes VWC and S only within the soil water uptake zone
    """
    pits = list(pits)
    useful_s = np.full((data.shape[0], len(pits) + 1), np.nan)
    for i, t in enumerate(data.index):
        z = int(depths.loc[t].iloc[0])
        useful_s[i] = [data.loc[t, f"h_{p}-{z}"] for p in pits] + [z]
    useful_s = pd.DataFrame(data=useful_s, columns=[f"S_{p}" for p in pits] + ["uptake_depth"]).set_index(data.index)
    useful_s[[f"VWC_{p}" for p in pits]] = (useful_s[[f"S_{p}" for p in pits]].values/useful_s[["uptake_depth"]].values)*100
    useful_s["Seff"] = useful_s.filter(regex="^S").mean(1, skipna=True)
    useful_s["VWCeff"] = useful_s.filter(regex="^VWC").mean(1, skipna=True)
    return useful_s

def YS_to_datetime(ys):
    """helper function to convert 2-season year-season value (2019.0, 2019.5)
    to a datetime"""
    season = ys % 1
    month = 7
    month = np.where(season == 0, 1, 7)
    year = ys.astype(int)
    date_strings = [f"{y}-{m}" for y, m in zip(year, month)]
    return pd.to_datetime(date_strings)

def round_depths(x, depths):
    """helper function to round an array x to the nearest values in a reference array depths"""
    # finds the nearest depth for each value in x
    return np.where(
        ~np.isnan(x), 
        depths[[(np.abs(ix - depths)).argmin() for ix in x]], 
        np.nan
    )

def compute_uptake_depth(drydown_fluxes):
    """compute the rooting zone/soil water extraction uptake depth given a dataframe of drydown_fluxes. Must have a grouping column 'xax'"""
    warnings.simplefilter("ignore")

    best_depths = []
    best_depth_stderrs = []
    for x in sorted(drydown_fluxes.xax.unique()):
        # slopes, stderrs = compute_slopes(drydown_fluxes.loc[drydown_fluxes.t_start.dt.year == year])
        slopes, stderrs = compute_slopes(drydown_fluxes.loc[(drydown_fluxes.xax == x) & (drydown_fluxes.t_start.dt.month.isin([4, 5, 6, 7, 8, 9, 10]))])
        new_headers = ["NF_eco", "UF_eco", "SF_eco", "NF_und", "UF_und"]
    
        # randomly sample possible slopes from the regression statistics
        slopes = pd.DataFrame(data=slopes.values, columns=new_headers).set_index(slopes.index)
        stderrs = pd.DataFrame(data=stderrs.values, columns=new_headers).set_index(stderrs.index)
        random_slopes = stats.norm.rvs(
            loc=np.where(np.isnan(stderrs), -1e6, slopes), 
            scale=np.where(np.isnan(stderrs), 0.000000001, stderrs), 
            size=(100, *slopes.shape)
        )
        random_slopes = np.where(random_slopes < -1e3, np.nan, random_slopes)
        
        # for each slope sample:
        # * interpolate slopes to 1cm depth resolution
        # * compute the optimal depth, where slope ~1
        best_depth = []
        for i, islopes in enumerate(random_slopes):
            # interpolate slopes to 1cm resolution
            new_depths = pd.DataFrame(dict(Depth=np.arange(100))).set_index("Depth")
            islopes_smooth = (
                pd.DataFrame(data=islopes, columns=new_headers).set_index(slopes.index)
                .merge(new_depths, left_index=True, right_index=True, how="right")
                .interpolate()
            )
            best_depth.append(islopes_smooth.index[np.argsort((islopes_smooth - 1).abs(), axis=0)[0]])
        # record the mean and std of the optimal depth across all samples
        best_depth = np.array(best_depth)
        best_depths.append(best_depth.mean(0))
        best_depth_stderrs.append(best_depth.std(0))
    # load results into dataframe
    # best_depths = pd.DataFrame(data=best_depths, columns=new_headers).set_index(sorted(drydown_fluxes.t_start.dt.year.unique()))
    # best_depth_stderrs = pd.DataFrame(data=best_depth_stderrs, columns=new_headers).set_index(sorted(drydown_fluxes.t_start.dt.year.unique()))
    best_depths = pd.DataFrame(data=best_depths, columns=new_headers).set_index(np.array(sorted(drydown_fluxes.xax.unique())))
    best_depth_stderrs = pd.DataFrame(data=best_depth_stderrs, columns=new_headers).set_index(np.array(sorted(drydown_fluxes.xax.unique())))
    
    best_depths = best_depths.where(best_depth_stderrs > 0)
    best_depth_stderrs = best_depth_stderrs.where(best_depth_stderrs > 0)

    best_depths_rounded = best_depths.copy()
    for c in best_depths_rounded:
        best_depths_rounded[c] = round_depths(best_depths_rounded[c], np.array([100, 75, 50, 30, 15, 10, 5]))
    best_depths_rounded["TIMESTAMP"] = YS_to_datetime(best_depths_rounded.index)
    
    warnings.resetwarnings()
    return best_depths, best_depth_stderrs, best_depths_rounded

    