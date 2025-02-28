from pathlib import Path
import itertools

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

def load_flux_data(path):
    flux = (
        pd.read_csv(path, sep="\t", skiprows=[1], na_values=-9999, parse_dates=["Date Time"])
        [["Date Time", "LE_uStar_f", "NEE_uStar_f", "GPP_uStar_f", "Reco_uStar", "Tair_f", "VPD_f"]]
        .rename(columns={"Date Time":"TIMESTAMP", "LE_uStar_f":"LE",  "NEE_uStar_f":"NEE", "GPP_uStar_f":"GPP", "Reco_uStar":"Reco", "Tair_f":"TA", "VPD_f":"VPD"})
    )
    
    # convert W/m2 latent heat flux to m/day evapotranspiration
    lambda_v = 2260e3  # J kg-1
    rho_w = 997 # kg m-3
    flux["ET"] = flux["LE"]*(1/lambda_v)*(1/rho_w)*86400
    # convert µmol/m2/s cflux to kgC m-2 d-1
    M_C = 0.012  # kg mol-1
    flux["NEE"] = flux["NEE"] * 1e-6*86400*M_C
    flux["GPP"] = flux["GPP"] * 1e-6*86400*M_C
    flux["Reco"] = flux["Reco"] * 1e-6*86400*M_C
    
    flux = flux[["TIMESTAMP", "ET", "NEE", "GPP", "Reco", "VPD", "TA"]].sort_values("TIMESTAMP").set_index("TIMESTAMP")
    return flux

def get_rg_ws(fullout_path, reddyproc_path, rnet_path):
    reddyproc = (
        pd.read_csv(
            # Path('/project/bbtrees/afox18/EddyProConfigEditor/workflows/Postproc_all') / 'BB-NF-17m_proc_.tsv', 
            reddyproc_path,
            sep="\t", skiprows=[1], na_values=-9999, parse_dates=["Date Time"]
        )
        .rename(columns={"Date Time":"TIMESTAMP", "PotRad_NEW":"PotRad"})
        [["TIMESTAMP", "Rg", "PotRad"]]
    )
    
    fullout=(
        # pd.read_csv(Path('/project/bbtrees/afox18/EddyProConfigEditor/workflows/Postproc_all/BB-NF-3m_eddypro_fullout.csv'), parse_dates=["date_time"])
        pd.read_csv(fullout_path, parse_dates=["date_time"])
        .rename(columns=dict(wind_speed="WS", date_time="TIMESTAMP"))
        [["TIMESTAMP", "WS"]]
    )

    rnet = pd.read_csv(rnet_path, parse_dates=["TIMESTAMP"])
    
    compare = (
        reddyproc
        .merge(fullout, on="TIMESTAMP", how="outer")
        .merge(rnet, on="TIMESTAMP", how="outer")
        .set_index("TIMESTAMP")
        .replace(-9999, np.nan)
    )
    return compare

def get_WY_and_season(ts):
    try:
        WY = np.where(ts.month.isin(range(1, 10)), ts.year, ts.year + 1)
        season_dict = {
            "Winter":ts.month.isin([10, 11, 12, 1, 2, 3]),
            "Spring":ts.month.isin([4, 5, 6]),
            "Summer":ts.month.isin([7, 8, 9]),
        }
        season = np.select(condlist=season_dict.values(), choicelist=season_dict.keys())
        WY_season = [s + " " + str(iwy) for s, iwy in zip(season, WY)]
    except AttributeError:
        WY = np.where(ts.dt.month.isin(range(1, 10)), ts.dt.year, ts.dt.year + 1)
        season_dict = {
            "Winter":ts.dt.month.isin([10, 11, 12, 1, 2, 3]),
            "Spring":ts.dt.month.isin([4, 5, 6]),
            "Summer":ts.dt.month.isin([7, 8, 9]),
        }
        season = np.select(condlist=season_dict.values(), choicelist=season_dict.keys())
        WY_season = [s + " " + str(iwy) for s, iwy in zip(season, WY)]
    return WY, season, WY_season

def make_timeseries_continuous(incomplete_df, tmin, tmax, freq):
    """take a timeseries with discontinuous time index and add in missing times. THIS IS NOT GAP FILLING -- all empty times will be populated with nans.
    tmin, tmax: the min and max time to extend the timeseries to
    freq: the time interval between datapoints (fed to pd.date_range)
    incomplete_df: the dataframe to make continuous. MUST HAVE A DATETIME INDEX"""

    ref_ts = pd.DataFrame(dict(TIMESTAMP=pd.date_range(tmin, tmax, freq="30min"))).set_index(incomplete_df.index.name)
    return incomplete_df.merge(ref_ts, left_index=True, right_index=True, how="right")

