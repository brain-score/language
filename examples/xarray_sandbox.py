###############################################################################
######## lipkinb's amazing xarray sandbox #####################################
###############################################################################


import numpy as np
import pandas as pd
import xarray as xr
​
# sample relevant metadata
stimuli = np.array(["abc" for i in range(50)] + ["def" for i in range(50)])
is_abc = stimuli == "abc"
subjects = np.repeat(np.arange(8), 12)
rois = np.tile(np.arange(12), 8)
time = np.arange(0, 8, 0.1)
is_in_window = time < 1
​
# metadata IDs for Dataset
stimid = np.arange(stimuli.size)
neuroid = np.arange(subjects.size)
timeid = np.arange(time.size)
​
# measurement data DataArray
data_values_np = np.random.randn(stimid.size, neuroid.size, timeid.size)
data_xr = xr.DataArray(
    data_values_np,
    dims=("stimid", "neuroid", "timeid"),
    coords={"stimid": stimid, "neuroid": neuroid, "timeid": timeid},
)
​
# stimid meta DataArray
stim_meta_pd = pd.DataFrame({"stimuli": stimuli, "is_abc": is_abc})
stim_xr = xr.DataArray(
    stim_meta_pd,
    dims=("stimid", "stim_meta"),
    coords={"stimid": stimid, "stim_meta": stim_meta_pd.columns},
)
​
# neuroid meta DataArray
neuro_meta_pd = pd.DataFrame({"subjects": subjects, "rois": rois})
neuro_xr = xr.DataArray(
    neuro_meta_pd,
    dims=("neuroid", "neuro_meta"),
    coords={"neuroid": neuroid, "neuro_meta": neuro_meta_pd.columns},
)
​
# timeid meta DataArray
time_meta_pd = pd.DataFrame({"time": time, "is_in_window": is_in_window})
time_xr = xr.DataArray(
    time_meta_pd,
    dims=("timeid", "time_meta"),
    coords={"timeid": timeid, "time_meta": time_meta_pd.columns},
)
​
# complete xarray Dataset
dataset_xr = xr.Dataset(
    {"data": data_xr, "stim": stim_xr, "neuro": neuro_xr, "time": time_xr}
)
​
# sample queries
​
# get corr matrix for subject 1 mean (over time) ROIs across stimuli
corr = (
    dataset_xr.sel(
        neuroid=dataset_xr.neuro.loc[
            (dataset_xr.neuro.loc[:, "subjects"] == 1), :
        ].neuroid
    )
    .groupby("neuroid")
    .mean("timeid")
    .data.T.to_pandas()
    .corr()
    .values
)
​
# select time series for abc stims from subject 3 ROI 2 where time is_in_window
ts = dataset_xr.sel(
    neuroid=dataset_xr.neuro.loc[
        (dataset_xr.neuro.loc[:, "subjects"] == 3)
        & (dataset_xr.neuro.loc[:, "rois"] == 2),
        :,
    ].neuroid,
    stimid=dataset_xr.stim.loc[(dataset_xr.stim.loc[:, "is_abc"] == True), :].stimid,
    timeid=dataset_xr.time.loc[
        (dataset_xr.time.loc[:, "is_in_window"] == True), :
    ].timeid,
).data.values.squeeze()
