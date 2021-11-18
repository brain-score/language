import numpy as np
import pandas as pd
import xarray as xr

# sample relevant metadata
stimuli = np.array(["abc" for i in range(50)] + ["def" for i in range(50)])
is_abc = stimuli == "abc"
subjects = np.repeat(np.arange(8), 12)
rois = np.tile(np.arange(12), 8)
time = np.arange(0, 8, 0.1)
is_in_window = time < 1

# metadata IDs for Dataset
stimid = np.arange(stimuli.size)
neuroid = np.arange(subjects.size)
timeid = np.arange(time.size)

# measurement data DataArray
data_values_np = np.random.randn(stimid.size, neuroid.size, timeid.size)
data_xr = xr.DataArray(
    data_values_np,
    dims=("stimid", "neuroid", "timeid"),
    coords={"stimid": stimid, "neuroid": neuroid, "timeid": timeid},
)

# stimid meta DataArray
stim_xr = xr.DataArray(
    stimid,
    dims=("stimid"),
    coords={
        "stimid": stimid,
        "stimuli": ("stimid", stimuli),
        "is_abc": ("stimid", is_abc),
    },
)

# neuroid meta DataArray
neuro_xr = xr.DataArray(
    neuroid,
    dims=("neuroid"),
    coords={
        "neuroid": neuroid,
        "subjects": ("neuroid", subjects),
        "rois": ("neuroid", rois),
    },
)

# timeid meta DataArray
time_xr = xr.DataArray(
    timeid,
    dims=("timeid"),
    coords={
        "timeid": timeid,
        "time": ("timeid", time),
        "is_in_window": ("timeid", is_in_window),
    },
)

# complete xarray Dataset
dataset_xr = xr.Dataset(
    {"data": data_xr, "stim": stim_xr, "neuro": neuro_xr, "time": time_xr}
)
