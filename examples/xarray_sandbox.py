###############################################################################
######## lipkinb's amazing xarray sandbox #####################################
###############################################################################

import types
import numpy as np
import pandas as pd
import xarray as xr

import langbrainscore as lbs
from langbrainscore.utils.logging import log

log('lipkinb\'s amazing xarray sandbox', cmap='WARN', type='INFO')


def typeshapeinfo(ob, name='object') -> str:
    """
    Returns a string describing the type and shape of the object.
    must have a shape attribute.
    """
    bonus = ''
    try:
        bonus = f'(!!XARRAY) with dimensions "{ob.dims}"'
    except AttributeError:
        pass
    shape = ob.shape if hasattr(ob, 'shape') else 'NO_SHAPE_ATTRIBUTE'
    return f'"{name}" is an instance of {type(ob)} with shape "{shape}" {bonus}. '

sep = lambda: log('-'*64  , cmap='ERR', type=' ')


###############################################################################
#### generate stimuli


# sample relevant metadata
stimuli = np.array(["abc" for i in range(50)] + ["def" for i in range(50)])
is_abc = stimuli == "abc" # filter based on stimulus matches 'abc' 
subjects = np.repeat(np.arange(8), 12) # 8 subjects, 12 ROIs per subject
rois = np.tile(np.arange(12), 8) # 12 ROIs per subject
time = np.arange(0, 8, 0.1) # 80 timepoints .1 units apart
is_in_window = time < 1

# metadata IDs for Dataset
stimid = np.arange(stimuli.size) # stimulus ID
neuroid = np.arange(subjects.size) # neuroid ID
timeid = np.arange(time.size) # timepoint

log(typeshapeinfo(stimid, 'stimid') + f'and looks like {stimid[:20,]}...{stimid[-20:,]}')
log(typeshapeinfo(neuroid, 'neuroid') + f'and looks like {neuroid[:20,]}...{neuroid[-20:,]}')
log(typeshapeinfo(timeid, 'timeid') + f'and looks like {timeid[:20,]}...{timeid[-20:,]}')
sep()

log(typeshapeinfo(stimuli, 'stimuli') + f'and looks like {stimuli[:20,]}...{stimuli[-20:,]}')
log(typeshapeinfo(subjects, 'subjects') + f'and looks like {subjects[:20]}...')
log(typeshapeinfo(rois, 'rois') + f'and looks like {rois[:20]}...')
log(typeshapeinfo(time, 'time') + f'and looks like {time[:20]}...')
sep()


###############################################################################
#### create xarray objects using data and metadata


# measurement data DataArray
data_values_np = np.random.randn(stimid.size, neuroid.size, timeid.size)
log(f'Let\'s initialize a DataArray based on object: {typeshapeinfo(data_values_np, "data_values_np")}')
data_xr = xr.DataArray(
    data_values_np,
    dims=("stimid", "neuroid", "timeid"),
    coords={"stimid": stimid, "neuroid": neuroid, "timeid": timeid},
)
log(f'we obtained: {typeshapeinfo(data_xr, "data_xr")} which looks like')
print(f'{data_xr[:3, :3, :3]}...')

# stimid meta DataArray
stim_meta_pd = pd.DataFrame({"stimuli": stimuli, "is_abc": is_abc, "stim_length": [len(s) for s in stimuli]})
log(f'Let\'s initialize a DataArray based on object: {typeshapeinfo(stim_meta_pd, "stim_meta_pd")}')
stim_xr = xr.DataArray(
    stim_meta_pd,
    dims=("stimid", "stim_meta"),
    coords={"stimid": stimid, "stim_meta": stim_meta_pd.columns},
)
log(f'we obtained: {typeshapeinfo(stim_xr, "stim_xr")} which looks like')
print(f'{stim_xr[:4, :3,]}...')

# neuroid meta DataArray
neuro_meta_pd = pd.DataFrame({"subject": subjects, "roi": rois})
log(f'Let\'s initialize a DataArray based on object: {typeshapeinfo(neuro_meta_pd, "neuro_meta_pd")}')
neuro_xr = xr.DataArray(
    neuro_meta_pd,
    dims=("neuroid", *neuro_meta_pd.columns), #"neuro_meta"),
    coords={"neuroid": neuroid, **{col: neuro_meta_pd[col] for col in neuro_meta_pd.columns}, #"neuro_meta": neuro_meta_pd.columns
            },
)
log(f'we obtained: {typeshapeinfo(neuro_xr, "neuro_xr")} which looks like')
print(f'{neuro_xr[:4, :3,]}...')


# timeid meta DataArray
time_meta_pd = pd.DataFrame({"time": time, "is_in_window": is_in_window})
log(f'Let\'s initialize a DataArray based on object: {typeshapeinfo(time_meta_pd, "time_meta_pd")}')
time_xr = xr.DataArray(
    time_meta_pd,
    dims=("timeid", "time_meta"),
    coords={"timeid": timeid, "time_meta": time_meta_pd.columns},
)
log(f'we obtained: {typeshapeinfo(time_xr, "time_xr")} which looks like')
print(f'{time_xr[:4, :3,]}...')
sep()


###############################################################################
#### compile inidividual DataArrays into an xarray Dataset object


# complete xarray Dataset
log(f'Now creating a Dataset object using data_xr, stim_xr, neuro_xr, time_xr')
dataset_xr = xr.Dataset(
    {"data": data_xr, "stim": stim_xr, "neuro": neuro_xr, "time": time_xr}
)
log(f'we obtained: {typeshapeinfo(dataset_xr, "dataset_xr")} which looks like')
print(f'{dataset_xr}...')

# sample queries

# get corr matrix for subject 1 mean (over time) ROIs across stimuli
corr = (
    dataset_xr.sel(
        neuroid=dataset_xr.neuro.loc[
            (dataset_xr.neuro.loc[:, "subject"] == 1), :
        ].neuroid
    )
    .groupby("neuroid")
    .mean("timeid")
    .data.T.to_pandas()
    .corr()
    .values
)

# select time series for abc stims from subject 3 ROI 2 where time is_in_window
ts = dataset_xr.sel(
    neuroid=dataset_xr.neuro.loc[
        (dataset_xr.neuro.loc[:, "subject"] == 3)
        & (dataset_xr.neuro.loc[:, "roi"] == 2),
        :,
    ].neuroid,
    stimid=dataset_xr.stim.loc[(dataset_xr.stim.loc[:, "is_abc"] == True), :].stimid,
    timeid=dataset_xr.time.loc[
        (dataset_xr.time.loc[:, "is_in_window"] == True), :
    ].timeid,
).data.values.squeeze()
