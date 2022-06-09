########################################################################
# ######## Language brain-score example API usage notebook
########################################################################

"""Test case for testing ANN Encoder and Brain Encoder where the data is identical"""

import pdb
import random

import brainscore_language as lbs
import numpy as np
import pandas as pd
import xarray as xr
from brainscore_language.utils.logging import log
import matplotlib.pyplot as plt

import IPython

mapping = 'linear'

########################################################################
# ######## Create mock dataset
# #### outcome: a DataSet instance that contains the data we will
# ####          randomly generate and use to test LBS
########################################################################
log("." * 79, type="WARN")

# define the size and dimensionality of mock data to create
#num_stimuli = 627
num_stimuli = 62
num_neuroid = 10_000

# now randomly generate the mock data
log(f"creating mock data with {num_stimuli} stimuli and {num_neuroid} neuroids")
recorded_data = np.random.rand(num_stimuli, num_neuroid)

log(f"recorded_data[:5,:5] == {recorded_data[:5,:5]}, of shape {recorded_data.shape}")

# also define metadata because the DataSet class requires it
# for now, pretend like each subject contributes 100 neuroids worth of data per stimulus
recording_metadata = pd.DataFrame(
    dict(
        neuro_id=[i for i in range(num_neuroid)],
        roi=[i % 12 for i in range(num_neuroid)],
        subj_id=np.repeat(np.arange(10), 1000)[:num_neuroid],
    )
)

# create random string stimuli
stimuli = np.array(
    [
        "".join(random.sample("abcdefghijklmnopqrstuvwxiz aeiou aeiou" * 20, 10))
        for _ in recorded_data
    ]
)
experiment = stimuli.copy()
experiment[:] = 2
# experiment[:243] = 1
experiment[:24] = 1
experiment = experiment.astype(int)

passage = np.concatenate(
    # (np.repeat(np.arange(100), 3)[:243], np.repeat(np.arange(100), 4)[:384])
    (np.repeat(np.arange(100), 3)[:24], np.repeat(np.arange(100), 4)[:38])
)

passage_experiment = [f'passage-{p}_exp-{e}' for p, e in zip(passage, experiment)]

log(f"num. of stimuli generated: {len(stimuli)}, example: {stimuli[:4]} ...")

#recorded_data[:243, :1000] = np.nan
#recorded_data[243:, 1000:2000] = np.nan
recorded_data[:24, :1000] = np.nan
recorded_data[24:, 1000:2000] = np.nan
recorded_data = recorded_data.reshape((*recorded_data.shape, 1))

# generate xarray DataSet
xr_dataset = xr.DataArray(
    recorded_data,
    dims=("sampleid", "neuroid", "timeid"),
    coords={
        "sampleid": np.arange(recorded_data.shape[0]),
        "neuroid": np.arange(recorded_data.shape[1]),
        "timeid": np.arange(recorded_data.shape[2]),
        "stimuli": ("sampleid", stimuli),
        "sent_identifier": ("sampleid", [f'mycorpus.{i:0>5}' for i in range(num_stimuli)]),
        "experiment": ("sampleid", experiment),
        "passage_experiment": ("sampleid", passage_experiment),
        "passage": ("sampleid", passage),
        "subject": ("neuroid", recording_metadata["subj_id"]),
        "roi": ("neuroid", recording_metadata["roi"]),
    },
).to_dataset(name="data")

# instantiate a mock dataset object with associated neuroimaging recordings as well as metadata
mock_neuro_dataset = lbs.dataset.Dataset(xr_dataset)

log(f'stimuli: {mock_neuro_dataset.stimuli.values}')



########################################################################
# ######## Create mock brain encoder
# #### outcome: a BrainEncoder instance that implements .encode()
########################################################################
log("." * 79, type="WARN")

log("creating mock brain encoder")
mock_brain_encoder = lbs.encoder.BrainEncoder(mock_neuro_dataset)



########################################################################
# ######## Obtain brain encoder representation of mock data
# #### outcome: return value of .encode()
########################################################################
log("." * 79, type="WARN")

# expect to obtain data of shape 627 x 10_000
brain_encoded_data = mock_brain_encoder.encode(mock_neuro_dataset, 
                                               average_time = False)
log(f"created brain-encoded data of shape: {brain_encoded_data.dims}")




########################################################################
# ######## Create an ANN encoder using distilgpt2 model identifier
# #### outcome: a HuggingfaceEncoder instance that implements .encode()
########################################################################
log("." * 79, type="WARN")

log("creating an ANN encoder using distilgpt2")
distilgpt_encoder = lbs.encoder.HuggingFaceEncoder('distilgpt2')


########################################################################
# ######## Obtain ANN encoder representation of mock data
# #### outcome: return value of .encode()
########################################################################
log("." * 79, type="WARN")

# expect to obtain data of shape 627 x 10_000
ann_encoded_data = distilgpt_encoder.encode(mock_neuro_dataset, context_dimension='passage_experiment')
log(f"created ann-encoded data of shape: {ann_encoded_data.dims}")







# ANN_encoded_data = brain_encoded_data + np.random.rand(*brain_encoded_data.dims.values()) / 2
# pretend that an ANN outputted 768-dim vector for each of the 627 stimuli


# ann_encoded_data = ann_encoded_data.sel(neuroid=slice(0, 768)) # [:, :768]
ann_encoded_data.sel(neuroid=ann_encoded_data.layer==1)
log(f"created ANN-encoded data of shape: {ann_encoded_data.dims}, {ann_encoded_data.data.shape}")


########################################################################
# ######## Fit a mapping between two encodings
# #### outcome: return k [y_pred, y_test] arrays in case of k-fold CV,
# ####          else return single [y, y_hat] pair
########################################################################
log("." * 79, type="WARN")

log("fitting a mapping using ridge regression")

identity_mapping = lbs.mapping.IdentityMap(brain_encoded_data, brain_encoded_data,
                                           mapping_class='linear')
met = lbs.metrics.Metric(lbs.metrics.RSA().run)
brsc = lbs.BrainScore(identity_mapping, met, run=True)

def test_cv(encoderA, encoderB,
            mapping: str = "ridge",
            k_fold: int = 5,
            split_coord: str = None,
            strat_coord: str = None,
            output_y_pred=False):
    
    # Group KFold (splits at group borders; same group stays in the same split)
    cv_mapping_split = lbs.mapping.Mapping(encoderA, encoderB,
                                              mapping,
                                              k_fold=k_fold,
                                              split_coord=split_coord,
                                            strat_coord=strat_coord)

    met = lbs.metrics.Metric(lbs.metrics.pearson_r)
    brsc = lbs.BrainScore(cv_mapping_split, met, run=True)
    
    print(f"{mapping} {k_fold} fold CV: {brsc}")
    
    if output_y_pred:
        result = cv_mapping_split.fit()
        y_splits, y_hat_splits = result['test'], result['pred']

        pearson_rs = [
            np.corrcoef(y_hat_splits[0][0][:, i], y_splits[0][:, i].values.squeeze())
            for i in range(len(y_hat_splits[0][0][0]))
        ] # the y_hat_splits are structured in list of arrays, while the y_splits (real data) is in an xarray

        log(
            f"number of splits: {len(y_hat_splits)}, shape of split 0 y_hat: {y_hat_splits[0].shape}, shape of split 0 y_test: {y_splits[1].shape}"
        )

    return brsc
    
    
brsc = test_cv(encoderA=brain_encoded_data,
               encoderB=brain_encoded_data,
               mapping="ridge", k_fold=5,
               split_coord=None, strat_coord=None,
               output_y_pred=True)

# Sanity check plots
# plt.hist(np.squeeze(brsc.scores.values))
# plt.show()

log(f'brainscore = {brsc}')





IPython.embed()
exit()


########################################################################
# ######## Compute a distance metric between one encoding mapper to the
#           other encoding, and the true values of the target encoding
# #### outcome: return d scalars each corresponding to a neuroid of the
# ####          target encoding
########################################################################
log("." * 79, type="WARN")



########################################################################
# FIN.
########################################################################
log("." * 79, type="WARN")
log("finished.")
