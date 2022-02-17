import pickle as pkl
from collections import defaultdict

import IPython
import langbrainscore as lbs
import numpy as np
import pandas as pd
import xarray as xr
from langbrainscore.utils.logging import log


def package_mean_froi_neural_control_data():
    with open(
        "../data/dict_UID-853_SESSION-FED_20211013b_3T1_PL2017_FL-control_tr1_20220109.pkl",
        "rb",
    ) as f:
        mnc = pkl.load(f)
    mnc_rois = mnc["df_rois_normalized"]
    mnc_stims = mnc["stimset"]
    recorded_data = np.expand_dims(mnc_rois.values, 2)
    mnc_xr = xr.DataArray(
        recorded_data,
        dims=("sampleid", "neuroid", "timeid"),
        coords={
            "sampleid": np.arange(recorded_data.shape[0]),
            "neuroid": np.arange(recorded_data.shape[1]),
            "timeid": np.arange(recorded_data.shape[2]),
        },
    ).to_dataset(name="data")
    neuroid_features = defaultdict(list)
    for col in mnc_rois.columns:
        neuroid_features["froi"].append(col)
        neuroid_features["netw"].append(col.split("_")[0])
        if len(col.split("_")) >= 3:
            neuroid_features["hemi"].append(col.split("_")[1])
            neuroid_features["region"].append("_".join(col.split("_")[2:]))
        elif len(col.split("_")) == 2:
            neuroid_features["hemi"].append("LHRH")
            neuroid_features["region"].append(col.split("_")[1])
        else:
            raise NotImplementedError()
    for key, val in neuroid_features.items():
        mnc_xr = mnc_xr.assign_coords({key: ("neuroid", val)})
    for col in mnc_stims.columns:
        mnc_xr = mnc_xr.assign_coords({col: ("sampleid", mnc_stims[col])})
    mnc_xr = mnc_xr.rename({"sentence": "stimuli"})
    return mnc_xr


def main():
    mnc_xr = package_mean_froi_neural_control_data()
    mnc_dataset = lbs.dataset.Dataset(mnc_xr)
    log(f"stimuli: {mnc_dataset.stimuli.values}")
    brain_enc = lbs.encoder.BrainEncoder(mnc_dataset)
    brain_enc_mnc = brain_enc.encode(average_time=False)
    log(f"created brain-encoded data of shape: {brain_enc_mnc.dims}")
    ann_enc = lbs.encoder.HuggingFaceEncoder("gpt2")
    ann_enc_mnc = ann_enc.encode(mnc_dataset, context_dimension="stimuli")
    ann_enc_mnc = ann_enc_mnc.isel(neuroid=(ann_enc_mnc.layer == 10))
    log(f"created ann-encoded data of shape: {ann_enc_mnc.dims}")
    ridge_cv_mapping_split = lbs.mapping.Mapping(
        ann_enc_mnc, brain_enc_mnc, "ridge_cv", k_fold=5
    )
    k_fold_split = ridge_cv_mapping_split.construct_splits()
    met = lbs.metrics.Metric(lbs.metrics.pearson_r)
    brsc = lbs.BrainScore(ridge_cv_mapping_split, met, run=True)
    log(f"brainscore = {brsc}")
    IPython.embed()


if __name__ == "__main__":
    main()
