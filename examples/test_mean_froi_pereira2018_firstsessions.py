from collections import defaultdict

import IPython
import langbrainscore as lbs
import numpy as np
import pandas as pd
import xarray as xr
from langbrainscore.utils.logging import log
from pathlib import Path

def package_mean_froi_pereira2018_firstsess():
    mpf = pd.read_csv(f"{Path(__file__).parents[1] / 'data/Pereira_FirstSession_TrialEffectSizes_20220223.csv'}")
    mpf = mpf.sort_values(by=["UID", "Session", "Experiment", "Stim"])
    subj_xrs = []
    neuroidx = 0
    for uid in mpf.UID.unique():
        mpf_subj = mpf[mpf.UID == uid]
        sess_xrs = []
        for sess in mpf_subj.Session.unique():
            mpf_sess = mpf_subj[mpf_subj.Session == sess]
            roi_filt = [any(n in c for n in ["Lang", "MD"]) for c in mpf_sess.columns]
            mpf_rois = mpf_sess.iloc[:, roi_filt]
            data_array = np.expand_dims(mpf_rois.values, 2)
            sess_xr = xr.DataArray(
                data_array,
                dims=("sampleid", "neuroid", "timeid"),
                coords={
                    "sampleid": (
                        np.arange(0, 384)
                        if data_array.shape[0] == 384
                        else np.arange(384, 384 + 243)
                    ),
                    "neuroid": np.arange(neuroidx, neuroidx + data_array.shape[1]),
                    "timeid": np.arange(data_array.shape[2]),
                    "stimuli": ("sampleid", mpf_sess.Sentence.str.strip('"')),
                    "passage": ("sampleid", mpf_sess.Stim),
                    "experiment": ("sampleid", mpf_sess.Experiment),
                    "session": (
                        "neuroid",
                        [mpf_sess.Session.values[0]] * data_array.shape[1],
                    ),
                    "uid": ("neuroid", [mpf_sess.UID.values[0]] * data_array.shape[1]),
                    "roi": ("neuroid", mpf_rois.columns),
                },
            )
            sess_xrs.append(sess_xr)
        neuroidx += data_array.shape[1]
        subj_xr = xr.concat(sess_xrs, dim="sampleid")
        subj_xrs.append(subj_xr)
    mpf_xr = xr.concat(subj_xrs, dim="neuroid")
    mpf_xr = mpf_xr.assign_coords(
        {
            "stimuli": ("sampleid", mpf_xr.stimuli[0].values),
            "passage": ("sampleid", mpf_xr.passage[0].values),
            "experiment": ("sampleid", mpf_xr.experiment[0].values),
        }
    )
    return mpf_xr


def main():
    mpf_xr = package_mean_froi_pereira2018_firstsess()
    mpf_dataset = lbs.dataset.Dataset(mpf_xr.isel(neuroid=mpf_xr.roi=='Lang_LH_AntTemp'))
    log(f"stimuli: {mpf_dataset.stimuli.values}")
    brain_enc = lbs.encoder.BrainEncoder(mpf_dataset)
    brain_enc_mpf = brain_enc.encode()
    log(f"created brain-encoded data of shape: {brain_enc_mpf.dims}")
    ann_enc = lbs.encoder.HuggingFaceEncoder("distilgpt2")
    ann_enc_mpf = ann_enc.encode(mpf_dataset, context_dimension="stimuli")
    ann_enc_mpf = ann_enc_mpf.isel(neuroid=(ann_enc_mpf.layer == 4))
    log(f"created ann-encoded data of shape: {ann_enc_mpf.dims}")
    ridge_cv_mapping_split = lbs.mapping.Mapping(
        ann_enc_mpf, brain_enc_mpf, "ridge_cv", k_fold=2
    )
    # k_fold_split = ridge_cv_mapping_split.construct_splits()
    met = lbs.metrics.Metric(lbs.metrics.pearson_r)
    brsc = lbs.BrainScore(ridge_cv_mapping_split, met, run=True)
    log(f"brainscore = {brsc}")
    IPython.embed()


if __name__ == "__main__":
    main()
