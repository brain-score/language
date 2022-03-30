from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import langbrainscore as lbs
from langbrainscore.utils.logging import log
from langbrainscore.utils.xarray import collapse_multidim_coord


def package_mean_froi_pereira2018_firstsess():
    mpf = pd.read_csv(
        f"{Path(__file__).parents[1] / 'data/Pereira_FirstSession_TrialEffectSizes_20220223.csv'}"
    )
    # mpf = pd.read_csv(f"{'../data/Pereira_FirstSession_TrialEffectSizes_20220223.csv'}")
    mpf = mpf.sort_values(by=["UID", "Session", "Experiment", "Stim"])
    subj_xrs = []
    neuroidx = 0
    for uid in tqdm(mpf.UID.unique()):
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
                    "stimulus": ("sampleid", mpf_sess.Sentence.str.strip('"')),
                    "passage": (
                        "sampleid",
                        list(map(lambda p_s: p_s.split("_")[0], mpf_sess.Stim)),
                    ),
                    "experiment": ("sampleid", mpf_sess.Experiment),
                    "session": (
                        "neuroid",
                        np.array(
                            [mpf_sess.Session.values[0]] * data_array.shape[1],
                            dtype=object,
                        ),
                    ),
                    "subject": (
                        "neuroid",
                        [mpf_sess.UID.values[0]] * data_array.shape[1],
                    ),
                    "roi": ("neuroid", mpf_rois.columns),
                },
            )
            sess_xrs.append(sess_xr)
        neuroidx += data_array.shape[1]
        subj_xr = xr.concat(sess_xrs, dim="sampleid")
        subj_xrs.append(subj_xr)
    mpf_xr = xr.concat(subj_xrs, dim="neuroid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "stimulus", "sampleid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "passage", "sampleid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "experiment", "sampleid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "session", "neuroid")
    return mpf_xr


def main():
    mpf_xr = package_mean_froi_pereira2018_firstsess()
    mpf_dataset = lbs.dataset.Dataset(
        mpf_xr.isel(neuroid=mpf_xr.roi == "Lang_LH_AntTemp")
    )

    log(f"stimuli: {mpf_dataset.stimuli.values}")
    mpf_dataset.to_cache('test_mpf_dataset_cache', cache_dir='./cache')
    mpf_dataset = lbs.dataset.Dataset.from_cache('test_mpf_dataset_cache', cache_dir='./cache')
    
    log(f"stimuli: {mpf_dataset.stimuli.values}")
    brain_enc = lbs.encoder.BrainEncoder()
    ann_enc = lbs.encoder.HuggingFaceEncoder("distilgpt2")
    brain_enc_mpf = brain_enc.encode(mpf_dataset)
    ann_enc_mpf = ann_enc.encode(mpf_dataset, context_dimension="passage")
    ann_enc_mpf = ann_enc_mpf.isel(neuroid=(ann_enc_mpf.layer == 4))
    log(f"created brain-encoded data of shape: {brain_enc_mpf.dims}")
    log(f"created ann-encoded data of shape: {ann_enc_mpf.dims}")
    pls_cv_kfold = lbs.mapping.LearnedMap("linpls", k_fold=5)
    pearson = lbs.metrics.Metric(lbs.metrics.PearsonR)
    brsc_pls_pearson = lbs.BrainScore(ann_enc_mpf, brain_enc_mpf, pls_cv_kfold, pearson)
    brsc_pls_pearson.score(score_split_coord="experiment")
    log(f"brainscore (pls, pearson) = {brsc_pls_pearson}")
    i_map = lbs.mapping.IdentityMap(nan_strategy="drop")
    cka = lbs.metrics.Metric(lbs.metrics.CKA)
    brsc_cka = lbs.BrainScore(ann_enc_mpf, brain_enc_mpf, i_map, cka)
    brsc_cka.score(score_split_coord="experiment")
    log(f"brainscore (cka) = {brsc_cka}")


if __name__ == "__main__":
    main()
